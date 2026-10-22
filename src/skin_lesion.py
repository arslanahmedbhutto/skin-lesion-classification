"""Shared code for the skin lesion classifier: classes, data loading, preprocessing and the CNN."""
import json
from pathlib import Path

import numpy as np

IMG_SIZE = 28
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "dermamnist.npz"
MODEL_PATH = REPO_ROOT / "model" / "skin_lesion_cnn.keras"
STATS_PATH = REPO_ROOT / "model" / "preprocessing.json"

# DermaMNIST label order
CLASSES = [
    ("akiec", "Actinic keratoses / intraepithelial carcinoma"),
    ("bcc", "Basal cell carcinoma"),
    ("bkl", "Benign keratosis-like lesions"),
    ("df", "Dermatofibroma"),
    ("mel", "Melanoma"),
    ("nv", "Melanocytic nevi"),
    ("vasc", "Vascular lesions"),
]
CODES = [c for c, _ in CLASSES]
NAMES = [n for _, n in CLASSES]


def load_split(split, path=DATA_PATH):
    """Return (images, labels) for 'train', 'val' or 'test'."""
    data = np.load(path, allow_pickle=False)
    return data[f"{split}_images"], data[f"{split}_labels"].ravel().astype(int)


def oversample(x, y, seed=42):
    """Randomly duplicate minority-class samples until every class matches the largest one."""
    rng = np.random.default_rng(seed)
    target = np.bincount(y).max()
    idx = np.concatenate([
        rng.choice(np.where(y == c)[0], target, replace=True) if np.sum(y == c) < target else np.where(y == c)[0]
        for c in np.unique(y)
    ])
    rng.shuffle(idx)
    return x[idx], y[idx]


def fit_stats(x):
    """Mean and standard deviation of the training pixels, used to standardise every image."""
    return {"mean": float(np.mean(x)), "std": float(np.std(x))}


def save_stats(stats, path=STATS_PATH):
    Path(path).write_text(json.dumps(stats, indent=2))


def load_stats(path=STATS_PATH):
    return json.loads(Path(path).read_text())


def preprocess(images, stats):
    """Standardise uint8 RGB images with the training-set statistics."""
    return (np.asarray(images, dtype="float32") - stats["mean"]) / stats["std"]


def prepare_image(pil_image):
    """Turn any PIL image into a 28x28 RGB uint8 array like the training data."""
    from PIL import Image
    img = pil_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    return np.asarray(img, dtype="uint8")


def class_weights(y):
    """Softened inverse-frequency weights (square root), so rare classes count more without overwhelming training."""
    counts = np.bincount(y)
    return {c: float(np.sqrt(counts.max() / n)) for c, n in enumerate(counts)}


def build_improved_model():
    """Improved CNN: augmentation, three conv blocks with batch norm, global pooling and dropout."""
    from tensorflow import keras
    from tensorflow.keras import layers

    def conv_block(x, filters):
        for _ in range(2):
            x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
        return layers.MaxPool2D(2)(x)

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    # Lesions have no natural orientation, so flips and rotations are label-preserving.
    x = layers.RandomFlip("horizontal_and_vertical")(inputs)
    x = layers.RandomRotation(0.25, fill_mode="reflect")(x)
    x = layers.RandomZoom(0.1, fill_mode="reflect")(x)
    for filters in (32, 64, 128):
        x = conv_block(x, filters)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(len(CLASSES), activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="improved_cnn")
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics=["accuracy"])
    return model


def build_model():
    """Baseline CNN: four convolution layers, two max-pooling layers and three dense layers."""
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.Conv2D(16, 3, activation="relu", padding="same"),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPool2D(2),
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPool2D(2, padding="same"),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(len(CLASSES), activation="softmax"),
    ])
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics=["accuracy"])
    return model
