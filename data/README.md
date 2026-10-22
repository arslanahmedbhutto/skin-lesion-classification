# Dataset

This project uses **DermaMNIST** from the [MedMNIST v2](https://medmnist.com/) collection. DermaMNIST is the [HAM10000](https://doi.org/10.7910/DVN/DBW86T) dermatoscopy dataset (10,015 images, 7 diagnostic categories) resized to 28×28 RGB, with official train / validation / test splits.

| Split | Images |
|-------|-------:|
| train | 7,007 |
| val   | 1,003 |
| test  | 2,005 |

## Download

Download `dermamnist.npz` (about 19 MB) into this folder:

```bash
curl -L -o data/dermamnist.npz "https://zenodo.org/records/10519652/files/dermamnist.npz?download=1"
```

Or with Python: `pip install medmnist`, then `medmnist.DermaMNIST(split="train", download=True, root="data")`.

## License and citation

DermaMNIST is released under **CC BY-NC 4.0**, the same license as HAM10000. If you use it, please cite:

- Yang, J. et al. *MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification.* Scientific Data, 2023.
- Tschandl, P., Rosendahl, C. & Kittler, H. *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.* Scientific Data, 2018.
