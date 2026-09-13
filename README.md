# CNN-GNN for Interpretable Chest X-ray Diagnosis

Code accompanying the paper *"A feature-driven graph neural network for learning
interpretable disease affinities in multi-class chest X-ray diagnosis."*

## Notebooks (run in this order)

1. **`Untitled7.ipynb`** — trains the proposed CNN-GNN model from scratch and saves
   the checkpoint (`cnn_gnn_final.pth`) used by all downstream notebooks.
2. **`NB1_Eval_GradCAM_Calibration_(1).ipynb`** — loads the trained checkpoint,
   evaluates it on the held-out test set, generates Grad-CAM visualizations, and
   computes the temperature-calibrated disease affinity matrix.
3. **`NB2_Ablation_Part1_(1).ipynb`** — trains ablation variants A0 (CNN-only),
   A1 (α = 0.5), A2 (α = 0.6).
4. **`NB3_Ablation_Part2_Baselines_(1).ipynb`** — trains A4 (α = 0.8), A5 (relation
   matrix frozen at 0), and the DenseNet121 baseline; begins ViT-B/16 training.
5. **`NB3_Ablation_Part3_Baselines_(1).ipynb`** — completes ViT-B/16 baseline training.
6. **`NB4_Aggregate_McNemar_FinalTables_(1).ipynb`** — aggregates all saved results,
   runs continuity-corrected McNemar's tests, and produces the final paper tables.

## Requirements

- Google Colab with a T4-class GPU (or equivalent local CUDA GPU).
- Python 3 with PyTorch, torchvision, scikit-learn, pandas, matplotlib, seaborn, scipy.
- A Google Drive folder `MyDrive/Dataset/` structured as:
  ```
  Dataset/
    train/{Cardiomegaly,Covid-19,Normal,Pneumonia,Pneumothorax,Tuberculosis}/*.png
    valid/{...same six class folders...}/*.png
    test/{...same six class folders...}/*.png
  ```
- An output folder `MyDrive/CNN_GNN_Results/` (created automatically on first run).

## Data

All images are drawn from public datasets — PadChest, the COVID-19 Radiography
Database, the SIIM-ACR Pneumothorax Segmentation Dataset, and the Shenzhen
Tuberculosis dataset. See the paper's Data Availability Statement for links.
No dataset files are included in this repository.

## Notes on notebook diagnostics

Each notebook prints a "filename-collision check" while loading the dataset.
This checks only whether filename strings repeat across splits (e.g. generic
names like `1.png` reused in different class folders) — it is **not** a check
for real patient- or image-level leakage, and nonzero counts here are expected
and benign. The actual leakage safeguard reported in the paper is a separate
full-file MD5 content hash scan, described in the manuscript's Limitations
section.

## Reproducibility

Random seed 42 is fixed across all notebooks (`random`, `numpy`, `torch`,
`torch.cuda`) with `torch.backends.cudnn.deterministic = True`.
