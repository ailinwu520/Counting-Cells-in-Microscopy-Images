# Counting Cells in Microscopy Images

Segments and counts cells in grayscale microscopy images using a U-Net
trained to predict per-pixel cell masks; cell counts are obtained by running
connected-component labeling on the predicted mask.

Developed for the Kaggle competition:
[Counting Cells in Microscopy Images (2024)](https://www.kaggle.com/competitions/counting-cells-in-microscopy-images-2024/overview)

See [`FinalProjectReport.pdf`](./FinalProjectReport.pdf) for the full write-up.

## Approach
1. **Segmentation** — a U-Net (`AWFinalProject.ipynb`) takes a 128x128
   grayscale image and predicts a per-pixel binary mask of cell regions,
   trained with `BCEWithLogitsLoss`.
2. **Counting** — the predicted mask is thresholded and passed through
   `cv2.connectedComponents` to count distinct cell regions.
3. **Evaluation** — mean absolute error (MAE) between predicted and true
   cell counts on a held-out validation split.

**Validation MAE: ~1.49 cells/image.**

Known limitation: cells that touch or overlap in the predicted mask are
merged into a single connected component, undercounting in dense regions.

## Setup
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Get the competition data (`train_data.npz`, `test_images.npz`) from
   Kaggle and place them in a `data/` directory in the project root, or set
   the `DATA_DIR` environment variable to their location.
3. Open and run `AWFinalProject.ipynb` top to bottom. (The notebook was
   developed on Google Colab with a GPU runtime; it also runs locally,
   falling back to CPU if no GPU is available.)

## Files
- `AWFinalProject.ipynb` — data loading, U-Net model, training, and
  evaluation.
- `FinalProjectReport.pdf` — write-up of the approach and results.
