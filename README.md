# ComputerVisionProject
This project implements and adapts the Semantic Diversity Learning (SDL) framework for Zero-Shot Multi-Label Classification (ZSMLC) on the MIRFlickr25K dataset. The model is based on a TResNet-M backbone and uses FastText embeddings to align image features with semantic label representations. The objective is to predict multiple labels for each image, including unseen labels (zero-shot), leveraging semantic relationships between labels.

## Overview

This project extends the Semantic Diversity Learning (SDL) framework to the MIRFlickr25K dataset, which contains 25,000 images annotated with 38 unique labels. The model is evaluated under two settings:
- Zero-Shot Learning (ZSL): Predicts only unseen labels.
- Generalized Zero-Shot Learning (GZSL): Predicts across both seen and unseen labels.

## Key features of this implementation:
- Uses TResNet-M as the visual feature extractor.
- Integrates FastText word embeddings (300-dimensional) for label representation.
- Applies semantic diversity weighting to emphasize diverse label co-occurrences.
- Utilizes hard negative mining during training for better discriminative learning.

## Dataset
MIRFlickr25K: A dataset of 25,000 Flickr images with user-generated multi-label annotations.
Annotations: Processed from raw text files, resulting in a structured CSV file (mirflickr25k_annotations_clean.csv).

The dataset is split as follows:
- Seen labels: 30 labels (80% of total labels).
- Unseen labels: 8 labels (20% of total labels).
- Images used for training contain only seen labels.

Evaluation includes:
- GZSL: Images with both seen and unseen labels.
- ZSL: Predicting only unseen labels.

## How to Run

### 1. Download the Dataset:
  - MIRFlickr25K Images and Annotations:
  - Download the dataset from the following links:
    - Images : https://arc.net/l/quote/tutuzmsd
    - Annotations : https://arc.net/l/quote/qtngzslx
  - Extract the images and annotation text files into separate folders and place into project directory :
    - /path/to/mirflickr/ (for images)
    - /path/to/mirflickr_annotations/ (for annotations)

### 2. Generate Clean Annotations:
  - Open the script and run the first code block to process annotations and create the
    file: mirflickr25k_annotations_clean.csv

### 3. Download FastText Embeddings:
  - Download wiki-news-300d-1M.vec from FastText: https://fasttext.cc/docs/en/english-vectors.html
  - Place it in the project directory.

### 4. Set File Paths:
- Update the following paths in the script:
  - annotations_dir (path to annotation text files)
  - image_dir (path to MIRFlickr images)
  - CSV_PATH (path to the cleaned annotation CSV)
  - fasttext_path (path to FastText embedding file)

### 5. Install Dependencies:
```
pip install torch torchvision timm pandas numpy scikit-learn pillow matplotlib
```
### 6. Train the Model:
- Run all cells in the script (or notebook) to:
  - Train the model on seen labels.
  - Evaluate on Generalized Zero-Shot (GZSL) and Pure Zero-Shot (ZSL) tasks.
  - Visualize sample predictions.

### 7. Outputs:
- Model checkpoints saved as: CV_best_model.pth
- Evaluation results: mAP, Precision@K, Recall@K, F1@K
- Visual sample predictions displayed at the end.
