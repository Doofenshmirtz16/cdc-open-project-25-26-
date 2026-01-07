# Satellite Imagery-Based Property Valuation  
**Multimodal Machine Learning Project**

---

## Project Overview

This project aims to predict residential property prices by combining **structured housing attributes** with **satellite imagery–derived visual context**. The motivation is to capture not only intrinsic property features (e.g., size, quality) but also extrinsic neighborhood characteristics (e.g., greenery, roads, waterfront proximity) using satellite images.

The project explores **multimodal learning**, evaluates multiple fusion strategies, and emphasizes **engineering robustness, explainability, and honest performance analysis**.

---

## Objectives

- Build a strong **tabular baseline regression model** for house price prediction.
- Programmatically fetch **satellite images** using latitude–longitude coordinates.
- Extract visual features from satellite images using a **pretrained CNN**.
- Experiment with **multimodal fusion architectures** (tabular + image features).
- Compare unimodal vs multimodal performance using RMSE and R².
- Provide **visual explainability** using Grad-CAM.
- Generate a **final prediction file** for the test dataset.

---

## Dataset Description

### Tabular Data
- Source: King County house sales dataset
- Key features:
  - `price` (target)
  - `bedrooms`, `bathrooms`
  - `sqft_living`, `sqft_lot`
  - `grade`, `condition`, `view`
  - `waterfront`
  - `lat`, `long`

### Visual Data
- Satellite images fetched using latitude and longitude.
- Images represent neighborhood-level context rather than individual building details.

---

### Installation
```bash
pip install -r requirements.txt
```

---

## Methodology

### Phase 1: Exploratory Data Analysis (EDA)
- Distribution analysis of prices and key numerical features.
- Correlation analysis among tabular variables.
- Identification of skewness and outliers.

### Phase 2: Tabular Baseline Models
- Linear Regression
- Ridge Regression
- Random Forest Regressor

-> **Best-performing model:**  
**Random Forest (tabular only)**

---

### Phase 3: Satellite Image Acquisition
- Satellite images fetched programmatically for properties using coordinates.
- Images stored locally and validated for completeness.

---

### Phase 4: CNN Feature Extraction
- Pretrained **ResNet-18** used as a feature extractor.
- Final classification layer removed.
- Each image converted into a **512-dimensional embedding**.
- Computation performed on **CPU-only system** using transfer learning.

---

### Phase 5: Multimodal Fusion Experiments

The following fusion strategies were evaluated:

1. **Random Forest + raw CNN embeddings**
2. **MLP + raw CNN embeddings**
3. **PCA-reduced CNN embeddings + Ridge Regression**

#### Observations:
- High-dimensional CNN embeddings introduced noise.
- Multimodal models did **not outperform** the strong tabular baseline.
- Tree-based and neural fusion models showed degradation due to feature dominance and limited sample size.

**Key Insight:**  
> Multimodal learning does not guarantee improved performance unless visual features are task-aligned and carefully regularized.

---

### Phase 6: Explainability (Grad-CAM)

- Grad-CAM applied to the CNN to visualize important regions in satellite images.
- Heatmaps revealed attention on:
  - Water bodies
  - Green cover
  - Road networks
- This validated that satellite imagery captures meaningful **environmental context**, even if it did not improve numerical accuracy.

---

## Model Performance Summary

| Model | RMSE | R² |
|-----|-----|----|
| **Tabular Random Forest** | **~125,838** | **~0.87** |
| Multimodal (RF + CNN) | ~159,758 | ~0.78 |
| Multimodal (PCA + Ridge) | ~170,288 | ~0.74 |

The **tabular-only Random Forest** was selected for final predictions due to superior accuracy and stability.

---

## Project Structure

satellite-property-valuation/

│
├── data/
│ ├── raw/ # Original datasets
│ ├── processed/ # Cleaned tabular data
│ ├── images/ # Satellite images
│ └── image_embeddings.npy # CNN embeddings
│
├── notebooks/
│ ├── eda.ipynb
│ ├── preprocessing.ipynb
│ └── model_training.ipynb
│
├── src/
│ └── data_fetcher.py # Satellite image downloader
│
├── predictions.csv # Final prediction file (tabular model)
├── requirements.txt
└── README.md

---

## Final Prediction File

- File: `predictions.csv`
- Format: `id, predicted_price`

- Generated using the **tabular-only Random Forest model**, which achieved the best validation performance.

---

## Tech Stack

- **Data Handling:** Pandas, NumPy, GeoPandas
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Deep Learning:** PyTorch (CPU)
- **Image Processing:** PIL
- **Explainability:** Grad-CAM

---

## System Constraints

- CPU-only system (Intel i5, integrated graphics)
- No GPU acceleration
- Transfer learning used to remain computationally efficient

---

## Key Learnings

- Strong tabular features can outperform naive multimodal fusion.
- High-dimensional visual embeddings require careful integration.
- Model complexity must be justified by empirical gains.
- Explainability adds value even when accuracy does not improve.

---

## Conclusion

This project demonstrates a **complete, end-to-end multimodal ML pipeline**, from data acquisition to explainability. While satellite imagery provided meaningful qualitative insights, the final deployed model prioritizes **accuracy, robustness, and interpretability** by leveraging structured data.

The work highlights an important real-world lesson:
> **More data modalities do not always lead to better models — thoughtful integration matters.**

---

## Project Completed by:
**Sumit Sharma**
**IIT ROORKEE**

