# Automated Multi-Modal X-Ray Fracture & Crack Reporting System

Academic TensorFlow/Keras project for fracture/crack decision support from X-ray images and structured patient factors.

## Files

- `Automated_Multi_Modal_XRay_Fracture_Reporting_Colab.ipynb` - Colab-ready full pipeline notebook: automatic dataset download, audit, training, evaluation, Grad-CAM, reporting, and export.
- `streamlit_app.py` - Streamlit inference app. It only loads `fracture_model.keras`; it does not train.
- `requirements.txt` - Streamlit Cloud dependencies.
- `.gitignore` - Keeps downloaded datasets out of GitHub.

## Colab Workflow

1. Open `Automated_Multi_Modal_XRay_Fracture_Reporting_Colab.ipynb` in Google Colab.
2. Set runtime to GPU.
3. Run all cells. The notebook downloads FracAtlas automatically with `kagglehub`, trains a CNN multi-input model, evaluates it, creates Grad-CAM heatmaps, and saves `fracture_model.keras`.
4. Download `fracture_model.keras` and place it beside `streamlit_app.py`.

The notebook also includes stronger academic/project sections:

- `pycocotools` COCO mask loading and mask overlay preview.
- `train/val/test` folder split with `fractured` and `non_fractured` subfolders.
- Keras EfficientNet CNN backbone by default. It does not use PyTorch or Vision Transformers.
- Dataset audit, image validation, class weighting, ROC/PR curves, error analysis, threshold metadata, and deployment bundling.

The local file `Data_Entry_2017_v2020 (1).csv` is NIH ChestX-ray14 metadata, not FracAtlas. The notebook includes an optional inspection cell for it, but it is not required for the fracture model.

### Automated Multi-Modal X-Ray Fracture & Crack Reporting System

## Full Documentation

---

## 1. PROJECT OVERVIEW

**Project type:** Deep Learning course project + Streamlit Cloud deployment prototype

**Main objective:** Detect X‑ray fracture/crack suspicion, visualise suspicious evidence with Grad‑CAM, and generate an interpretable risk report that fuses image evidence with patient clinical factors.

**Key design principles:**

- CNN‑only vision model – no Vision Transformer or transformer‑based architecture is used.
- TensorFlow / Keras is the primary framework for training, saving, and inference.
- FracAtlas is downloaded automatically inside the notebook with `kagglehub`; no dataset files are embedded in the repository.
- The final trained model is saved as `fracture_model.keras`; the Streamlit app loads this file for inference only.
- An in‑notebook Gradio GUI is provided for Colab demonstration after training.

**Clinical safety note:** This is an academic decision‑support prototype. It is **not** a medical device and must not replace radiologist or physician review.

**Reference:**  
Abedeen et al., *FracAtlas: A Dataset for Fracture Classification, Localization and Segmentation of Musculoskeletal Radiographs*. Scientific Data, 2023.  
[https://doi.org/10.1038/s41597‑023‑02432‑4](https://doi.org/10.1038/s41597-023-02432-4)

---

## 2. RUNTIME ENVIRONMENT AND SETUP

The notebook is designed for **Google Colab GPU runtime**. All dependencies are installed inside the notebook; no manual file upload is required.

- **Python version:** 3.12.13
- **TensorFlow version:** 2.20.0
- **GPU:** PhysicalDevice (GPU:0) detected

**Core libraries installed:**

- TensorFlow / Keras (deep learning)
- OpenCV (`opencv-python-headless`) – image processing
- Seaborn, Matplotlib – visualisation
- scikit‑learn – metrics, train/test split
- `pycocotools` – COCO annotation handling
- Gradio – in‑notebook GUI

Reproducibility is enforced with a fixed random seed (`SEED = 42`).

---

## 3. PROJECT CONFIGURATION

Configuration constants control runtime cost and model strength.

| Parameter                        | Value                                                                                     | Description                                               |
| -------------------------------- | ----------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| `IMG_SIZE`                       | (224, 224)                                                                                | Input image dimensions                                    |
| `BATCH_SIZE`                     | 32                                                                                        | Training batch size                                       |
| `CNN_BACKBONE_NAME`              | `DenseNet121`                                                                             | Default CNN backbone (can be changed)                     |
| `SUPPORTED_BACKBONES`            | EfficientNetB0, EfficientNetB3, DenseNet121, ResNet50, MobileNetV2, InceptionV3, Xception | All available CNN backbones                               |
| `EPOCHS_HEAD`                    | 10                                                                                        | Epochs for frozen backbone training                       |
| `EPOCHS_FINE`                    | 8                                                                                         | Epochs for fine‑tuning                                    |
| `FINE_TUNE_LAST_N_LAYERS`        | 30                                                                                        | Number of backbone layers to unfreeze                     |
| `MAX_IMAGES_PER_CLASS`           | `None`                                                                                    | Use full dataset (`None`); set to e.g. 400 for quick runs |
| `RUN_BACKBONE_COMPARISON`        | `False`                                                                                   | Optional fast comparison of all backbones                 |
| `FINAL_BACKBONE_FROM_COMPARISON` | `True`                                                                                    | If comparison runs, auto‑select best backbone             |
| `COMPARISON_IMAGES_PER_CLASS`    | 250                                                                                       | Subset size for backbone comparison                       |
| `COMPARISON_EPOCHS`              | 2                                                                                         | Epochs for backbone comparison                            |

If a quick test is desired, set `MAX_IMAGES_PER_CLASS = 400`. For the final academic submission, use `None` (full dataset).

---

## 4. DATA SOURCE: FRACATLAS

The **FracAtlas** dataset is automatically downloaded via `kagglehub`:

```
kagglehub.dataset_download('mahmudulhasantasin/fracatlas-original-dataset')
```

- **Raw files:** 12,258 files (∼327 MB)
- Contains X‑ray images and COCO‑style fracture masks.
- The notebook constructs a binary classification dataset:
  - **Class 0:** Normal / Non‑fractured
  - **Class 1:** Fracture / Crack

### 4.1 Label Construction

Labels are inferred from folder names (`Fractured`, `Non_fractured`) and, if necessary, from CSV metadata. After robust validation with PIL and TensorFlow decoders, the final dataset contains:

| Class                  | Count     |
| ---------------------- | --------- |
| Normal / Non‑fractured | 3,307     |
| Fracture / Crack       | 717       |
| **Total valid images** | **4,024** |

59 images were removed due to validity errors (corrupt or truncated files).

### 4.2 COCO Annotations

FracAtlas includes COCO‑style fracture masks. A preview overlay is generated for dataset understanding, but the final model is a classifier with Grad‑CAM localisation (not a segmentation model).

---

## 5. SYNTHETIC PATIENT TABULAR DATA

Because FracAtlas does not provide patient history fields, **synthetic structured data** is generated with clinically plausible relationships to create a multi‑modal dataset.

**17 tabular features** are engineered:

| Feature                                                                                                    | Type    | Description                           |
| ---------------------------------------------------------------------------------------------------------- | ------- | ------------------------------------- |
| `age_norm`                                                                                                 | float   | Age normalised to [0,1]               |
| `gender_male`                                                                                              | binary  | 1 = Male, 0 = Female                  |
| `diabetes`                                                                                                 | binary  |                                       |
| `osteoporosis`                                                                                             | binary  |                                       |
| `smoking`                                                                                                  | binary  |                                       |
| `previous_fracture`                                                                                        | binary  |                                       |
| `pain_norm`                                                                                                | float   | Pain level (1‑10) normalised to [0,1] |
| `injury_arm`, `injury_leg`, `injury_spine`, `injury_rib`, `injury_hip`                                     | one‑hot | Injury location                       |
| `treatment_cast`, `treatment_surgery`, `treatment_physiotherapy`, `treatment_medication`, `treatment_none` | one‑hot | Treatment history (derived from text) |

Synthetic data generation uses rules: older age and fracture label increase the probability of osteoporosis and pain; positive label increases the chance of previous fracture, etc. Patient name is stored but **never used as a model feature** – it appears only in report text.

---

## 6. TRAIN / VALIDATION / TEST SPLIT

The dataset is split **stratified** to preserve class proportions:

| Split      | Total | Normal | Fracture |
| ---------- | ----- | ------ | -------- |
| Train      | 2,816 | 2,314  | 502      |
| Validation | 604   | 496    | 108      |
| Test       | 604   | 497    | 107      |

**Class weights** are computed from the training fold to counter imbalance:

- Class 0 (Normal): 0.608
- Class 1 (Fracture): 2.805

---

## 7. DATA PIPELINE (TENSORFLOW)

A `tf.data` pipeline decodes images, resizes to 224×224, and pairs them with the 17‑dimensional patient vector. Inputs are returned as a dictionary:

```python
{'xray_image': (batch, 224, 224, 3), 'patient_features': (batch, 17)}
```

Data augmentation (applied only during training):

- Random horizontal flip
- Random rotation (±0.04 rad)
- Random zoom (0.08)
- Random contrast (0.10)

The pipeline uses `AUTOTUNE` for parallel data loading and prefetching.

---

## 8. MODEL ARCHITECTURE

The model is a **multi‑modal CNN** that fuses an image branch with a dense patient‑feature branch.

### 8.1 Image Branch

- **Backbone:** Pretrained CNN (default `DenseNet121`, ImageNet weights)
- Preprocessing: rescaling + ImageNet normalisation (mean/std)
- **Feature extraction:** `last_conv_features` layer (for Grad‑CAM)
- Global Average Pooling → Dropout(0.3) → Dense(128, relu) → **image embedding**

### 8.2 Patient Feature Branch

- BatchNormalisation
- Dense(64, relu) → Dropout(0.2)
- Dense(32, relu) → **patient embedding**

### 8.3 Fusion & Decision

- Concatenate(image_embedding, patient_embedding)
- Dense(128, relu) with L2 regularisation → Dropout(0.35)
- Dense(64, relu)
- Dense(1, sigmoid) → **fracture probability**

**Total trainable parameters:** ~163 k (out of 7.2 M total) – the majority belong to the frozen backbone.

All supported backbones (EfficientNet, DenseNet, ResNet, MobileNet, Inception, Xception) follow the same architecture pattern. The `last_conv_features` activation layer is present in every variant, enabling Grad‑CAM for any backbone.

### 8.4 Optional Backbone Comparison

If `RUN_BACKBONE_COMPARISON = True`, a small subset (250 images per class) is used to train each backbone for 2 epochs. Validation AUC is compared, and the best backbone can be automatically selected for the final full training. By default this comparison is disabled to save runtime.

---

## 9. TRAINING STRATEGY

### 9.1 Phase 1: Frozen Backbone

The pretrained CNN is frozen; only the classifier head and patient‑feature branch are trained.

- **Optimizer:** Adam (learning rate 1e‑3)
- **Loss:** Binary crossentropy
- **Class weights** applied
- **Callbacks:**
  - `ModelCheckpoint` (monitor `val_auc`, save best only)
  - `EarlyStopping` (patience 4, restore best weights)
  - `ReduceLROnPlateau` (factor 0.3, patience 2)

**Best epoch (Phase 1):** validation AUC = **0.9634** (epoch 8, early stop at epoch 10)

### 9.2 Phase 2: Fine‑Tuning

The last 30 layers of the backbone are unfrozen; the model is trained with a very low learning rate (1e‑5) to adapt X‑ray‑specific textures.

- Fine‑tuning continued for up to 8 additional epochs (epochs 11‑18)
- Early stopping triggered after epoch 14 (no improvement in val_auc)
- Best model checkpoint (from Phase 1) is restored after fine‑tuning

**Final best validation AUC:** 0.9634

The best model is saved as `fracture_model.keras` (∼30.2 MB).

---

## 10. EVALUATION

### 10.1 Threshold Selection

The **decision threshold** is optimised on the validation set by maximising the F1 score.  
**Selected threshold:** 0.667

### 10.2 Test Set Performance

| Metric               | Value  |
| -------------------- | ------ |
| Accuracy             | 0.9172 |
| Precision (Fracture) | 0.835  |
| Recall (Fracture)    | 0.664  |
| F1‑score (Fracture)  | 0.740  |
| ROC‑AUC              | 0.9482 |
| Average Precision    | 0.8711 |

**Confusion Matrix (Test set, threshold = 0.667):**

|                   | Predicted Normal | Predicted Fracture |
| ----------------- | ---------------- | ------------------ |
| **True Normal**   | 483              | 14                 |
| **True Fracture** | 36               | 71                 |

The model achieves high overall accuracy but exhibits a moderate false‑negative rate (36 missed fractures out of 107). This is clinically undesirable, highlighting the need for cautious use.

### 10.3 Error Analysis

The most confident mistakes (both false positives and false negatives) are examined. Confident false negatives (model strongly predicts “Normal” despite a true fracture) are especially important to discuss as residual risk.

---

## 11. GRAD‑CAM LOCALISATION

Grad‑CAM highlights the convolutional image regions that most influence fracture probability. It is computed using the `last_conv_features` layer of the model.

Procedure:

1. Forward pass to obtain feature maps and prediction.
2. Compute gradients of the fracture probability w.r.t. the feature maps.
3. Average gradients globally, weight feature maps, and apply ReLU.
4. Resize heatmap, apply colormap, and overlay on the original image.

This visual explanation is intended as **explanatory evidence**, not a diagnostic segmentation mask. Heatmaps are displayed for sample test images.

---

## 12. RISK STRATIFICATION AND AUTOMATED REPORT

A **transparent scoring rule** fuses CNN probability with patient risk factors to produce a clinical risk level.

**Clinical risk points** (max 7):

- Age ≥ 65: +1
- Diabetes: +1
- Osteoporosis: +1
- Smoking: +1
- Previous fracture: +1
- Pain level ≥ 8: +1
- Injury location Hip or Spine: +1

**Risk score** = 100 × (0.65 × probability + 0.35 × (clinical_points / 7))

| Risk Level   | Criteria                                  |
| ------------ | ----------------------------------------- |
| **Critical** | risk_score ≥ 80 **or** probability ≥ 0.90 |
| **High**     | risk_score ≥ 60 **or** probability ≥ 0.70 |
| **Moderate** | risk_score ≥ 35 **or** probability ≥ 0.45 |
| **Low**      | otherwise                                 |

An example generated report includes:

- Patient demographics and history
- Model finding (Fracture/Crack suspected or not)
- Probability, threshold, clinical risk level, and risk score
- Mandatory clinical safety disclaimer

This approach makes the system more auditable: the CNN outputs probability, while clinical factors adjust the risk level without modifying the model.

---

## 13. INFERENCE GUI (GRADIO COLAB DEMO)

An in‑notebook Gradio interface is provided for academic demonstration:

- Accepts an uploaded X‑ray image and patient details (age, gender, injury location, pain level, medical history).
- Displays:
  - Fracture/crack probability
  - Clinical risk level and score
  - Grad‑CAM overlay
  - Full textual report (also saved as a text file)
- The saved `fracture_model.keras` is loaded; no retraining is needed.

This GUI is for demonstration within Colab; the production deployment separates the Streamlit app for cloud access.

---

## 14. REPRODUCIBILITY AND METADATA

After training, the following artifacts are saved for deployment:

| Artifact                 | Description                                                                               |
| ------------------------ | ----------------------------------------------------------------------------------------- |
| `fracture_model.keras`   | Trained multi‑modal model (30.2 MB)                                                       |
| `training_metadata.json` | Includes backbone, threshold, feature names, performance metrics, and clinical disclaimer |

The metadata ensures that the Streamlit inference app can deterministically reconstruct the preprocessing and threshold.

---

## 15. DEPLOYMENT: STREAMLIT CLOUD PROTOTYPE

The project includes a separate Streamlit application (not shown in the notebook) that:

- Loads `fracture_model.keras` and `training_metadata.json`
- Offers a user‑friendly interface for uploading an X‑ray and entering patient data
- Displays probability, risk level, Grad‑CAM overlay, and a printable report

This keeps training and inference strictly separated: the notebook is for model development, while Streamlit serves the final prototype.

---

## 16. MODEL INTERPRETATION AND LIMITATIONS

- **Strengths:**
  - Multi‑modal fusion improves robustness.
  - Grad‑CAM provides interpretable localisation.
  - Risk scoring is transparent, not a black box.
  - Multiple CNN backbones are supported; DenseNet121 gave the best validation AUC in this run.
- **Limitations:**
  - Synthetic patient data approximates reality but cannot replace real clinical records.
  - Moderate recall on fracture class (0.66) means some fractures may be missed.
  - The model was trained on a single dataset (FracAtlas); generalisation to other populations is not guaranteed.
  - This is an academic prototype **not approved for clinical use**.

---

## 17. FILES AND ARTIFACTS PRODUCED BY THE NOTEBOOK

| File                                 | Description                                     |
| ------------------------------------ | ----------------------------------------------- |
| `fracture_model.keras`               | Final trained multi‑modal model                 |
| `training_metadata.json`             | Model metadata and threshold                    |
| `keras_folder_split/`                | Symlinked train/val/test folders for inspection |
| `gradio_fracture_support_report.txt` | Example report generated by Gradio GUI          |

The notebook is designed to be run top‑to‑bottom in Google Colab with a GPU runtime, producing all necessary deployment artifacts.

---

## 18. CONCLUSION

The **Automated Multi‑Modal X‑Ray Fracture & Crack Reporting System** demonstrates a complete deep learning pipeline:

- Automatic dataset acquisition and cleaning
- Binary classification of musculoskeletal X‑rays using a CNN
- Fusion with structured patient data for improved risk assessment
- Interpretable explanations via Grad‑CAM
- A transparent, rule‑based clinical risk score
- Ready‑to‑deploy model and metadata

This work fulfills the requirements of a deep learning course project and provides a foundation for further research in medical decision support.
