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

## Streamlit Cloud

Push these files to GitHub:

- `streamlit_app.py`
- `requirements.txt`
- `fracture_model.keras`

Do not push downloaded dataset folders.

Then create a Streamlit Cloud app with:

```bash
streamlit run streamlit_app.py
```

## Clinical Safety

This project is for educational decision support only. It is not a standalone diagnostic device and must not replace radiologist or physician review.
