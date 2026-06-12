---
title: Skin Cancer Classifier
emoji: 📈
colorFrom: yellow
colorTo: gray
sdk: gradio
sdk_version: "6.10.0"
app_file: app.py
pinned: false
license: mit
---

# Skin Cancer Classifier — ViT-Large

Fine-tuned **google/vit-large-patch16-224** on the HAM10000 dataset (10,000+ dermoscopy images, 7 classes).

**Results:** 92.74% Accuracy · 92.60% Weighted F1

## Classes
| Code | Full Name |
|------|-----------|
| akiec | Actinic Keratoses |
| bcc | Basal Cell Carcinoma |
| bkl | Benign Keratosis |
| df | Dermatofibroma |
| mel | Melanoma |
| nv | Melanocytic Nevi |
| vasc | Vascular Lesions |

## Validation Gate
Non-skin images are automatically rejected by a CLIP-based validation gate — the model only classifies dermoscopy/skin lesion images.

> ⚠️ This is a portfolio/learning project — not a medical diagnostic tool.
