# Plant Leaf Disease Classification using Semi-Supervised Learning

An end-to-end **Semi-Supervised Learning (SSL)** pipeline built with **TensorFlow/Keras**, **Scikit-Learn**, and **Streamlit** to classify **38 plant leaf disease classes** on the **PlantVillage** dataset (54,306 total images).

## Overview

In real-world agricultural settings, labeling every image can be costly and time-consuming. This project explores a **Teacher–Student Convolutional Neural Network (MobileNetV2 backbone)** with **pseudo-labeling** to leverage a small labeled subset alongside a much larger unlabeled pool.

## Dataset & Split

- **Dataset:** PlantVillage (38 classes, 54,306 total images)
- **Labeled Subset (20%):** 9,232 images
- **Unlabeled Pool (80%):** 36,928 images
- **Test Set:** 8,146 images

## Methodology & Results

1. **Teacher Model:** Trained a `MobileNetV2`-based CNN on the 20% labeled split, achieving **0.8879 (~88.79%)** test accuracy.
2. **Pseudo-Labeling:** Generated pseudo-labels on the unlabeled images using a **0.90 confidence threshold**, selecting **25,060** high-confidence images to combine with the labeled training set.
3. **Student Model:** Trained a Student CNN on the combined dataset, achieving **0.8673 (~86.73%)** test accuracy, and saved the trained model as `ssl_plant_disease_cnn.keras`.
4. **Interactive Web App:** Built a Streamlit application (`app.py`) that loads `ssl_plant_disease_cnn.keras` and `class_names.json` to predict leaf diseases from uploaded images along with confidence scores and Top 5 predictions.

## Project Structure

```text
plant-disease-ssl/
├── plant_disease_ssl.ipynb      # Training, pseudo-labeling, and evaluation notebook
├── app.py                       # Streamlit inference web application
├── ssl_plant_disease_cnn.keras  # Saved Student CNN model
├── class_names.json             # Mapping of 38 plant disease classes
├── .gitignore                   # Ignores local dataset (data/) and cache files
└── README.md                    # Project documentation
