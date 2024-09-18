# Medical Diagnosis using Deep Learning

## Overview

In response to the COVID-19 pandemic, this project focuses on the rapid and accurate diagnosis of respiratory conditions such as **Pneumonia and COVID-19** using deep learning techniques. Leveraging advancements in medical imaging and convolutional neural networks (CNNs), this system aims to automate the detection and classification of these conditions from chest X-ray images, improving patient management and virus containment.

## Dataset

Downloaded Covid and Pneumonia datasets from Kaggle and tored in Drive.

### Content

- **COVID**: 1626 images
- **NORMAL**: 1802 images
- **PNEUMONIA**: 1800 images

All images are preprocessed, resized to 256x256 pixels, and saved in PNG format.

### Inspiration

This project is inspired by the need to support the medical community in detecting and classifying COVID-19 and Pneumonia through automated methods.

### Acknowledgements & References

- [Eurorad](https://www.eurorad.org/)
- [Radiopaedia](https://radiopaedia.org/)
- [Coronacases](https://coronacases.org/)
- [P. Mooney. (2018). Chest X-Ray Images (Pneumonia), Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

## Objectives

1. Develop a deep learning model for classifying pneumonia, COVID-19, and normal chest X-ray images.
2. Preprocess and augment the dataset to enhance model performance.
3. Implement and optimize a convolutional neural network (CNN) architecture.
4. Evaluate the model's performance using appropriate metrics.
5. Validate the model's real-world applicability in clinical settings.

## Getting Started

### Prerequisites

- TensorFlow 2.13.0
- Matplotlib 3.6.2
- NumPy 1.24.2

Install the required packages using:

```bash
pip install tensorflow==2.13.0 matplotlib==3.6.2 numpy==1.24.2
