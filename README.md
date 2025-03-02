# Human Emotions Detection

This repository implements **deep learning-based emotion detection** from facial expressions using **CNNs, ResNet, EfficientNet, and Vision Transformers (ViT)**. The project explores multiple architectures, **transfer learning**, **ensemble learning**, and **model quantization** to improve accuracy and optimize model deployment.

## Features
✅ **Dataset Handling** – Uses Kaggle API integration, TFRecords, and data augmentation (CutMix) for robust training.  
✅ **Deep Learning Architectures** – Implements **LeNet, ResNet34, EfficientNet, and Vision Transformers (ViT)** for emotion classification.  
✅ **Transfer Learning & Fine-Tuning** – Utilizes **EfficientNet and Hugging Face ViT** for pre-trained model adaptation.  
✅ **Ensembling Techniques** – Combines multiple models to improve classification performance.  
✅ **Explainability with Grad-CAM** – Visualizes important facial features influencing model predictions.  
✅ **Model Evaluation** – Uses confusion matrices and performance metrics to assess model accuracy.  
✅ **ONNX Model Export & Benchmarking** – Converts trained models to **ONNX format** and compares performance across different frameworks.  
✅ **Model Quantization** – Implements **Post Training Quantization (PTQ)** and **Quantization Aware Training (QAT)** in TensorFlow and ONNX to optimize inference for low-power devices.  
✅ **TFLite Runtime Deployment** – Supports TensorFlow Lite inference for mobile and edge computing.  

## How to Use
1. **Clone the repository** and install dependencies.
2. **Preprocess the dataset** using the provided augmentation and TFRecords scripts.
3. **Train the model** using CNNs, ResNet, EfficientNet, or ViTs.
4. **Evaluate the model** using confusion matrices and Grad-CAM visualization.
5. **Optimize model for deployment** using ONNX conversion, benchmarking, and quantization.

## Technologies Used
- **TensorFlow & Keras** – Deep learning framework for model training.
- **PyTorch & Hugging Face Transformers** – Used for Vision Transformers (ViT).
- **OpenCV & Matplotlib** – Used for data preprocessing and visualization.
- **ONNX & TensorRT** – Used for model conversion, benchmarking, and quantization.
- **Weights & Biases (WandB)** – Used for experiment tracking and hyperparameter tuning.

## References
This project is inspired by research in **facial expression recognition using deep learning** and is designed for **real-time emotion analysis applications**.

