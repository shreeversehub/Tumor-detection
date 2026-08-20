# 🧠 Brain Tumor Detection using Deep Learning

A CNN-based Brain Tumor Detection system built using Python and TensorFlow/Keras.

This application allows users to upload an MRI scan and instantly classify it as tumorous or non-tumorous using a trained deep learning model.

## 🚀 Features

- Deep Learning-based classification
- Built with Convolutional Neural Networks (CNN)
- Classifies MRI scans instantly
- Image preprocessing (grayscale, normalization, augmentation)
- User-friendly application interface
- Assists in early tumor diagnosis

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib

## 📸 How It Works

1. Upload an MRI scan image.
2. Click the **Analyze** button.
3. The application processes the image using:
   - Grayscale conversion
   - Normalization
   - CNN-based prediction

### Example

Input:
```
mri_scan_045.jpg
```

Output:
```
Tumor Detected — Confidence: 94.2%
```

## 📂 Project Structure

```
Brain-Tumor-Detection/
│── dataset/
│── model/
│   └── cnn_model.h5
│── app.py
│── preprocess.py
│── train.py
│── requirements.txt
│── README.md
```

## ▶️ Run the Project

### Clone the Repository

```bash
git clone https://github.com/your-username/Brain-Tumor-Detection.git
```

### Navigate to Project Folder

```bash
cd Brain-Tumor-Detection
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run

```bash
python app.py
```

## 🎯 Learning Outcomes

This project helped me learn:

- Convolutional Neural Networks (CNN)
- Image Preprocessing Techniques
- Model Training and Evaluation
- TensorFlow/Keras Workflow
- Building AI-powered Applications
- Medical Image Classification

## 🔮 Future Improvements

- Multi-class tumor classification (glioma, meningioma, pituitary)
- Deploy as a web application
- Grad-CAM visualization for explainability
- Larger and more diverse dataset
- Improved model accuracy



