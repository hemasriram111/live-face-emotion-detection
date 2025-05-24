# Emotion Detection Using CNN and Webcam

This project implements a real-time facial emotion detection system using a Convolutional Neural Network (CNN) built with TensorFlow and Keras. The model is trained to recognize seven emotions (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral) from facial images and uses OpenCV for live webcam-based emotion detection.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Real-Time Emotion Detection](#real-time-emotion-detection)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
The project consists of two main components:
1. **Training a CNN Model**: A deep learning model is trained on a dataset of facial images to classify seven emotions.
2. **Real-Time Detection**: A live webcam feed is used to detect faces and predict emotions in real-time using the trained model.

## Features
- Trained CNN model to classify seven emotions.
- Real-time face detection and emotion prediction using a webcam.
- Data augmentation for robust model training.
- Early stopping to prevent overfitting.
- User-friendly visualization of detected emotions on the webcam feed.

## Requirements
To run this project, you need the following dependencies:
- Python 3.8+
- TensorFlow 2.4+
- Keras
- OpenCV (`opencv-python`)
- NumPy
- A webcam for real-time detection

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/emotion-detection.git
   cd emotion-detection
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install tensorflow opencv-python numpy
   ```

4. Ensure you have a compatible webcam connected for real-time detection.

## Dataset
The model is trained on a facial expression dataset located at `C:\Users\hemas\Downloads\archive (6)\images\train`. The dataset should be organized in subdirectories, each representing one of the seven emotion classes (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral). You can replace this with any compatible dataset, such as the [FER-2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013).

## Usage
1. **Train the Model**:
   - Run the training script to train the CNN model:
     ```bash
     python train_emotion_model.py
     ```
   - The script loads the dataset, applies data augmentation, trains the model, and saves it as `emotion_detection_model.keras`.

2. **Real-Time Emotion Detection**:
   - Run the real-time detection script:
     ```bash
     python detect_emotion.py
     ```
   - The webcam will open, and the system will detect faces and display the predicted emotion in real-time.
   - Press `q` to exit the webcam feed.

## Model Architecture
The CNN model consists of the following layers:
- 3 Convolutional layers with ReLU activation (32, 64, and 128 filters).
- MaxPooling layers after each convolutional layer.
- Flatten layer to transition to dense layers.
- Dense layer with 128 units and ReLU activation.
- Dropout layer (0.5) to prevent overfitting.
- Output layer with 7 units (softmax) for the seven emotions.

The model is compiled with the Adam optimizer and categorical crossentropy loss.

## Training
- The model is trained for 20 epochs with early stopping (patience=3) to prevent overfitting.
- Data augmentation includes rescaling, rotation, zoom, and horizontal flipping.
- Training and validation data are split (85% training, 15% validation).

## Real-Time Emotion Detection
- Uses OpenCV's Haar Cascade Classifier for face detection.
- Processes webcam frames in grayscale, resizes them to 48x48 pixels, and feeds them to the trained model.
- Displays a bounding box around detected faces with the predicted emotion label.

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.