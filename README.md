# DeepLearning-DeiT-Brain-Stroke-Prediction

## Description
This project leverages a state-of-the-art deep learning model using DeiT (Data-Efficient Image Transformers) to predict strokes from CT scans. The primary objective is to enhance early detection and intervention in stroke cases, leading to improved patient outcomes and potentially saving lives.

DeiT is a variant of the Vision Transformer (ViT), which has gained popularity for its effectiveness in image classification tasks. Unlike traditional Convolutional Neural Networks (CNNs), DeiT models excel at capturing global patterns in images with less data requirements, making them particularly suitable for medical imaging applications such as CT scans.

## Dataset
The model is trained on a dataset of CT scans sourced from Kaggle, comprising both positive (stroke) and negative
(no stroke) cases.

- [Kaggle Dataset 1](https://www.kaggle.com/datasets/noshintasnia/brain-stroke-prediction-ct-scan-image-dataset)
- [Kaggle Dataset 2](https://www.kaggle.com/datasets/alymaher/brain-stroke-ct-scan-image)

## Model
The core of this project is the DeiT model, an Efficient Vision Transformer designed for high performance with reduced computational complexity. DeiT leverages attention mechanisms to capture contextual relationships across different parts of the image, which can be particularly beneficial in identifying subtle stroke indicators in medical imaging.

### Benefits of Using DeiT
- **Global Context Awareness**: Unlike CNNs, DeiT processes the entire image to capture global context, which is crucial for accurately detecting strokes.
- **Efficiency**: DeiT achieves comparable accuracy to larger ViT models but with fewer parameters and less computational overhead, making it more feasible for deployment in resource-constrained environments.

## Installation
To set up the project, follow these steps:

1. **Install Dependencies**
   - Ensure you have Python installed (version 3.6 or higher).
   - Install necessary libraries:
     ```bash
     pip install torch torchvision timm seaborn numpy matplotlib scikit-learn 
     ```

2. **Clone the Repository**
   ```bash
   git clone https://github.com/AkramOM606/DeepLearning-DeiT-Brain-Stroke-Prediction.git
   cd DeepLearning-DeiT-Brain-Stroke-Prediction
   ```

3. **Download and Prepare Data**
   - Obtain the Kaggle datasets mentioned above or in the dataset folder of this repo.
   - Preprocess the images according to your model's requirements (resizing, normalization, etc.).

4. **Set Up Environment**
   - Create a virtual environment if desired:
     ```bash
     python -m venv env
     source env/bin/activate  # On Windows use `env\Scripts\activate`
     ```

5. **Run the Model**

## Usage
### Training
1. Prepare your dataset by splitting it into training and validation sets.
2. Run the training script with appropriate parameters.

### Inference
1. Use a trained model to make predictions on new CT scans.
2. The prediction results will indicate whether a stroke is detected, providing a valuable tool for clinicians in diagnosis and treatment planning.

## Contributing
We welcome contributions to enhance this project! Follow these steps to contribute:

1. Fork the repository.
2. Create a new branch for your improvements.
3. Make your changes and commit them.
4. Open a pull request to propose your contributions.

We'll review your pull request and provide feedback promptly.

## License
This project is licensed under the MIT License:
[https://opensource.org/licenses/MIT](https://opensource.org/licenses/MIT) (see LICENSE.md for details).

---

By transitioning from CNNs to DeiT, this project aims to leverage cutting-edge deep learning techniques to improve stroke prediction accuracy and efficiency in medical imaging.
