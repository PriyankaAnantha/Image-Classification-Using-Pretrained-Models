# Image Classification using Pretrained CNN Models
Implemented Transfer Learning with CNNs (Xception, ResNet101V2, InceptionResNetV2) for high-accuracy classification. Applied Batch Normalization and Dropout for enhanced generalization.

## Project Overview
This project focuses on image classification using deep learning techniques. The Car Images Dataset is used to train three different CNN architectures: ResNet101V2, InceptionResNetV2, and DenseNet201. Both Transfer Learning (TL) and Fine-Tuning (FT) are applied to achieve optimal performance.

## Dataset Details
- **Original Dataset**: 4165 images across 7 classes
- **Classes**: Audi, Hyundai Creta, Mahindra Scorpio, Rolls Royce, Swift, Tata Safari, Toyota Innova
- **Training Samples**: 3123
- **Testing Samples**: 1042
- **Subset Dataset**: 1400 images (200 per class), split into 75% training and 25% testing

## Model Architectures
Three pretrained CNN models were used:
1. **ResNet101V2**
2. **InceptionResNetV2**
3. **DenseNet201**

## Training Phases
### 1. Transfer Learning (TL)
- **Modifications applied to classifier layers:**
  - Model-1: Batch Normalization & Dropout (25%) before the output layer
  - Model-2: Batch Normalization & Dropout (35%) before the output layer
  - Model-3: Dropout (15%) before the output layer
- Each model was trained for 10 epochs, using 10% of training data for validation.
- The best performing TL model was saved.

### 2. Fine-Tuning (FT)
- The saved TL models were further fine-tuned using the following strategies:
  - **ResNet101V2**: First 25% of layers frozen, rest trainable
  - **InceptionResNetV2**: First 35% of layers frozen, rest trainable
  - **DenseNet201**: All layers set as trainable
- Each model was trained for another 10 epochs, preserving the best performing FT models.

## Results
For each model, the following were generated:
- **Confusion Matrix**
- **Precision, Recall, and F1-Score**

## File Structure
```
Project_Files/
│── Notebooks_Source/        # Jupyter Notebook Files (.ipynb)
│   │── Model1_TL.ipynb
│   │── Model1_FT.ipynb
│   │── Model2_TL.ipynb
│   │── Model2_FT.ipynb
│   │── Model3_TL.ipynb
│   │── Model3_FT.ipynb
│
│── Notebooks_PDFs/         # PDFs of the notebooks
│   │── Model1_TL.pdf
│   │── Model1_FT.pdf
│   │── Model2_TL.pdf
│   │── Model2_FT.pdf
│   │── Model3_TL.pdf
│   │── Model3_FT.pdf
│
│── README.md               # Project documentation
```

## How to Run
1. **Enable GPU Runtime** in Google Colab before executing the notebooks.
2. Run `ModelX_TL.ipynb` first to train and save the Transfer Learning model.
3. Load the saved TL model and run `ModelX_FT.ipynb` for fine-tuning.
4. Evaluate the final model and generate performance metrics.

## Author
Priyanka A  


