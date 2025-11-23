# Waste Classification using CNN (TensorFlow/Keras)

A deep learning-based image classification model that identifies Organic and Recyclable waste categories using a Convolutional Neural Network (CNN).
This project demonstrates end-to-end development of an image classifier including data loading, preprocessing, model training, evaluation, and prediction.


Objective

The primary objective is to develop a machine learning model that can accurately classify waste images into predefined categories using the Waste Classification Dataset from Kaggle.

Dataset Description

Dataset is divided into train data (85%) and test data (15%)  

Training data - 22564 images  Test data - 2513 images

Methodology

Data Preprocessing

Resizing images to a uniform dimension

Normalizing pixel values

Data augmentation to reduce overfitting

Model Development

Built a Convolutional Neural Network (CNN) using TensorFlow/Keras

Included convolution, max-pooling, dense layers

Training & Evaluation

Trained for multiple epochs

Evaluated using accuracy, loss curves, and confusion matrix

Model Saving

The trained model is exported as waste_classifier_model.h5

Results

The model demonstrates strong performance with high accuracy and consistent validation results, indicating good generalization across waste categories.

Conclusion

This project successfully showcases the use of deep learning in waste classification and highlights its potential for integration in automated recycling systems.

