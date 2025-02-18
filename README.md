# Celsius to Fahrenheit Machine Learning Model

## Overview
This repository contains a simple machine learning model built using TensorFlow to predict Fahrenheit temperatures given Celsius inputs. The model is trained using a small dataset and demonstrates the basics of training a neural network for a straightforward regression task.

## Files Included
- `CelsiusToFahrenheit.ipynb` - Jupyter Notebook with the complete code to train and test the model.
- `Celsius_to_Fahrenheit.csv` - Dataset containing Celsius-to-Fahrenheit conversion values used for training.

## Requirements
To run this project, you need:
- Python 3.x
- TensorFlow
- Pandas
- NumPy
- Matplotlib
- Seaborn

## Model Details
- The model is a simple feedforward neural network with one dense layer.
- It uses **Mean Squared Error (MSE)** as the loss function.
- The optimizer used is **Adam** with a learning rate of 1.
- It is trained for **500 epochs** to minimize error.

## Results
- The model learns to approximate the formula:
  
  ```math
  Fahrenheit = (Celsius × 9/5) + 32
  ```
- After training, the model can predict Fahrenheit values with minimal error.
---
Happy coding! 🚀

