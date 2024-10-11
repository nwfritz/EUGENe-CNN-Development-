# Predicting Sequencing Data with EUGENe and a Custom CNN Model

This notebook demonstrates the use of the EUGENe framework to develop and train a custom Convolutional Neural Network (CNN) model for predicting sequencing data. 

## Notebook Overview

1. **Introduction:** Introduces the goal of the notebook and the EUGENe framework.
2. **Data Loading and Preprocessing:** Loads the 'leaf' dataset from Jores21 using `seqdatasets` and performs one-hot encoding and ID generation using `eugene.preprocess`.
3. **Data Splitting:** Splits the data into training, validation, and test sets.
4. **Model Definition:** Defines a custom CNN model named `Fritz_CNN` with five convolutional layers, batch normalization, max pooling, a residual connection, and a fully connected layer.
5. **Model Initialization and Summary:** Initializes the weights of the custom CNN model and the EUGENe-provided CNN model using `models.init_weights` and prints model summaries using `model.summary()`.
6. **Model Training:** Trains both models using `train.fit_sequence_module` with specified parameters like epochs, batch size, and loss function.
7. **Performance Evaluation:** Plots the loss and R-squared curves for both models using `eugene.plot.training_summary` to compare their performance.
8. **Conclusion:** Summarizes the findings based on the training results and discusses potential improvements.

## Dependencies

- EUGENe tools: `pip install 'eugene-tools'`
- torchmetrics: `pip install torchmetrics==0.10.1`
- seqdatasets
- eugene
- torch
- torch.nn

## Usage

1. Mount your Google Drive to access the dataset and save the model.
2. Update the `os.chdir()` command to your project directory.
3. Run the notebook cells sequentially to load data, define, train, and evaluate the models.

## Results

The notebook provides a comparison of the performance of the custom `Fritz_CNN` model and the EUGENe-provided CNN model. The conclusions drawn from the training results are discussed in the "Conclusion" section of the notebook.

## Further Exploration

- Experiment with different model architectures, hyperparameters, and training strategies to further improve performance.
- Explore additional features and functionalities provided by the EUGENe framework.
