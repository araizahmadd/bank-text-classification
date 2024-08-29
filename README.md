# bank-text-classification
Welcome to the Bank Text Classification Project! This project is one of my earliest endeavors in the field of Machine Learning and Natural Language Processing (NLP). It focuses on classifying bank transaction descriptions into predefined categories, such as 'Shopping', 'Food', 'Utilities', 'Transport', 'Health', 'Entertainment', 'Cash/Transfers', and 'Income'.

## Project Overview
The main goal of this project was to develop a model that accurately classifies text data into these specific categories. To achieve this, I explored several machine learning algorithms, including Naive Bayes, SGDClassifier, and Logistic Regression. After careful experimentation, Logistic Regression was selected as the final model due to its superior performance.

## Data Preprocessing
Given the nature of text data, preprocessing was a crucial step in this project. Below are the key preprocessing techniques applied:

Label Mapping: Transaction descriptions were mapped to their corresponding categories using a label mapping technique.

Handling Missing Data: Rows with missing values were identified and dropped to ensure data integrity.

Text Cleaning: Text data was cleaned to remove noise, such as special characters, numbers, and extra spaces. This step helped in normalizing the data for better model performance.

## Feature Extraction
The textual data was transformed into numerical representations using the following methods:

CountVectorizer: This step involved converting the cleaned text data into a matrix of token counts, which is a representation of the frequency of each word in the corpus.

TF-IDF Transformation: To account for the importance of words in the context of the entire dataset, the CountVectorizer output was transformed into TF-IDF (Term Frequency-Inverse Document Frequency) scores. This helps to highlight important words while down-weighting common words.

## Model Selection
After preprocessing and feature extraction, three models were tested:

Naive Bayes
SGDClassifier
Logistic Regression
Logistic Regression was chosen as the final model due to its high accuracy and robustness in this specific text classification task.

## Project Structure
The project consists of the following key files:

train_bank.ipynb: This notebook is dedicated to training the model. It walks through the entire process, from data preprocessing to model training.

Outputs: The trained models and transformers are saved as .pkl files.
countvectorizer.pkl
tfidf_transformer.pkl
logreg_model.pkl
main-bank.ipynb: This notebook is used for deploying the model. It allows you to load the trained model and transformers and use them to classify new text data. Simply input the transaction descriptions, and the model will output the predicted category.

Model Deployment
In the main-bank.ipynb notebook, the trained Logistic Regression model can be utilized to classify new bank transaction data. The steps include:

Loading the Model: Load the pre-trained CountVectorizer, TF-IDF Transformer, and Logistic Regression model from the saved .pkl files.

Preprocessing the Input Data: Clean and preprocess the new input text data using the same techniques applied during training.

Vectorization and Transformation: Convert the text into numerical form using the loaded CountVectorizer and TF-IDF Transformer.

Prediction: Use the Logistic Regression model to predict the category of each transaction.

Confidentiality
Please note that the original dataset used for training is unavailable due to confidentiality constraints. However, the project framework and code provide a solid foundation for applying similar techniques to other datasets.
