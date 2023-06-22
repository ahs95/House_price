# House Price Prediction using Linear Regression
This is a machine learning project for house price prediction using linear regression. The goal is to develop a model that can predict the price of a house based on various features. The project is implemented in Python programming language using the scikit-learn library.
## Dataset
The dataset used for training and evaluation is collected from [Kaggle](https://www.kaggle.com/datasets/ijajdatanerd/property-listing-data-in-bangladesh). It is a CSV file containing information about property listings in Bangladesh.
## Approach
1. Data Preprocessing: The dataset is preprocessed to handle missing values, remove irrelevant columns, and transform features as needed.
2. Feature Engineering: The "beds" and "bath" columns are combined by multiplying them to create a new feature.
3. Feature Scaling: The "area_sqft" column is scaled using Z-score normalization to bring it to a common scale.
4. Train-Validation-Test Split: The dataset is split into the train, validation, and test sets with a ratio of 60:20:20.
5. Model Training: Linear regression model is trained on the training set.
6. Model Evaluation: The model's performance is evaluated using mean squared error (MSE).
## Requirements
+ Python 3.x
+ pandas
+ scikit-learn
+ matplotlib (for visualizing purposes)
