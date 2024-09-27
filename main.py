import pandas as pd
from preprocessing.clean_data import preprocess_text
from utils.train_test_split import split_data, vectorize_text
from models.decision_tree import train_decision_tree
from models.random_forest import train_random_forest
from models.XGBoost import train_xgboost

# Load dataset
df = pd.read_csv('./data/IMDB-Dataset.csv')

# Clean dataset
df['review'] = df['review'].apply(preprocess_text)

# Split dataset
x_train, x_test, y_train, y_test = split_data(df)

# Vectorize text
x_train_encoded, x_test_encoded = vectorize_text(x_train, x_test)

# Train models
# dt_accuracy = train_decision_tree(
#     x_train_encoded, y_train, x_test_encoded, y_test)
# rf_accuracy = train_random_forest(
#     x_train_encoded, y_train, x_test_encoded, y_test)
xgboost_accuracy = train_xgboost(
    x_train_encoded, y_train, x_test_encoded, y_test)

# print(f"Decision Tree Accuracy: {dt_accuracy}") # -> 0.7259
# print(f"Random Forest Accuracy: {rf_accuracy}") # -> 0.8259

print(f"XGBoost Accuracy: {xgboost_accuracy}")
