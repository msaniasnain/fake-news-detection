import pandas as pd

# Create a simple dataset
data = {
    'Temperature': [28, 32, 25, 20, 22, 30, 35, 18, 28, 30],
    'Play': [1, 1, 0, 0, 0, 1, 1, 0, 1, 1]
}

df = pd.DataFrame(data)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Split the data into features (X) and target variable (y)
X = df[['Temperature']]
y = df['Play']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a logistic regression model
lg_model = LogisticRegression()

# Train the model on the training set
lg_model.fit(X_train, y_train)

# Make predictions on the testing set
predictions = lg_model.predict(X_test)

# Evaluate the performance of the model
accuracy = accuracy_score(y_test, predictions)
classification_report_str = classification_report(y_test, predictions)

# # Display the results
# print(f"Accuracy: {accuracy:.2f}")
# print("\nClassification Report:")
# print(classification_report_str)
