"""
Name: Jalyin Gonzalez 
Date: 3/2/26
Project 1: Loan Classifier High/Low Risk
Model: Random Forest Model 
"""
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler #scaling
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, RocCurveDisplay

df = pd.read_csv('german_credit_data.csv')

#Step 1 -- Data Exploration/Cleaning/Encoding -- 
#1A. Missing NaN values 

'''Analysis: Saving accounts     183
Checking account    394
Would lose too much data if we simply deleted the missing valued rows
''' 
df['Saving accounts'] = df['Saving accounts'].fillna('none')
df['Checking account'] = df['Checking account'].fillna('none')

#Noise (overfitting) COlumn deletion
df = df.drop(columns=['Id'])

#1B Encoding 
# Binary map for Sex
df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
#Binary Map for saving/checking account
account_map = {'none': 0, 'little': 1, 'moderate': 2, 'quite rich': 3, 'rich': 4}

# Apply it to both columns / Noting I did not want to change any of the data names hense why it is savings account w/ an s
df['Saving accounts'] = df['Saving accounts'].map(account_map)
df['Checking account'] = df['Checking account'].map(account_map)

# One-hot encode the rest (Housing and Purpose)
df = pd.get_dummies(df, columns=['Housing' ,'Purpose'])

#Strict pred column
df['Loan_status'] = np.where(
    (df['Duration'] > 24) & (df['Credit amount'] > 4000), 
    1,  # High Risk
    0   # Low Risk
)

print("Target Variable Counts:")
print(df['Loan_status'].value_counts())

#only scale the continuous numbers
num_cols = ['Age', 'Credit amount', 'Duration', "Saving accounts", "Checking account"]
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

#Step 2: --- Feature engineering and Target ---
X = df.drop(columns=['Loan_status', 'Duration', 'Credit amount'])
y = df['Loan_status']

#Step 3: --- Model & Training ---
#handling potential class imbalance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# a. Initialize the model
log_model = LogisticRegression()

log_model.fit(X_train, y_train)

# b. Predictions
y_pred = log_model.predict(X_test)

# Step 4: Evaluation 
accuracy = accuracy_score(y_test, y_pred)

print("\n------ LOGISTIC REGRESSION RESULTS ------")
print(f"Accuracy Score: {accuracy * 100:.2f}%")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

X_test_with_results = X_test.copy()
X_test_with_results['Actual_Risk'] = y_test
X_test_with_results['Predicted_Risk'] = y_pred
print("\nFirst 5 predictions:")
print(X_test_with_results[['Actual_Risk', 'Predicted_Risk']].head())

# Step 5: Visualization : graphs/plots
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Confusion Matrix Visual
ConfusionMatrixDisplay.from_estimator(log_model, X_test, y_test, display_labels=['Low Risk', 'High Risk'], cmap='Blues', ax=ax[0])
ax[0].set_title('Confusion Matrix')

# Plot 2: ROC Curve
RocCurveDisplay.from_estimator(log_model, X_test, y_test, color='black', ax=ax[1])
ax[1].plot([0, 1], [0, 1], color='navy', linestyle='--')
ax[1].set_title('ROC Curve (Model Performance)')

plt.tight_layout()
plt.show()