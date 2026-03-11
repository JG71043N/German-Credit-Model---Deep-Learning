"""
Name: Jalyin Gonzalez 
Date: 3/2/26
Project 1: Loan Classifier High/Low Risk
Model: Random Forest Model 
"""
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Changed to Random Forest
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, RocCurveDisplay
from imblearn.over_sampling import SMOTE # type: ignore
 

# Step 1 Data Preperation and Cleaning
df = pd.read_csv('german_credit_data.csv')

# Handle Missing Values
df['Saving accounts'] = df['Saving accounts'].fillna('none')
df['Checking account'] = df['Checking account'].fillna('none')
df = df.drop(columns=['Id'])

# Encoding (both Map and Hot-encoding)
df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
account_map = {'none': 0, 'little': 1, 'moderate': 2, 'quite rich': 3, 'rich': 4}
df['Saving accounts'] = df['Saving accounts'].map(account_map)
df['Checking account'] = df['Checking account'].map(account_map)
df = pd.get_dummies(df, columns=['Housing' ,'Purpose'])

# Same Rule to keep the tests the same for comparison between models (loan Status == Target)
df['Loan_status'] = np.where(
    (df['Duration'] > 24) & (df['Credit amount'] > 4000), 
    1, 0
)



# Scaling
num_cols = ['Age', 'Credit amount', 'Duration']
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Step 2: Features and Target
X = df.drop(columns=['Loan_status', 'Duration', 'Credit amount'])
y = df['Loan_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# --- RANDOM FOREST MODEL ---
# n_estimators=100 means we are using 100 individual decision trees

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_resampled, y_train_resampled)

# Predictions
y_pred = rf_model.predict(X_test)

#Step 3: Evaluation
print("\n--- RANDOM FOREST RESULTS ---")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 4: Feature Importance 
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
importances.nlargest(5).plot(kind='barh', color='teal')
plt.title('Top 5 Factors Driving Loan Risk')
plt.show()

# Step 5: Visualizing Performance
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Confusion Matrix
ConfusionMatrixDisplay.from_estimator(rf_model, X_test, y_test, cmap='Greens', ax=ax[0])
ax[0].set_title('RF Confusion Matrix')

# ROC Curve
RocCurveDisplay.from_estimator(rf_model, X_test, y_test, color='darkgreen', ax=ax[1])
ax[1].plot([0, 1], [0, 1], linestyle='--', color='gray')
ax[1].set_title('RF ROC Curve')

plt.tight_layout()
plt.show()