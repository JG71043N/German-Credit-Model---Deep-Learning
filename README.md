# German-Credit-Model---Deep-Learning

Project Report: Credit Risk Classification

Course: Introduction to Deep Learning

Date: March 10, 2026

Dataset: German Credit Data
___________________________________________________________________________________________________________________

1. ***Introduction & Problem Statement***
   
    The objective of this project is to develop a supervised machine learning model to assess credit risk. Using the German Credit Dataset, the goal is to classify loan applicants as either a "non-risk" or "risk” based on historical data. This classification is vital for financial institutions to automate loan approval processes, minimize the risk of default, and optimize lending strategies within a banking institution.

    A key focus of this study is the comparative analysis of two different modeling approaches: Linear Regression (adapted for classification) and Random Forest. While Linear Regression provides a baseline for understanding the direct relationships between borrower features and risk, the Random Forest model is employed to capture non-linear patterns and complex interactions within the data. By comparing these two models, I aim to determine which approach offers the highest predictive accuracy and the most reliable risk assessment for financial decision-making.

4. ***Data Preprocessing and Feature Engineering***


    To prepare the dataset for the machine learning models, several preprocessing steps were performed:

    •	Data Cleaning: Checked for missing values and ensured all data types were correct for analysis. The only columns with missing data were checking account and saving accounts which where non-numerical. Saving accounts had 183 missing values while Checking account had 394. Simply deleting the rows where there are missing values would result in a large portion of the data missing. Due to this I filled the missing values of the checking and saving accounts to 'none' since the values of the data were nominal to begin with (none < little < moderate < rich < quite rich)

    Data Deletion: Id was deleted to reduce noise -> it provided no predictive power.

    •	Encoding: Categorical variables such as "Checking Account," “Savings Account," and "Sex" were transformed into numerical values using Binary Map Encoding. On the contrary, the variables "Purpose" and "Housing" were one-hot encoded.

    •	Feature Scaling: Since features like "Credit Amount", "Duration", and "Age" have different scales, I applied StandardScaler to ensure that features with larger magnitudes did not unfairly dominate the model in training
    
    •	Feature Selection: The features used for the target variable (Loan_Status) included Duration, Credit Amount, Saving Accounts, Age, Job, Purpose, and Housing.


5. ***Methodology and Model Implementation***

    The dataset was split into a training set (70%) and a testing set (30%) using a random_state of 42 to ensure reproducibility. Two distinct classifiers were implemented to compare performance:

        A.	Logistic Regression: This was used as the baseline model to establish a linear relationship between the borrower’s features and the probability of default.

        B. Random Forest: This model was selected to capture more complex, non-linear patterns within the applicant data that a simple logistic regression might miss.

    ***Baseline for Both Models:***
   
    Important to note that during the development a 80/20 split was attempted but ultimately a 70/30 split was chosen due to its accuracy 

    Initially, including Age made the High-Risk class too small for the model to learn effectively. The first rule classifier was as follows: Age<30 AND Loan Duration>24 AND Credit Amount >$4000 . By focusing on Loan Duration (>24) and Credit Amount (>$4000), By doing so I was able to capture 15.1% of the dataset as High Risk, allowing the models to achieve a more reliable predictive weight.

    Additionally, I initially achieved a high accuracy of 98% to 99% on the first models. However, when I questioned these results, I realized there was a huge data leak. The model was essentially cheating to get that high accuracy. Even though I excluded the target from the predictors, including variables like credit amount and loan duration significantly increased the accuracy because the model was just plugging in the calculation for the risk status. By fixing this, the model performed slightly lower, but it is much more accurate in a real-world banking sense for detecting high and low risk loaners.

   **Taking note the Loan Status was not included in the original data therefore I made a Target for the predictions**
   
    Hence why for the predictors(X) Credit amount , Duration and Loan Status were opted out because it would cheat / Data leak with those attributes

7. ***Performance Evaluation***
   
The models were evaluated using Accuracy, a Confusion Matrix, and a Classification Report.

    4.1 Logistic Regression Results
    •	Accuracy:   84.67%
    •	Precision (Bad Risk): 0.33
    •	Recall (Bad Risk): 0.02
    •	Confusion Matrix Observations: The model correctly identified 253 good applicants but struggled with 44 false negatives.
    
        4.2 Random Forest Results
        •	Accuracy: 82.00%
        •	Precision (Bad Risk): 0.34
        •	Recall (Bad Risk): 0.22
        •	Confusion Matrix Observations: The model correctly identified 236 good applicants but struggled with 34 false negatives.


9. ***Model Comparison and Interpretation***
    Comparing the two models, the random forest model performed better overall. While Logistic Regression provided a solid baseline and even had a higher Accuracy percentage that is only one part of the whole picture. The Random Forest showed a higher ability to identify high-risk borrowers. 
    Looking at the Regression model it is clear that the model learned to predict low-risk for virtually all the tests. In doing so it only correctly predicted 1 instance as high risk while failing to truly predict risky 

    In a financial context, Recall for "Bad" credit applicants is the most critical metric. Missing a "Bad" risk (a False Negative) is significantly more expensive for a bank than wrongly rejecting a "Good" applicant. My results show that random forest is better in its lending approach.

10. ***Conclusion***
    This project highlights a critical lesson in machine learning: Accuracy is not a only measure for success, especially in imbalanced financial datasets. While the Logistic Regression model achieved a higher overall accuracy of 84.67%, it proved to be practically ineffective for risk management, as it successfully identified only 1 out of 45 high-risk applicants. This "lazy" model defaulted to the majority class, creating a dangerous blind spot for a financial institution.

In contrast, the Random Forest model—despite its lower accuracy of 82.00%—demonstrated far superior utility. By capturing non-linear interactions between features like age and savings, it achieved a significantly higher Recall (0.22 vs. 0.02). In a banking context, the cost of a False Negative (lending to a high-risk borrower who defaults) far outweighs the cost of a False Positive (denying a safe borrower). This project has shown me the importance of not solely relying on headline metrics. It is importance to even question Why or How the model achieved such a high accuracy.
    

    Extras Done on This Project
        •	Addressing class imbalance (as there are more "Good" than "Bad" cases in this data):
        •	SMOTE done on the random forest model (as an extra) comparison was done on the non-SMOTE models (Model Logistic Regression.py & Model Random Forest.py)

            SMOTE REFLECTION/EVALUATION: 
            Accuracy: 79.67%
        •	Precision (Bad Risk): 0.34
        •	Recall (Bad Risk): 0.38
        •	Confusion Matrix Observations: The model correctly identified 222 good applicants but struggled with 28 false negatives.
