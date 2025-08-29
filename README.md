Customer Churn Prediction: An End-to-End Machine Learning Project
🎯 Project Overview
This project focuses on building a machine learning model to predict customer churn for a telecommunications company. By analyzing various customer attributes and behaviors, the goal is to identify customers at risk of leaving the service. The final output is a deployed, interactive web application that provides real-time churn predictions, offering a valuable tool for business decision-making and enhancing customer retention strategies.

📁 Project Structure
data/: Contains the raw dataset (tel_churn.csv).

notebooks/: Jupyter notebooks for data exploration, cleaning, and modeling.

src/: Source code for the Streamlit application (app.py).

README.md: This file.

📋 Key Steps & Methodology
Step 1: Data Preprocessing & Exploratory Data Analysis (EDA)
This initial phase involved a deep dive into the provided dataset. The key tasks included:

Data Cleaning: Handling missing values, converting data types, and correcting inconsistencies.

In-depth EDA: Understanding customer demographics, service usage, and payment behavior through statistical analysis and visualizations.

Churn Analysis: Gaining insights into which factors are most correlated with churn, such as contract type, monthly charges, and total charges.

Feature Engineering: Creating new features or transforming existing ones to improve model performance, such as creating tenure_group from the tenure column.

Step 2: Model Building & Evaluation
In this phase, multiple machine learning algorithms were trained and evaluated to find the best-performing model for the prediction task.

Model Selection: Initial models explored included Decision Tree and Random Forest Classifiers.

Handling Imbalance: The dataset was imbalanced, so techniques like SMOTEENN were used to resample the data and improve the model's ability to predict the minority class (churned customers).

Model Training: The models were trained on the preprocessed data.

Performance Evaluation: The models were evaluated using key metrics like accuracy, precision, recall, and F1-score to ensure robust performance. The Random Forest Classifier with SMOTEENN was selected as the final model due to its high accuracy and balanced F1-score.

Step 3: Deployment via Streamlit
The final and most crucial step was to deploy the trained model into an interactive application.

Interactive UI: A user-friendly interface was built using the Streamlit library.

Real-time Prediction: The app allows users to input customer features and instantly receive a churn prediction.

Accessible Tool: This deployment transforms the static model into a dynamic, accessible tool that can be used by business stakeholders to identify and target at-risk customers.

🛠️ Technologies & Libraries
Python: The core programming language.

Pandas: For data manipulation and analysis.

Scikit-learn: For building and evaluating machine learning models.

Imblearn: For handling imbalanced datasets with SMOTEENN.

Streamlit: For creating and deploying the interactive web application.

🚀 How to Run the Project
Clone the repository:

git clone <repository_link>
Navigate to the project directory:

cd churn-prediction-project
Install the required libraries:

pip install -r requirements.txt
Run the Streamlit application:

streamlit run app.py
The app will open in your browser, and you can begin making predictions!
