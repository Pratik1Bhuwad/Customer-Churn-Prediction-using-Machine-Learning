import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.combine import SMOTEENN
from sklearn.preprocessing import LabelEncoder
import io
import pickle

# --- DATA PREPARATION AND MODEL TRAINING ---

@st.cache_data
def load_data():
    """Loads and preprocesses the tel_churn.csv data."""
    # To make this app self-contained, we'll create a dummy dataframe if the file isn't available.
    try:
        df = pd.read_csv("tel_churn.csv")
    except FileNotFoundError:
        st.error("tel_churn.csv not found. Using a dummy dataset for demonstration.")
        data = {
            'SeniorCitizen': [0, 1, 0, 1, 0],
            'MonthlyCharges': [29.85, 56.95, 70.70, 99.00, 45.00],
            'TotalCharges': ['29.85', '1889.5', '151.65', '450.0', '120.0'],
            'gender': ['Female', 'Male', 'Female', 'Male', 'Female'],
            'Partner': ['Yes', 'No', 'No', 'Yes', 'No'],
            'Dependents': ['No', 'No', 'No', 'No', 'Yes'],
            'PhoneService': ['No', 'Yes', 'Yes', 'Yes', 'No'],
            'MultipleLines': ['No phone service', 'No', 'Yes', 'Yes', 'No phone service'],
            'InternetService': ['DSL', 'DSL', 'Fiber optic', 'Fiber optic', 'DSL'],
            'OnlineSecurity': ['No', 'Yes', 'No', 'No', 'No'],
            'OnlineBackup': ['Yes', 'No', 'No', 'No', 'Yes'],
            'DeviceProtection': ['No', 'Yes', 'No', 'No', 'Yes'],
            'TechSupport': ['No', 'No', 'No', 'Yes', 'No'],
            'StreamingTV': ['No', 'No', 'Yes', 'Yes', 'Yes'],
            'StreamingMovies': ['No', 'No', 'Yes', 'No', 'Yes'],
            'Contract': ['Month-to-month', 'One year', 'Month-to-month', 'One year', 'Month-to-month'],
            'PaperlessBilling': ['Yes', 'No', 'Yes', 'No', 'Yes'],
            'PaymentMethod': ['Electronic check', 'Mailed check', 'Electronic check', 'Credit card (automatic)', 'Mailed check'],
            'tenure': [1, 34, 2, 45, 1],
            'Churn': [0, 0, 1, 0, 1]
        }
        df = pd.DataFrame(data)

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)

    # Encode categorical features
    for column in df.columns:
        if df[column].dtype == 'object':
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])

    # Convert binary columns to boolean/integer
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'gender_Male']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].astype(bool).astype(int)

    # Convert all columns to numeric, coercing errors to NaN
    df = df.apply(pd.to_numeric, errors='coerce')

    # Re-fill NaNs after conversion (for any non-convertible values)
    df.fillna(df.mean(), inplace=True)

    # Separate features (x) and target (y)
    x = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Resample the data using SMOTEENN
    sm = SMOTEENN(random_state=100)
    x_resampled, y_resampled = sm.fit_resample(x, y)
    
    return x_resampled, y_resampled

@st.cache_resource
def train_model(x_train, y_train):
    """Trains and returns the RandomForestClassifier model."""
    model = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=100, max_depth=6, min_samples_leaf=8)
    model.fit(x_train, y_train)
    return model

# Load data and train the model
x_resampled, y_resampled = load_data()
x_train, x_test, y_train, y_test = train_test_split(x_resampled, y_resampled, test_size=0.2, random_state=100)
model = train_model(x_train, y_train)

# --- STREAMLIT UI ---

st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Customer Churn Prediction App")
st.markdown("This app predicts whether a customer will churn based on various features.")

# Sidebar for user input
st.sidebar.header("User Input Features")

# Helper function to get user input for different features
def get_user_input():
    # Columns from the original data that we need inputs for.
    # We must ensure the input features match the model's training features.
    
    
    # ['SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender_Female', 'gender_Male', 'Partner_No', 'Partner_Yes', 'Dependents_No', 'Dependents_Yes', 'PhoneService_No', 'PhoneService_Yes', 'MultipleLines_No', 'MultipleLines_No phone service', 'MultipleLines_Yes', 'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No', 'OnlineSecurity_No', 'OnlineSecurity_No internet service', 'OnlineSecurity_Yes', 'OnlineBackup_No', 'OnlineBackup_No internet service', 'OnlineBackup_Yes', 'DeviceProtection_No', 'DeviceProtection_No internet service', 'DeviceProtection_Yes', 'TechSupport_No', 'TechSupport_No internet service', 'TechSupport_Yes', 'StreamingTV_No', 'StreamingTV_No internet service', 'StreamingTV_Yes', 'StreamingMovies_No', 'StreamingMovies_No internet service', 'StreamingMovies_Yes', 'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year', 'PaperlessBilling_No', 'PaperlessBilling_Yes', 'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check', 'tenure_group_1 - 12', 'tenure_group_13 - 24', 'tenure_group_25 - 36', 'tenure_group_37 - 48', 'tenure_group_49 - 60', 'tenure_group_61 - 72']
    # We will create a user-friendly UI to generate these inputs.
    
    gender = st.sidebar.radio("Gender", ('Male', 'Female'))
    senior_citizen = st.sidebar.radio("Senior Citizen", ('Yes', 'No'))
    partner = st.sidebar.radio("Partner", ('Yes', 'No'))
    dependents = st.sidebar.radio("Dependents", ('Yes', 'No'))
    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    phone_service = st.sidebar.radio("Phone Service", ('Yes', 'No'))
    multiple_lines = st.sidebar.radio("Multiple Lines", ('Yes', 'No', 'No phone service'))
    internet_service = st.sidebar.radio("Internet Service", ('DSL', 'Fiber optic', 'No'))
    online_security = st.sidebar.radio("Online Security", ('Yes', 'No', 'No internet service'))
    online_backup = st.sidebar.radio("Online Backup", ('Yes', 'No', 'No internet service'))
    device_protection = st.sidebar.radio("Device Protection", ('Yes', 'No', 'No internet service'))
    tech_support = st.sidebar.radio("Tech Support", ('Yes', 'No', 'No internet service'))
    streaming_tv = st.sidebar.radio("Streaming TV", ('Yes', 'No', 'No internet service'))
    streaming_movies = st.sidebar.radio("Streaming Movies", ('Yes', 'No', 'No internet service'))
    contract = st.sidebar.radio("Contract", ('Month-to-month', 'One year', 'Two year'))
    paperless_billing = st.sidebar.radio("Paperless Billing", ('Yes', 'No'))
    payment_method = st.sidebar.radio("Payment Method", ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'))
    monthly_charges = st.sidebar.number_input("Monthly Charges", value=50.0)
    total_charges = st.sidebar.number_input("Total Charges", value=500.0)

    # Create a dictionary to hold the raw user input
    user_data = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    
    # One-hot encode the user input to match the training data
    features = pd.DataFrame(user_data, index=[0])
    
    # Create the full dataframe with all columns from training data
    # and set them to 0
    df_encoded = pd.DataFrame(0, index=[0], columns=x_resampled.columns)
    
    # Map raw input to the one-hot encoded columns
    df_encoded['SeniorCitizen'] = 1 if user_data['SeniorCitizen'] == 'Yes' else 0
    df_encoded['MonthlyCharges'] = user_data['MonthlyCharges']
    df_encoded['TotalCharges'] = user_data['TotalCharges']
    
    # Handle gender
    if user_data['gender'] == 'Female':
        if 'gender_Female' in df_encoded.columns:
            df_encoded['gender_Female'] = 1
        else:
            # Fallback for models without gender_Female column
            pass
    
    # Handle Partner and Dependents (binary)
    if 'Partner_Yes' in df_encoded.columns:
        df_encoded['Partner_Yes'] = 1 if user_data['Partner'] == 'Yes' else 0
        df_encoded['Partner_No'] = 1 if user_data['Partner'] == 'No' else 0
    
    if 'Dependents_Yes' in df_encoded.columns:
        df_encoded['Dependents_Yes'] = 1 if user_data['Dependents'] == 'Yes' else 0
        df_encoded['Dependents_No'] = 1 if user_data['Dependents'] == 'No' else 0
    
    # Handle other categorical features
    for col in ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']:
        
        # Check if the column exists in the training data
        prefix = f'{col}_'
        
        # Get the value from user input and create the correct column name
        val = user_data[col].replace(' ', '_')
        encoded_col_name = f'{prefix}{val}'
        
        if encoded_col_name in df_encoded.columns:
            df_encoded[encoded_col_name] = 1

    # Handle tenure groups
    tenure_val = user_data['tenure']
    if tenure_val <= 12:
        if 'tenure_group_1 - 12' in df_encoded.columns:
            df_encoded['tenure_group_1 - 12'] = 1
    elif 13 <= tenure_val <= 24:
        if 'tenure_group_13 - 24' in df_encoded.columns:
            df_encoded['tenure_group_13 - 24'] = 1
    elif 25 <= tenure_val <= 36:
        if 'tenure_group_25 - 36' in df_encoded.columns:
            df_encoded['tenure_group_25 - 36'] = 1
    elif 37 <= tenure_val <= 48:
        if 'tenure_group_37 - 48' in df_encoded.columns:
            df_encoded['tenure_group_37 - 48'] = 1
    elif 49 <= tenure_val <= 60:
        if 'tenure_group_49 - 60' in df_encoded.columns:
            df_encoded['tenure_group_49 - 60'] = 1
    else:
        if 'tenure_group_61 - 72' in df_encoded.columns:
            df_encoded['tenure_group_61 - 72'] = 1
    
    return df_encoded

# Display the user input
st.subheader("Your Input Features")
input_df = get_user_input()
st.write(input_df)

# Prediction button
if st.button("Predict Churn"):
    with st.spinner("Predicting..."):
        # Make the prediction
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        st.subheader("Prediction Result")
        churn_status = "will churn" if prediction[0] == 1 else "will not churn"
        st.info(f"The customer **{churn_status}**.")
        
        st.subheader("Prediction Probability")
        
        # Display the probability using a bar chart
        proba_df = pd.DataFrame(
            {'Churn': ['Will Not Churn', 'Will Churn'],
             'Probability': [prediction_proba[0][0], prediction_proba[0][1]]}
        )
        
        st.bar_chart(proba_df, x='Churn', y='Probability')
        st.markdown(f"**Probability of not churning:** {prediction_proba[0][0]:.2f}")
        st.markdown(f"**Probability of churning:** {prediction_proba[0][1]:.2f}")

