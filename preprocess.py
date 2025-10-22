import pandas as pd
import sys
import os

# 1. Load Data
# URL for the Telco Customer Churn dataset
url = 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv'

try:
    df = pd.read_csv(url)
    print("Telco Customer Churn dataset loaded successfully!")
except Exception as e:
    print(f"Error loading data from URL: {e}")
    sys.exit(1)

# 2. Preprocessing Steps
# Convert 'TotalCharges' to numeric and drop NA values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# Convert target column 'Churn' to binary (1 for 'Yes', 0 for 'No')
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Apply One-Hot Encoding to categorical features
categorical_cols = df.select_dtypes(include=['object']).columns.drop('customerID')
df_processed = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Drop the original 'customerID' column
df_processed.drop('customerID', axis=1, inplace=True)

# 3. Save the Processed File
# Ensure the 'data' directory exists
os.makedirs('data', exist_ok=True)
# Save the file
df_processed.to_csv('data/preprocessed_churn.csv', index=False)

print("Preprocessing complete and file saved to data/preprocessed_churn.csv.")