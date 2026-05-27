import pandas as pd

# =========================
# STEP 1: LOAD DATA
# =========================
df = pd.read_csv("Telco-Customer-Churn.csv", encoding='latin1')


# =========================
# STEP 2: CLEAN WHITESPACES IN TEXT
# =========================
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].astype(str).str.strip()


# =========================
# STEP 3: FIX DATA TYPES
# =========================
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())



# =========================
# STEP 4: TARGET VARIABLE
# =========================
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})


# =========================
# STEP 5: CLEAN YES/NO SERVICE COLUMNS
# =========================
service_cols = [
    'Partner','Dependents','PhoneService','PaperlessBilling',
    'MultipleLines','OnlineSecurity','OnlineBackup',
    'DeviceProtection','TechSupport','StreamingTV','StreamingMovies'
]

for col in service_cols:
    df[col] = df[col].replace({
        'Yes': 1,
        'No': 0,
        'No internet service': 0,
        'No phone service': 0
    })

df[service_cols] = df[service_cols].astype(int)


# =========================
# STEP 6: FEATURE ENGINEERING
# =========================
df['AvgCharges'] = df['TotalCharges'] / (df['tenure'] + 1)


# =========================
# STEP 7: ONE-HOT ENCODING
# =========================
df = pd.get_dummies(df, columns=[
    'gender',
    'InternetService',
    'Contract',
    'PaymentMethod'
], drop_first=True)


# =========================
# STEP 8: REMOVE CUSTOMER ID
# =========================
df.drop(columns=['customerID'], inplace=True)


# =========================
# STEP 9: CLEAN COLUMN NAMES (VERY IMPORTANT FOR SNOWFLAKE)
# =========================
df.columns = (
    df.columns
    .str.strip()
    .str.replace(' ', '_')
    .str.replace('-', '_')
    .str.replace('(', '')
    .str.replace(')', '')
)


# =========================
# STEP 10: ENSURE NO BOOLEAN ISSUES
# =========================
df = df.replace({True: 1, False: 0})


# =========================
# STEP 11: SAVE FINAL FILE
# =========================
df.to_csv("final_churn.csv", index=False)


# =========================
# STEP 12: FINAL CHECK
# =========================
print(df.info())
print(df.head())
print(df.isnull().sum())

df = df.replace({
    True: 1,
    False: 0,
    'TRUE': 1,
    'FALSE': 0,
    'Yes': 1,
    'No': 0,
    'No phone service': 0,
    'No internet service': 0
})

df.to_csv("final_clean.csv", index=False)