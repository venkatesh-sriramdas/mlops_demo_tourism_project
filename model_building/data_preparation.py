from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from huggingface_hub import HfApi
import pandas as pd
import numpy as np
import joblib
import os

# Download dataset from HuggingFace
print("📥 Downloading dataset from HuggingFace...")
file_path = hf_hub_download(
    repo_id="svenkateshdotnet/tourism_project",
    filename="tourism.csv",
    repo_type="dataset",
    token=os.getenv("HF_TOKEN")
)

# Load the dataset
df = pd.read_csv(file_path)
print(f"✅ Dataset loaded: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Data Cleaning
print("\n🧹 Cleaning data...")
# Drop unnamed index column if it exists
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# Drop CustomerID as it's not needed for modeling
if 'CustomerID' in df.columns:
    df = df.drop(columns=['CustomerID'])

# Handle missing values
print(f"Missing values before:\n{df.isnull().sum()}")

# Fill missing numerical values with median
numerical_cols = df.select_dtypes(include=[np.number]).columns
for col in numerical_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

# Fill missing categorical values with mode
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

print(f"Missing values after:\n{df.isnull().sum().sum()}")

# Feature Engineering
print("\n🔧 Creating engineered features...")
df['Income_per_person'] = df['MonthlyIncome'] / (df['NumberOfPersonVisiting'] + 1)
df['Trips_per_year_ratio'] = df['NumberOfTrips'] / (df['Age'] + 1)
df['Children_ratio'] = df['NumberOfChildrenVisiting'] / (df['NumberOfPersonVisiting'] + 1)
df['Followup_per_pitch'] = df['NumberOfFollowups'] / (df['DurationOfPitch'] + 1)

# Separate features and target
X = df.drop(columns=['ProdTaken'])
y = df['ProdTaken']

# Encode categorical variables
print("\n🔤 Encoding categorical variables...")
label_encoders = {}
for col in categorical_cols:
    if col in X.columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

# Split the data
print("\n✂️ Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

# Scale numerical features
print("\n📊 Scaling numerical features...")
scaler = StandardScaler()
numerical_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

# Save processed data locally
print("\n💾 Saving processed data...")
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False, header=True)
y_test.to_csv("y_test.csv", index=False, header=True)

# Save scaler and encoders
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

# Upload processed files to HuggingFace
print("\n📤 Uploading processed data to HuggingFace...")
api = HfApi(token=os.getenv("HF_TOKEN"))

files = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv", "scaler.pkl", "label_encoders.pkl"]
for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=f"processed_data/{file_path}",
        repo_id="svenkateshdotnet/tourism_project",
        repo_type="dataset"
    )
    print(f"✅ Uploaded {file_path}")

print("✅ Data preparation completed successfully!")
