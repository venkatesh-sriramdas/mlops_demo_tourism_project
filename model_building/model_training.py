from huggingface_hub import hf_hub_download, HfApi, create_repo
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
import os

print("=" * 80)
print("🚀 TOURISM PROJECT - MODEL TRAINING WITH MLFLOW")
print("=" * 80)

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism_package_prediction")

print("\n📥 Downloading processed data from HuggingFace...")
# Download processed data
X_train = pd.read_csv(hf_hub_download(
    repo_id="svenkateshdotnet/tourism_project",
    filename="processed_data/X_train.csv",
    repo_type="dataset",
    token=os.getenv("HF_TOKEN")
))

X_test = pd.read_csv(hf_hub_download(
    repo_id="svenkateshdotnet/tourism_project",
    filename="processed_data/X_test.csv",
    repo_type="dataset",
    token=os.getenv("HF_TOKEN")
))

y_train = pd.read_csv(hf_hub_download(
    repo_id="svenkateshdotnet/tourism_project",
    filename="processed_data/y_train.csv",
    repo_type="dataset",
    token=os.getenv("HF_TOKEN")
))['ProdTaken']

y_test = pd.read_csv(hf_hub_download(
    repo_id="svenkateshdotnet/tourism_project",
    filename="processed_data/y_test.csv",
    repo_type="dataset",
    token=os.getenv("HF_TOKEN")
))['ProdTaken']

print(f"✅ Data loaded - Train: {X_train.shape}, Test: {X_test.shape}")

# Define models and hyperparameter grids
models = {
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
    },
    'Gradient Boosting': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'subsample': [0.8, 0.9, 1.0]
        }
    },
    'XGBoost': {
        'model': XGBClassifier(random_state=42, eval_metric='logloss'),
        'params': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
    }
}

best_models = {}
results = []

print("\n" + "=" * 80)
print("🔧 STARTING MODEL TRAINING AND HYPERPARAMETER TUNING")
print("=" * 80)

# Train and tune each model
for model_name, model_config in models.items():
    print(f"\n{'=' * 80}")
    print(f"📊 Training: {model_name}")
    print(f"{'=' * 80}")

    with mlflow.start_run(run_name=model_name):
        # Log model type
        mlflow.log_param("model_type", model_name)

        # Perform hyperparameter tuning
        print(f"🔍 Performing RandomizedSearchCV with 10 iterations...")
        random_search = RandomizedSearchCV(
            estimator=model_config['model'],
            param_distributions=model_config['params'],
            n_iter=10,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )

        random_search.fit(X_train, y_train)
        best_model = random_search.best_estimator_

        # Log best parameters
        print(f"✅ Best parameters found:")
        for param, value in random_search.best_params_.items():
            print(f"   {param}: {value}")
            mlflow.log_param(param, value)

        # Make predictions
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        # Log metrics
        print(f"\n📈 Model Performance:")
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1-Score:  {f1:.4f}")
        print(f"   ROC-AUC:   {roc_auc:.4f}")

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        # Log model
        mlflow.sklearn.log_model(best_model, "model")

        # Store results
        best_models[model_name] = {
            'model': best_model,
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc
            }
        }

        results.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': roc_auc
        })

# Display comparison table
print("\n" + "=" * 80)
print("📊 MODEL COMPARISON")
print("=" * 80)
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Select best model based on ROC-AUC
best_model_name = max(best_models.items(), key=lambda x: x[1]['metrics']['roc_auc'])[0]
best_model_obj = best_models[best_model_name]['model']
best_metrics = best_models[best_model_name]['metrics']

print("\n" + "=" * 80)
print(f"🏆 BEST MODEL: {best_model_name}")
print("=" * 80)
print(f"ROC-AUC: {best_metrics['roc_auc']:.4f}")
print(f"Accuracy: {best_metrics['accuracy']:.4f}")
print(f"F1-Score: {best_metrics['f1_score']:.4f}")

# Save best model and artifacts
print("\n💾 Saving best model and artifacts...")
os.makedirs("tourism_project/final_model", exist_ok=True)
joblib.dump(best_model_obj, "tourism_project/final_model/model.pkl")

# Download and save scaler and encoders
scaler_path = hf_hub_download(
    repo_id="svenkateshdotnet/tourism_project",
    filename="processed_data/scaler.pkl",
    repo_type="dataset",
    token=os.getenv("HF_TOKEN")
)
encoders_path = hf_hub_download(
    repo_id="svenkateshdotnet/tourism_project",
    filename="processed_data/label_encoders.pkl",
    repo_type="dataset",
    token=os.getenv("HF_TOKEN")
)

# Copy to final_model folder
import shutil
shutil.copy(scaler_path, "tourism_project/final_model/scaler.pkl")
shutil.copy(encoders_path, "tourism_project/final_model/label_encoders.pkl")

# Save model info
with open("tourism_project/final_model/model_info.txt", "w") as f:
    f.write(f"Best Model: {best_model_name}\n")
    f.write(f"ROC-AUC: {best_metrics['roc_auc']:.4f}\n")
    f.write(f"Accuracy: {best_metrics['accuracy']:.4f}\n")
    f.write(f"Precision: {best_metrics['precision']:.4f}\n")
    f.write(f"Recall: {best_metrics['recall']:.4f}\n")
    f.write(f"F1-Score: {best_metrics['f1_score']:.4f}\n")

print("✅ Model and artifacts saved locally")

# Register model to HuggingFace Model Hub
print("\n📤 Registering model to HuggingFace Model Hub...")
model_repo_id = "svenkateshdotnet/tourism_project_model"
api = HfApi(token=os.getenv("HF_TOKEN"))

# Create model repository
try:
    api.repo_info(repo_id=model_repo_id, repo_type="model")
    print(f"✅ Model repository '{model_repo_id}' already exists")
except:
    print(f"⚠️ Creating new model repository '{model_repo_id}'...")
    create_repo(repo_id=model_repo_id, repo_type="model", private=False, token=os.getenv("HF_TOKEN"))
    print(f"✅ Model repository created!")

# Upload model files
api.upload_folder(
    folder_path="tourism_project/final_model",
    repo_id=model_repo_id,
    repo_type="model"
)

print("\n" + "=" * 80)
print("✅ MODEL TRAINING COMPLETED SUCCESSFULLY!")
print(f"🎯 Best Model: {best_model_name}")
print(f"📊 ROC-AUC Score: {best_metrics['roc_auc']:.4f}")
print(f"🔗 Model Hub: https://huggingface.co/{model_repo_id}")
print("=" * 80)
