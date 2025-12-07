from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os

# Define HuggingFace repository details
repo_id = "svenkateshdotnet/tourism_project"
repo_type = "dataset"

# Initialize API client with token from environment variable
api = HfApi(token=os.getenv("HF_TOKEN"))

# Check if the dataset repository exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"✅ Repository '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"⚠️ Repository '{repo_id}' not found. Creating new repository...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False, token=os.getenv("HF_TOKEN"))
    print(f"✅ Repository '{repo_id}' created successfully!")

# Get the data folder path (relative to script location)
# Script is in: model_building/data_register.py
# Data folder is in: data/
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # Go up one level from model_building to project root
data_folder = os.path.join(project_root, "data")

print(f"📁 Looking for data folder at: {data_folder}")

if not os.path.isdir(data_folder):
    raise ValueError(f"Data folder not found at: {data_folder}")

# Upload the data folder to HuggingFace
print("📤 Uploading tourism.csv to HuggingFace...")
api.upload_folder(
    folder_path=data_folder,
    repo_id=repo_id,
    repo_type=repo_type,
)
print("✅ Data registration completed successfully!")
