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

# Upload the data folder to HuggingFace
print("📤 Uploading tourism.csv to HuggingFace...")
api.upload_folder(
    folder_path="tourism_project/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
print("✅ Data registration completed successfully!")
