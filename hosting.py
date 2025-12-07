from huggingface_hub import HfApi, create_repo
import os

print("=" * 80)
print("🚀 DEPLOYING TO HUGGINGFACE SPACES")
print("=" * 80)

# Define repository details
space_id = "svenkateshdotnet/tourism-package-predictor"
api = HfApi(token=os.getenv("HF_TOKEN"))

# Create Space repository
try:
    api.repo_info(repo_id=space_id, repo_type="space")
    print(f"✅ Space '{space_id}' already exists")
except:
    print(f"⚠️ Creating new Space '{space_id}'...")
    create_repo(
        repo_id=space_id,
        repo_type="space",
        space_sdk="streamlit",
        private=False,
        token=os.getenv("HF_TOKEN")
    )
    print(f"✅ Space created successfully!")

# Upload deployment files to Space
print("\n📤 Uploading deployment files to HuggingFace Space...")

# Upload app.py
api.upload_file(
    path_or_fileobj="tourism_project/deployment/app.py",
    path_in_repo="app.py",
    repo_id=space_id,
    repo_type="space"
)
print("✅ Uploaded app.py")

# Upload requirements.txt
api.upload_file(
    path_or_fileobj="tourism_project/deployment/requirements.txt",
    path_in_repo="requirements.txt",
    repo_id=space_id,
    repo_type="space"
)
print("✅ Uploaded requirements.txt")

# Upload Dockerfile (optional, but good for custom builds)
try:
    api.upload_file(
        path_or_fileobj="tourism_project/deployment/Dockerfile",
        path_in_repo="Dockerfile",
        repo_id=space_id,
        repo_type="space"
    )
    print("✅ Uploaded Dockerfile")
except Exception as e:
    print(f"⚠️ Dockerfile upload skipped: {e}")

print("\n" + "=" * 80)
print("✅ DEPLOYMENT COMPLETED SUCCESSFULLY!")
print(f"🌐 Space URL: https://huggingface.co/spaces/{space_id}")
print("⏳ The Space will build and deploy automatically in a few minutes")
print("=" * 80)
