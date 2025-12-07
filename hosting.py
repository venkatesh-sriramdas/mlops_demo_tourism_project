from huggingface_hub import HfApi, create_repo
import os
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

logger.info("=" * 80)
logger.info("🚀 DEPLOYING TO HUGGINGFACE SPACES")
logger.info("=" * 80)

# Define repository details
space_id = "svenkateshdotnet/tourism-package-predictor"
logger.info(f"Space ID: {space_id}")
logger.info(f"HF_TOKEN present: {bool(os.getenv('HF_TOKEN'))}")

api = HfApi(token=os.getenv("HF_TOKEN"))

# Create Space repository
logger.info("Checking if Space repository exists...")
try:
    repo_info = api.repo_info(repo_id=space_id, repo_type="space")
    logger.info(f"✅ Space '{space_id}' already exists")
    logger.info(f"   Space SDK: {repo_info.cardData.get('sdk', 'unknown') if repo_info.cardData else 'unknown'}")
except Exception as e:
    logger.warning(f"Space not found: {e}")
    logger.info(f"⚠️ Creating new Space '{space_id}'...")
    try:
        create_repo(
            repo_id=space_id,
            repo_type="space",
            space_sdk="StreamLit",
            private=False,
            token=os.getenv("HF_TOKEN")
        )
        logger.info(f"✅ Space created successfully!")
    except Exception as create_error:
        logger.error(f"Failed to create Space: {create_error}")
        raise

# Upload deployment files to Space
logger.info("")
logger.info("=" * 80)
logger.info("📤 UPLOADING DEPLOYMENT FILES")
logger.info("=" * 80)

# Get the script directory and navigate to correct paths
script_dir = os.path.dirname(os.path.abspath(__file__))
deployment_dir = os.path.join(script_dir, "deployment")

logger.info(f"Script directory: {script_dir}")
logger.info(f"Deployment directory: {deployment_dir}")
logger.info(f"Deployment directory exists: {os.path.isdir(deployment_dir)}")

# Upload app.py
app_path = os.path.join(deployment_dir, "app.py")
logger.info(f"\n📁 Uploading app.py...")
logger.info(f"   Source path: {app_path}")
logger.info(f"   File exists: {os.path.isfile(app_path)}")
logger.info(f"   File size: {os.path.getsize(app_path) / 1024:.2f} KB")

try:
    api.upload_file(
        path_or_fileobj=app_path,
        path_in_repo="app.py",
        repo_id=space_id,
        repo_type="space"
    )
    logger.info("✅ Uploaded app.py successfully")
except Exception as e:
    logger.error(f"❌ Failed to upload app.py: {e}")
    raise

# Upload requirements.txt
requirements_path = os.path.join(deployment_dir, "requirements.txt")
logger.info(f"\n📁 Uploading requirements.txt...")
logger.info(f"   Source path: {requirements_path}")
logger.info(f"   File exists: {os.path.isfile(requirements_path)}")
logger.info(f"   File size: {os.path.getsize(requirements_path)} bytes")

try:
    api.upload_file(
        path_or_fileobj=requirements_path,
        path_in_repo="requirements.txt",
        repo_id=space_id,
        repo_type="space"
    )
    logger.info("✅ Uploaded requirements.txt successfully")
except Exception as e:
    logger.error(f"❌ Failed to upload requirements.txt: {e}")
    raise

# Upload Dockerfile
dockerfile_path = os.path.join(deployment_dir, "Dockerfile")
logger.info(f"\n📁 Uploading Dockerfile...")
logger.info(f"   Source path: {dockerfile_path}")
logger.info(f"   File exists: {os.path.isfile(dockerfile_path)}")

try:
    logger.info(f"   File size: {os.path.getsize(dockerfile_path)} bytes")
    api.upload_file(
        path_or_fileobj=dockerfile_path,
        path_in_repo="Dockerfile",
        repo_id=space_id,
        repo_type="space"
    )
    logger.info("✅ Uploaded Dockerfile successfully")
except Exception as e:
    logger.warning(f"⚠️ Dockerfile upload skipped: {e}")

logger.info("")
logger.info("=" * 80)
logger.info("✅ DEPLOYMENT COMPLETED SUCCESSFULLY!")
logger.info("=" * 80)
logger.info(f"🌐 Space URL: https://huggingface.co/spaces/{space_id}")
logger.info("⏳ The Space will build and deploy automatically in a few minutes")
logger.info("📋 Check build logs at: https://huggingface.co/spaces/{}/settings".format(space_id))
logger.info("=" * 80)
