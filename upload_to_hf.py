from huggingface_hub import HfApi
import os

# Initialize the Hugging Face API
api = HfApi()

# Paths to the checkpoint folders
checkpoint_folders = [
    "/workspace/LLaVA-LLM-Encoder/checkpoints/llava-v1.5-7b-gpt2vision-scratch-stage2",
]

# Login to Hugging Face (this will prompt for token if not logged in)
print("Please enter your Hugging Face token when prompted")
api.token = input("Enter your Hugging Face token: ")

# Get the username
username = input("Enter your Hugging Face username: ")

# Upload files from each folder
for folder_path in checkpoint_folders:
    # Extract folder name for repo name
    folder_name = os.path.basename(folder_path)
    repo_name = f"{username}/{folder_name}"
    
    print(f"\nProcessing folder: {folder_path}")
    print(f"Repository will be: {repo_name}")
    
    # Create repository if it doesn't exist
    try:
        api.create_repo(repo_id=repo_name, exist_ok=True)
        print(f"Repository '{repo_name}' is ready")
    except Exception as e:
        print(f"Error creating repository: {e}")
        continue  # Skip to next folder if repo creation fails
    
    # Get all files in the folder
    files_to_upload = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            # Calculate relative path for uploading
            rel_path = os.path.relpath(file_path, folder_path)
            files_to_upload.append((file_path, rel_path))
    
    # Upload each file
    for file_path, rel_path in files_to_upload:
        try:
            print(f"Uploading: {rel_path}...")
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=rel_path,
                repo_id=repo_name,
            )
            print(f"Successfully uploaded {rel_path}")
        except Exception as e:
            print(f"Error uploading {rel_path}: {e}")
    
    print(f"\nCompleted uploading files to {repo_name}")
    print(f"You can access your files at: https://huggingface.co/{repo_name}")