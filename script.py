import os

# Create project directory structure
project_dirs = [
    "cuda_histogram_project",
    "cuda_histogram_project/src",
    "cuda_histogram_project/data", 
    "cuda_histogram_project/output",
    "cuda_histogram_project/build"
]

for dir_path in project_dirs:
    os.makedirs(dir_path, exist_ok=True)
    print(f"Created directory: {dir_path}")