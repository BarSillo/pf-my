import sys
import os

# Get the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
rlhedge_path = os.path.join(project_root, 'rlhedge')

# Add paths to Python path if they're not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if rlhedge_path not in sys.path:
    sys.path.insert(0, rlhedge_path)

print(f"Added to PYTHONPATH:\n{project_root}\n{rlhedge_path}")
