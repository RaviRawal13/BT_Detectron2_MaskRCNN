from pathlib import Path
import sys

# Get the absolute path of the current file
file_path = Path(__file__).resolve()

# Get the parent directory of the current file
root_path = file_path.parent

# Add the root path to the sys.path list if it is not already there
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Get the relative path of the root directory with respect to the current working directory
ROOT = root_path.relative_to(Path.cwd())

# Sources
IMAGE = 'Image'
# Images config
IMAGES_DIR = ROOT / 'default'
DEFAULT_IMAGE = IMAGES_DIR / 'y781_jpg.rf.ee3ab96815e77047e207cfc76b59937b.jpg'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'y781_jpg.rf.ee3ab96815e77047e207cfc76b59937b_detected.png'

# ML Model config 
MODEL_DIR = ROOT / 'output' 
SEGMENTATION_MODEL = MODEL_DIR / 'model_final.pth'


