# utils.py
import numpy as np
from PIL import Image
import io

# CIFAR-10 labels
LABELS = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

def preprocess_image_bytes(image_bytes):
    """Take file bytes, return normalized numpy array shape (1,32,32,3)."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((32,32))
    arr = np.array(image).astype("float32") / 255.0
    arr = np.expand_dims(arr, 0)
    return arr

def get_labels():
    return LABELS
