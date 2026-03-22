import io
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import sys
import os

# Add parent directory to path so it can import the original model.py if pickled that way
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Output class names in order
CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]

# Input image size
IMG_SIZE = 224

# Preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_model(model_path="model.pth"):
    # Load model as specified by user requirements
    # weights_only=False because the user might have saved the full model object
    model = torch.load(model_path, map_location='cpu', weights_only=False)
    model.eval()
    return model

def predict(model, image_bytes: bytes):
    # Convert uploaded bytes -> PIL Image -> RGB
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Preprocess
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

    with torch.no_grad():
        outputs = model(input_batch)
        
    # Apply softmax to logits
    probabilities = F.softmax(outputs, dim=1).squeeze().tolist()
    
    # Extract prediction
    predicted_idx = probabilities.index(max(probabilities))
    predicted_label = CLASS_NAMES[predicted_idx]
    confidence = probabilities[predicted_idx]
    
    # Format all scores
    all_scores = {CLASS_NAMES[i]: probabilities[i] for i in range(len(CLASS_NAMES))}
    
    return predicted_label, confidence, all_scores
