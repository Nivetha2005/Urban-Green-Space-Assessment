# === FILE: utils/inference.py ===
import numpy as np
import torch
import cv2
from PIL import Image
from ultralytics import YOLO
import sys
import os

# Add parent directory to path for model imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.unet_model import UNetWithDropout, mask_to_rgb

def load_unet(model_path: str, device: str):
    """
    Load UNet model from checkpoint file
    """
    try:
        model = UNetWithDropout(in_channels=3, out_channels=6, dropout_rate=0.3)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Remap keys if needed (handle bottleneck naming differences)
        new_state_dict = {}
        for k, v in state_dict.items():
            # Handle potential key mismatches
            if k.startswith('b.conv.'):
                new_k = k.replace('b.conv.', 'b.0.conv.')
                new_state_dict[new_k] = v
            else:
                new_state_dict[k] = v
        
        # Load with strict=False to handle minor architecture differences
        model.load_state_dict(new_state_dict, strict=False)
        model.to(device)
        model.eval()
        
        return model
    
    except Exception as e:
        raise RuntimeError(f"Failed to load UNet model from {model_path}: {e}")

def run_unet(model, image_pil: Image.Image, device: str):
    """
    Run UNet inference on a PIL image
    Returns: pred_mask (256x256 numpy int), pred_rgb (256x256x3 numpy uint8)
    """
    try:
        # Convert and resize to 256x256 (model input size)
        image = image_pil.convert('RGB')
        image_resized = image.resize((256, 256), Image.BILINEAR)
        
        # Normalize and convert to tensor
        arr = np.array(image_resized).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            logits = model(tensor, enable_dropout=False)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy().astype(np.int32)
        
        # Convert to RGB for visualization
        pred_rgb = mask_to_rgb(pred)
        
        return pred, pred_rgb
    
    except Exception as e:
        raise RuntimeError(f"UNet inference failed: {e}")

def load_yolo(model_path: str):
    """
    Load YOLO model for street-level segmentation
    """
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load YOLO model from {model_path}: {e}")

def run_yolo(model, image_pil: Image.Image):
    """
    Run YOLO inference on street view image
    Returns: street_gvi (float), annotated_image (numpy uint8)
    
    Cityscapes classes:
    - class 8: vegetation
    - class 9: terrain
    """
    try:
        image = image_pil.convert('RGB')
        img_np = np.array(image)
        h, w = img_np.shape[:2]
        
        # Run inference
        results = model(img_np)
        
        if len(results) == 0:
            raise RuntimeError("YOLO returned no results")
        
        result = results[0]
        total_pixels = h * w
        veg_pixels = 0
        terrain_pixels = 0
        
        # Initialize class map
        class_map = np.zeros((h, w), dtype=np.int32)
        
        # Process masks if available
        if hasattr(result, 'masks') and result.masks is not None and len(result.masks.data) > 0:
            mask_data = result.masks.data.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            for i, m in enumerate(mask_data):
                cls = cls_ids[i]
                
                # Resize mask to original image size
                m_resized = cv2.resize(
                    m.astype(np.uint8),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST
                )
                
                class_map[m_resized.astype(bool)] = cls
        else:
            # Fallback: rasterize bounding boxes
            if hasattr(result, 'boxes') and len(result.boxes.xyxy) > 0:
                for box, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy().astype(int)):
                    x1, y1, x2, y2 = map(int, box)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w - 1, x2), min(h - 1, y2)
                    class_map[y1:y2 + 1, x1:x2 + 1] = cls
        
        # Calculate vegetation pixels (class 8) and terrain pixels (class 9)
        veg_pixels = int((class_map == 8).sum())
        terrain_pixels = int((class_map == 9).sum())
        
        # GVI = (vegetation + terrain) / total * 100
        street_gvi = (veg_pixels + terrain_pixels) / total_pixels * 100 if total_pixels > 0 else 0.0
        
        # Create annotated image
        annotated = img_np.copy()
        if hasattr(result, 'boxes') and len(result.boxes.xyxy) > 0:
            for box, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy().astype(int)):
                x1, y1, x2, y2 = map(int, box)
                # Green for vegetation/terrain, red for others
                color = (0, 255, 0) if cls in (8, 9) else (255, 0, 0)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = "Veg" if cls in (8, 9) else f"c{cls}"
                cv2.putText(annotated, label, (x1, max(y1 - 8, 0)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return float(street_gvi), annotated
    
    except Exception as e:
        raise RuntimeError(f"YOLO inference failed: {e}")