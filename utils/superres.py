# === FILE: utils/superres.py ===
import cv2
import numpy as np
from PIL import Image
import os

# Try to import Real-ESRGAN components
try:
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    REALESRGAN_AVAILABLE = True
except ImportError:
    REALESRGAN_AVAILABLE = False
    print("Warning: Real-ESRGAN not available. Super-resolution will be disabled.")

def load_sr_model():
    """
    Load Real-ESRGAN super-resolution model
    Model will be downloaded automatically on first use
    """
    if not REALESRGAN_AVAILABLE:
        return None
    
    try:
        # Define model architecture
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4
        )
        
        # Path for cached model
        cache_dir = os.path.expanduser("~/.cache/realesrgan")
        os.makedirs(cache_dir, exist_ok=True)
        model_path = os.path.join(cache_dir, "RealESRGAN_x4plus.pth")
        
        # Initialize the upscaler
        sr_model = RealESRGANer(
            scale=4,
            model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            model=model,
            tile=256,
            tile_pad=10,
            pre_pad=0,
            half=False  # Use FP32 for better compatibility
        )
        
        return sr_model
    
    except Exception as e:
        print(f"Warning: Could not load super-resolution model: {e}")
        return None

def apply_sr(pil_image: Image.Image, sr_model, min_size: int = 800) -> Image.Image:
    """
    Apply super-resolution to image if it's below minimum size
    
    Args:
        pil_image: Input PIL Image
        sr_model: Loaded Real-ESRGAN model (or None)
        min_size: Minimum dimension threshold (pixels)
    
    Returns:
        Upscaled PIL Image (or original if conditions not met)
    """
    # Skip if no model available
    if sr_model is None:
        return pil_image
    
    # Only apply if image is below threshold
    width, height = pil_image.size
    if max(width, height) >= min_size:
        return pil_image
    
    try:
        # Convert PIL to OpenCV format (RGB to BGR)
        img = pil_image.convert('RGB')
        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Apply super-resolution
        sr_bgr, _ = sr_model.enhance(img_bgr, outscale=4)
        
        # Convert back to RGB PIL
        sr_rgb = cv2.cvtColor(sr_bgr, cv2.COLOR_BGR2RGB)
        result = Image.fromarray(sr_rgb)
        
        return result
    
    except Exception as e:
        print(f"Warning: Super-resolution failed: {e}")
        return pil_image

def apply_sr_if_needed(pil_image: Image.Image, sr_model) -> tuple:
    """
    Apply SR and return both original and upscaled for comparison
    
    Returns:
        tuple: (processed_image, was_sr_applied)
    """
    width, height = pil_image.size
    if max(width, height) < 800 and sr_model is not None:
        return apply_sr(pil_image, sr_model), True
    return pil_image, False