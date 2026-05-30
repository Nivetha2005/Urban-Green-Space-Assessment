# === FILE: utils/analysis.py ===
import numpy as np
import torch
import torch.nn as nn
import cv2
from PIL import Image

def run_mc_dropout(model, image_tensor, device, n_passes=20):
    """
    Monte Carlo Dropout for uncertainty estimation
    """
    model.train()  # Enable dropout
    all_probs = []
    
    with torch.no_grad():
        for _ in range(n_passes):
            logits = model(image_tensor.to(device), enable_dropout=True)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
    
    # Stack all predictions
    all_probs = np.stack(all_probs, axis=0)  # (n_passes, 1, 6, H, W)
    mean_probs = np.mean(all_probs, axis=0)[0]  # (6, H, W)
    std_probs = np.std(all_probs, axis=0)[0]
    
    # Compute entropy map: -sum(p * log(p))
    entropy_map = -np.sum(mean_probs * np.log(mean_probs + 1e-8), axis=0)
    
    # Get mean prediction
    mean_pred = np.argmax(mean_probs, axis=0).astype(np.int32)
    
    # Sample GVI distribution from probability maps
    np.random.seed(42)
    h, w = mean_probs.shape[1], mean_probs.shape[2]
    n_samples = 50
    gvi_samples = []
    
    for _ in range(n_samples):
        # Sample a class for each pixel based on probabilities
        flat_probs = mean_probs.reshape(6, -1).T  # (H*W, 6)
        sampled_classes = np.array([
            np.random.choice(6, p=flat_probs[i]) for i in range(h * w)
        ])
        sampled_mask = sampled_classes.reshape(h, w)
        gvi = ((sampled_mask == 2).sum() + (sampled_mask == 3).sum()) / (h * w) * 100
        gvi_samples.append(gvi)
    
    gvi_mean = float(np.mean(gvi_samples))
    gvi_ci_lower = float(np.percentile(gvi_samples, 2.5))
    gvi_ci_upper = float(np.percentile(gvi_samples, 97.5))
    
    return {
        'mean_pred': mean_pred,
        'entropy_map': entropy_map,
        'gvi_mean': gvi_mean,
        'gvi_ci_lower': gvi_ci_lower,
        'gvi_ci_upper': gvi_ci_upper,
    }

def compute_discrepancy(aerial_gvi: float, street_gvi: float) -> dict:
    """
    Compute discrepancy between aerial and street-level GVI
    """
    discrepancy = float(aerial_gvi - street_gvi)
    
    if discrepancy > 15:
        label = "Hidden Canopy"
        recommendation = "Aerial imagery shows more vegetation than what pedestrians see. Consider creating green corridors and improving street-level visibility of existing canopy."
    elif discrepancy < -15:
        label = "Vertical Greenery Dominance"
        recommendation = "Street-level vegetation exceeds aerial coverage. This indicates strong vertical greening. Plan for canopy expansion to balance the green infrastructure."
    else:
        label = "Consistent Green Visibility"
        recommendation = "Good alignment between aerial and street-level greenery. Maintain current vegetation management practices and monitor for changes."
    
    return {
        'discrepancy': discrepancy,
        'label': label,
        'recommendation': recommendation
    }

def compute_uncertainty_weighted_fusion(aerial_gvi, street_gvi, entropy_map):
    """
    Fuse aerial and street GVI based on uncertainty (entropy)
    Higher uncertainty = more weight on street view (ground truth)
    """
    # Normalize entropy to [0, 1]
    entropy_min = np.min(entropy_map)
    entropy_max = np.max(entropy_map)
    if entropy_max - entropy_min > 1e-8:
        normalized_entropy = (entropy_map - entropy_min) / (entropy_max - entropy_min)
    else:
        normalized_entropy = np.zeros_like(entropy_map)
    
    uncertainty_weight = np.clip(np.mean(normalized_entropy), 0.0, 1.0)
    aerial_weight = 1.0 - uncertainty_weight
    street_weight = uncertainty_weight
    
    fused_gvi = float(aerial_weight * aerial_gvi + street_weight * street_gvi)
    
    return {
        'fused_gvi': fused_gvi,
        'aerial_weight': aerial_weight,
        'street_weight': street_weight,
        'description': f"Fusion weights: Aerial={aerial_weight:.2f}, Street={street_weight:.2f}"
    }

def detect_hotspots(pred_mask, entropy_map, aerial_gvi, street_gvi):
    """
    Detect hotspot areas (high discrepancy + high uncertainty)
    """
    h, w = pred_mask.shape
    patch_h = h // 4
    patch_w = w // 4
    
    discrepancy = abs(aerial_gvi - street_gvi)
    hotspot_scores = np.zeros((4, 4), dtype=float)
    patch_gvis = []
    flagged_patches = []
    
    for i in range(4):
        for j in range(4):
            y1, y2 = i * patch_h, h if i == 3 else (i + 1) * patch_h
            x1, x2 = j * patch_w, w if j == 3 else (j + 1) * patch_w
            
            submask = pred_mask[y1:y2, x1:x2]
            subentropy = entropy_map[y1:y2, x1:x2]
            
            patch_gvi = ((submask == 2).sum() + (submask == 3).sum()) / (submask.size + 1e-8) * 100
            patch_uncert = float(np.mean(subentropy))
            
            # Score = discrepancy * uncertainty
            score = discrepancy * patch_uncert
            hotspot_scores[i, j] = score
            patch_gvis.append(float(patch_gvi))
    
    # Flag top 25% patches as hotspots
    threshold = np.percentile(hotspot_scores, 75)
    for i in range(4):
        for j in range(4):
            if hotspot_scores[i, j] > threshold:
                flagged_patches.append({
                    'row': i, 
                    'col': j, 
                    'score': float(hotspot_scores[i, j]),
                    'patch_gvi': patch_gvis[i * 4 + j]
                })
    
    return {
        'flagged_patches': flagged_patches,
        'hotspot_scores': hotspot_scores.tolist(),
        'patch_gvis': patch_gvis,
    }

def compute_equity_index(pred_mask):
    """
    Compute Gini coefficient for green space equity across 4x4 patches
    """
    h, w = pred_mask.shape
    patch_h = h // 4
    patch_w = w // 4
    
    gvis = []
    gvi_map = np.zeros_like(pred_mask, dtype=float)
    
    for i in range(4):
        for j in range(4):
            y1, y2 = i * patch_h, h if i == 3 else (i + 1) * patch_h
            x1, x2 = j * patch_w, w if j == 3 else (j + 1) * patch_w
            
            patch = pred_mask[y1:y2, x1:x2]
            patch_gvi = ((patch == 2).sum() + (patch == 3).sum()) / (patch.size + 1e-8) * 100
            gvis.append(patch_gvi)
            gvi_map[y1:y2, x1:x2] = patch_gvi
    
    # Calculate Gini coefficient
    gvis_arr = np.array(sorted(gvis))
    n = len(gvis_arr)
    if np.sum(gvis_arr) == 0:
        gini = 0.0
    else:
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * gvis_arr)) / (n * np.sum(gvis_arr)) - (n + 1) / n
    
    return {
        'gini_coef': float(gini),
        'patch_gvis': [float(v) for v in gvis],
        'gvi_map': gvi_map
    }

def compute_gradcam(model, image_tensor, device):
    """
    Compute Grad-CAM for Low Vegetation (class 2) and Tree (class 3)
    """
    model.eval()
    activations = {}
    gradients = {}
    
    def forward_hook(module, input, output):
        activations['value'] = output.detach().cpu()
    
    def backward_hook(module, grad_in, grad_out):
        gradients['value'] = grad_out[0].detach().cpu()
    
    # Hook the last convolutional layer (u4.conv[-2] is the second conv in u4 block)
    # Fallback to u4.conv if u4.conv[-2] doesn't exist
    target_layer = None
    if hasattr(model, 'u4') and hasattr(model.u4, 'conv'):
        if isinstance(model.u4.conv, nn.Sequential) and len(model.u4.conv) >= 2:
            target_layer = model.u4.conv[-2]
        else:
            target_layer = model.u4.conv
    
    if target_layer is None:
        # Fallback: return dummy maps
        h, w = image_tensor.shape[2], image_tensor.shape[3]
        return {'cam_lowveg': np.zeros((h, w)), 'cam_tree': np.zeros((h, w))}
    
    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_backward_hook(backward_hook)
    
    image_tensor = image_tensor.to(device)
    image_tensor.requires_grad_(True)
    logits = model(image_tensor, enable_dropout=False)
    
    cams = {}
    for cls in [2, 3]:  # Low vegetation and Tree
        model.zero_grad()
        target = logits[0, cls].mean()
        target.backward(retain_graph=True)
        
        act = activations['value'][0]  # (C, H, W)
        grad = gradients['value'][0]   # (C, H, W)
        
        # Global average pooling of gradients
        weights = grad.mean(dim=(1, 2))  # (C,)
        
        # Weighted sum of activations
        cam = torch.relu(torch.sum(weights.view(-1, 1, 1) * act, dim=0)).numpy()
        
        # Resize to original input size
        cam = cv2.resize(cam, (image_tensor.shape[3], image_tensor.shape[2]))
        
        # Normalize to [0, 1]
        if cam.max() - cam.min() > 1e-8:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)
        
        cams[f'cam_{"lowveg" if cls == 2 else "tree"}'] = cam
    
    handle_fwd.remove()
    handle_bwd.remove()
    
    return {
        'cam_lowveg': cams['cam_lowveg'],
        'cam_tree': cams['cam_tree']
    }

def simulate_winter(model, image_pil, pred_mask, device):
    """
    Simulate winter appearance by modifying vegetation pixels
    """
    image_np = np.array(image_pil.convert('RGB')).astype(np.uint8)
    
    # Resize pred_mask to match image size if needed
    if pred_mask.shape[:2] != (image_np.shape[0], image_np.shape[1]):
        pred_mask_resized = cv2.resize(
            pred_mask.astype(np.uint8),
            (image_np.shape[1], image_np.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
    else:
        pred_mask_resized = pred_mask
    
    # Create winter effect on vegetation pixels
    winter_image = image_np.copy().astype(float)
    veg_mask = np.isin(pred_mask_resized, [2, 3])  # Low vegetation and trees
    
    # Convert to HSV for vegetation modification
    hsv = cv2.cvtColor(winter_image.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(float)
    
    # Apply winter transformations to vegetation areas
    hsv[veg_mask, 1] = np.clip(hsv[veg_mask, 1] * 0.3, 0, 255)  # Reduce saturation
    hsv[veg_mask, 0] = (hsv[veg_mask, 0] + 25) % 180  # Shift hue toward brown
    hsv[veg_mask, 2] = np.clip(hsv[veg_mask, 2] * 0.85, 0, 255)  # Reduce brightness
    
    winter_rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    # Blend with original
    blended = image_np.copy()
    blended[veg_mask] = winter_rgb[veg_mask]
    winter_image = blended.astype(np.uint8)
    
    # Compute GVI for summer (from original mask)
    summer_gvi = ((pred_mask == 2).sum() + (pred_mask == 3).sum()) / (pred_mask.size + 1e-8) * 100
    
    # Run inference on winter image to get winter GVI
    winter_pil = Image.fromarray(winter_image)
    winter_pred, _ = run_unet(model, winter_pil, device)
    winter_gvi = ((winter_pred == 2).sum() + (winter_pred == 3).sum()) / (winter_pred.size + 1e-8) * 100
    delta = winter_gvi - summer_gvi
    
    return {
        'winter_image': winter_image,
        'winter_gvi': float(winter_gvi),
        'summer_gvi': float(summer_gvi),
        'delta': float(delta),
    }

def compute_cooling(pred_mask):
    """
    Compute urban cooling potential based on Bowler et al. 2010
    Coefficient: 0.3°C per 10% GVI
    """
    h, w = pred_mask.shape
    patch_h = h // 4
    patch_w = w // 4
    
    cooling_map = np.zeros((h, w), dtype=float)
    patch_cooling = []
    
    for i in range(4):
        for j in range(4):
            y1, y2 = i * patch_h, h if i == 3 else (i + 1) * patch_h
            x1, x2 = j * patch_w, w if j == 3 else (j + 1) * patch_w
            
            patch = pred_mask[y1:y2, x1:x2]
            patch_gvi = ((patch == 2).sum() + (patch == 3).sum()) / (patch.size + 1e-8) * 100
            
            # Cooling = (GVI / 10) * 0.3 °C
            cooling = (patch_gvi / 10.0) * 0.3
            cooling_map[y1:y2, x1:x2] = cooling
            patch_cooling.append(float(cooling))
    
    city_avg_cooling = float(np.mean(patch_cooling))
    
    # Classify cooling potential
    if city_avg_cooling < 0.5:
        cooling_label = "Low"
    elif city_avg_cooling < 1.0:
        cooling_label = "Moderate"
    else:
        cooling_label = "High"
    
    return {
        'cooling_map': cooling_map,
        'city_avg_cooling': city_avg_cooling,
        'patch_cooling_values': patch_cooling,
        'cooling_label': cooling_label
    }

# Import run_unet for seasonal simulation
from utils.inference import run_unet