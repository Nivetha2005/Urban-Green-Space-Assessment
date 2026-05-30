# === FILE: utils/visualizations.py ===
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.patches as mpatches

# Class colors and names for consistent visualization
CLASS_COLORS = {
    0: [255, 255, 255],  # Impervious
    1: [0, 0, 255],      # Building
    2: [0, 255, 255],    # Low vegetation
    3: [0, 255, 0],      # Tree
    4: [255, 255, 0],    # Car
    5: [255, 0, 0]       # Clutter
}

CLASS_NAMES = {
    0: "Impervious",
    1: "Building", 
    2: "Low Veg",
    3: "Tree",
    4: "Car",
    5: "Clutter"
}

def plot_segmentation_overlay(image_np, pred_mask, class_colors=None, class_names=None) -> Figure:
    """Plot segmentation mask overlaid on original image"""
    if class_colors is None:
        class_colors = CLASS_COLORS
    if class_names is None:
        class_names = CLASS_NAMES
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Normalize image
    if image_np.max() > 1:
        overlay = image_np.astype(float) / 255.0
    else:
        overlay = image_np.copy()
    
    # Create RGB mask
    mask_rgb = np.zeros_like(image_np, dtype=np.float32)
    if image_np.max() > 1:
        mask_rgb = mask_rgb / 255.0
    
    for cls, color in class_colors.items():
        mask_rgb[pred_mask == cls] = np.array(color) / 255.0
    
    # Blend
    blended = np.clip(overlay * 0.5 + mask_rgb * 0.5, 0, 1)
    ax.imshow(blended)
    ax.set_title('Segmentation Overlay', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Legend
    legend_patches = []
    for cls, name in class_names.items():
        patch = mpatches.Patch(color=np.array(class_colors[cls]) / 255.0, label=f"{name}")
        legend_patches.append(patch)
    ax.legend(handles=legend_patches, loc='lower left', fontsize=8, ncol=2)
    
    plt.tight_layout()
    return fig

def plot_uncertainty_heatmap(image_np, entropy_map, gvi_mean, ci_lower, ci_upper) -> Figure:
    """Plot uncertainty heatmap and GVI confidence interval"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Normalize image for display
    if image_np.max() > 1:
        display_img = image_np.astype(np.uint8)
    else:
        display_img = (image_np * 255).astype(np.uint8)
    
    # Left: Entropy heatmap
    axes[0].imshow(display_img)
    im = axes[0].imshow(entropy_map, cmap='hot', alpha=0.6)
    axes[0].set_title('Prediction Uncertainty (Entropy)', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, label='Entropy')
    
    # Right: GVI confidence interval
    bars = axes[1].bar(['GVI Mean', 'CI Lower', 'CI Upper'], 
                       [gvi_mean, ci_lower, ci_upper],
                       color=['#2E8B57', '#FF6B6B', '#FF6B6B'])
    axes[1].set_ylabel('Green Visibility Index (%)', fontsize=11)
    axes[1].set_title(f'GVI: {gvi_mean:.1f}% [95% CI: {ci_lower:.1f}-{ci_upper:.1f}%]', 
                      fontsize=12, fontweight='bold')
    axes[1].axhline(y=gvi_mean, color='green', linestyle='--', alpha=0.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, [gvi_mean, ci_lower, ci_upper]):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    return fig

def plot_equity_map(image_np, gvi_map, patch_gvis, gini_coef) -> Figure:
    """Plot equity choropleth, Lorenz curve, and patch ranking"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Normalize image
    if image_np.max() > 1:
        display_img = image_np.astype(np.uint8)
    else:
        display_img = (image_np * 255).astype(np.uint8)
    
    # Left: Equity choropleth
    axes[0].imshow(display_img)
    im = axes[0].imshow(gvi_map, cmap='RdYlGn', alpha=0.6, vmin=0, vmax=100)
    axes[0].set_title('Green Space Equity Map', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, label='GVI (%)')
    
    # Middle: Lorenz curve
    sorted_gvis = np.sort(patch_gvis)
    cumsum = np.cumsum(sorted_gvis)
    lorenz = cumsum / (cumsum[-1] + 1e-8)
    lorenz = np.insert(lorenz, 0, 0)
    
    x = np.linspace(0, 1, len(lorenz))
    axes[1].plot(x, lorenz, 'b-', linewidth=2, label='Lorenz Curve')
    axes[1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect Equality')
    axes[1].fill_between(x, lorenz, x, alpha=0.3, color='gray')
    axes[1].set_xlabel('Population Proportion (Patches)', fontsize=11)
    axes[1].set_ylabel('Cumulative Green Space', fontsize=11)
    axes[1].set_title(f'Lorenz Curve (Gini = {gini_coef:.3f})', fontsize=12, fontweight='bold')
    axes[1].legend(loc='lower right')
    axes[1].grid(True, alpha=0.3)
    
    # Right: Patch ranking
    patches = np.arange(1, len(patch_gvis) + 1)
    colors = plt.cm.RdYlGn(np.array(patch_gvis) / 100)
    axes[2].barh(patches, sorted(patch_gvis), color=colors)
    axes[2].set_xlabel('GVI (%)', fontsize=11)
    axes[2].set_ylabel('Patch Rank (1 = lowest GVI)', fontsize=11)
    axes[2].set_title('Patch GVI Distribution', fontsize=12, fontweight='bold')
    axes[2].axvline(x=np.mean(patch_gvis), color='blue', linestyle='--', 
                    label=f'Mean: {np.mean(patch_gvis):.1f}%')
    axes[2].legend()
    
    plt.tight_layout()
    return fig

def plot_gradcam(image_np, cam_lowveg, cam_tree) -> Figure:
    """Plot Grad-CAM visualizations for vegetation classes"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Normalize image
    if image_np.max() > 1:
        display_img = image_np.astype(np.uint8)
    else:
        display_img = (image_np * 255).astype(np.uint8)
    
    # Original image
    axes[0].imshow(display_img)
    axes[0].set_title('Original Satellite Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Low vegetation CAM
    axes[1].imshow(display_img)
    im1 = axes[1].imshow(cam_lowveg, cmap='hot', alpha=0.5)
    axes[1].set_title('Low Vegetation Attention (Grad-CAM)', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label='Attention')
    
    # Tree CAM
    axes[2].imshow(display_img)
    im2 = axes[2].imshow(cam_tree, cmap='hot', alpha=0.5)
    axes[2].set_title('Tree Attention (Grad-CAM)', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label='Attention')
    
    plt.tight_layout()
    return fig

def plot_seasonal(image_np, winter_image_np, summer_gvi, winter_gvi, delta) -> Figure:
    """Plot seasonal simulation comparison"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Normalize images
    if image_np.max() > 1:
        summer_img = image_np.astype(np.uint8)
    else:
        summer_img = (image_np * 255).astype(np.uint8)
    
    if winter_image_np.max() > 1:
        winter_img = winter_image_np.astype(np.uint8)
    else:
        winter_img = (winter_image_np * 255).astype(np.uint8)
    
    # Summer image
    axes[0].imshow(summer_img)
    axes[0].set_title('Summer (Current)', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Winter simulation
    axes[1].imshow(winter_img)
    axes[1].set_title('Winter Simulation', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # GVI comparison bar chart
    bars = axes[2].bar(['Summer', 'Winter'], [summer_gvi, winter_gvi], 
                       color=['#2E8B57', '#4682B4'])
    axes[2].set_ylabel('Green Visibility Index (%)', fontsize=11)
    axes[2].set_title(f'Seasonal GVI Change: Δ = {delta:+.1f}%', fontsize=12, fontweight='bold')
    
    # Add value labels
    for bar, val in zip(bars, [summer_gvi, winter_gvi]):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # Color code the delta
    if delta > 0:
        axes[2].text(0.5, 0.95, f'↑ Increase of {delta:.1f}%', transform=axes[2].transAxes,
                    ha='center', fontsize=11, color='green', fontweight='bold')
    else:
        axes[2].text(0.5, 0.95, f'↓ Decrease of {abs(delta):.1f}%', transform=axes[2].transAxes,
                    ha='center', fontsize=11, color='red', fontweight='bold')
    
    plt.tight_layout()
    return fig

def plot_cooling_map(image_np, cooling_map, city_avg_cooling) -> Figure:
    """Plot cooling potential map and statistics"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Normalize image
    if image_np.max() > 1:
        display_img = image_np.astype(np.uint8)
    else:
        display_img = (image_np * 255).astype(np.uint8)
    
    # Left: Cooling choropleth
    axes[0].imshow(display_img)
    im = axes[0].imshow(cooling_map, cmap='coolwarm', alpha=0.6)
    axes[0].set_title('Urban Cooling Potential Map', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    cbar = plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    cbar.set_label('Temperature Reduction (°C)', fontsize=10)
    
    # Right: Cooling metrics
    # Calculate patch cooling distribution
    patch_cooling_vals = cooling_map.reshape(-1)
    patch_cooling_vals = patch_cooling_vals[patch_cooling_vals > 0]
    
    axes[1].hist(patch_cooling_vals, bins=20, color='skyblue', edgecolor='navy', alpha=0.7)
    axes[1].axvline(x=city_avg_cooling, color='red', linestyle='--', linewidth=2,
                    label=f'City Average: {city_avg_cooling:.2f}°C')
    axes[1].set_xlabel('Potential Cooling (°C)', fontsize=11)
    axes[1].set_ylabel('Frequency (Patches)', fontsize=11)
    axes[1].set_title('Cooling Potential Distribution', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Add interpretation text
    if city_avg_cooling < 0.5:
        interpretation = "Low Cooling Potential"
        color = 'orange'
    elif city_avg_cooling < 1.0:
        interpretation = "Moderate Cooling Potential"
        color = 'blue'
    else:
        interpretation = "High Cooling Potential"
        color = 'green'
    
    fig.suptitle(f'Cooling Assessment: {interpretation}', fontsize=14, fontweight='bold', color=color, y=1.02)
    
    plt.tight_layout()
    return fig

def plot_discrepancy(aerial_gvi, street_gvi, discrepancy, label) -> Figure:
    """Plot dual-view GVI discrepancy visualization"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color based on discrepancy type
    if 'Hidden' in label:
        bar_colors = ['#2E8B57', '#FF6B6B']
        title_color = 'orange'
    elif 'Vertical' in label:
        bar_colors = ['#FF6B6B', '#2E8B57']
        title_color = 'purple'
    else:
        bar_colors = ['#2E8B57', '#2E8B57']
        title_color = 'green'
    
    bars = ax.bar(['Aerial View (Satellite)', 'Street View (Ground)'], 
                  [aerial_gvi, street_gvi], 
                  color=bar_colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, val in zip(bars, [aerial_gvi, street_gvi]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add discrepancy annotation
    ax.text(0.5, 0.95, f'Discrepancy: {discrepancy:+.1f}%', transform=ax.transAxes,
           ha='center', fontsize=14, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_ylabel('Green Visibility Index (%)', fontsize=12)
    ax.set_title(f'Dual-View GVI Comparison: {label}', fontsize=14, fontweight='bold', color=title_color)
    ax.set_ylim(0, max(max(aerial_gvi, street_gvi) + 10, 30))
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig

def plot_hotspots(image_np, flagged_patches, hotspot_scores) -> Figure:
    """Plot hotspot detection overlay"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Normalize image
    if image_np.max() > 1:
        display_img = image_np.astype(np.uint8)
    else:
        display_img = (image_np * 255).astype(np.uint8)
    
    ax.imshow(display_img)
    
    h, w = display_img.shape[:2]
    patch_h = h // 4
    patch_w = w // 4
    
    # Draw grid and highlight hotspots
    for i in range(4):
        for j in range(4):
            # Draw grid lines
            rect = plt.Rectangle((j * patch_w, i * patch_h), patch_w, patch_h,
                                linewidth=1, edgecolor='white', facecolor='none', alpha=0.5)
            ax.add_patch(rect)
    
    # Highlight hotspots
    for fp in flagged_patches:
        i, j = fp['row'], fp['col']
        score = fp['score']
        rect = plt.Rectangle((j * patch_w, i * patch_h), patch_w, patch_h,
                            linewidth=3, edgecolor='red', facecolor='red', alpha=0.3)
        ax.add_patch(rect)
        # Add score label
        ax.text(j * patch_w + patch_w/2, i * patch_h + patch_h/2, f'{score:.1f}',
               ha='center', va='center', fontsize=12, fontweight='bold',
               color='white', bbox=dict(boxstyle='round', facecolor='red', alpha=0.8))
    
    ax.set_title(f'Intervention Hotspots ({len(flagged_patches)} zones flagged)', 
                fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', alpha=0.3, edgecolor='red', label='Hotspot Zone'),
                      Patch(facecolor='none', edgecolor='white', alpha=0.5, label='Grid Cell')]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=10)
    
    plt.tight_layout()
    return fig

def plot_fusion(aerial_gvi, street_gvi, fused_gvi, aerial_weight, street_weight) -> Figure:
    """Plot uncertainty-weighted fusion results"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(['Aerial GVI', 'Street GVI', 'Fused GVI'], 
                  [aerial_gvi, street_gvi, fused_gvi],
                  color=['#2E8B57', '#4682B4', '#9370DB'],
                  edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, val in zip(bars, [aerial_gvi, street_gvi, fused_gvi]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Green Visibility Index (%)', fontsize=12)
    ax.set_title('Uncertainty-Weighted Fusion', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(max(aerial_gvi, street_gvi, fused_gvi) + 10, 30))
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add weight information
    info_text = f'Weights: Aerial = {aerial_weight:.2f}, Street = {street_weight:.2f}\n(Higher street weight = higher uncertainty)'
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    return fig