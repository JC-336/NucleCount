import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io
import math

# Constants for the detection process
BOTTOM_REGION = 0.40
MIN_WIDTH_FRAC = 0.05
DILATE_ITER = 2
SMALL_OBJ_SIZE = 150
MIN_NUCLEUS_AREA = 200
BLUE_THRESHOLD = 80  # Percentage threshold for blue content
SMALL_CIRCLE_RADIUS = 18  # Define what counts as a "small" circle
OVERLAP_THRESHOLD = 0.7  # How much circles need to overlap to be merged

def read_rgb(path):
    try:
        img = io.imread(path)
        if img.dtype != np.uint8:
            img = (img.astype(np.float32) / img.max() * 255).astype(np.uint8)
        return img
    except ValueError as e:
        if 'COMPRESSION.LZW' not in str(e):
            raise
        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f'Cannot read {path}') from e
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def detect_scale_bar(img_rgb):
    h, w = img_rgb.shape[:2]
    y0 = int(h * (1 - BOTTOM_REGION))
    roi = img_rgb[y0:, :]

    gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    _, th = cv2.threshold(gray_roi, 200, 255, cv2.THRESH_BINARY)

    num, lbls, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    mask_bar = np.zeros_like(th, dtype=bool)

    for i in range(1, num):  # skip background
        x, y, w_cc, h_cc, area = stats[i]
        aspect = w_cc / max(h_cc, 1)
        if (aspect >= 5 and w_cc >= MIN_WIDTH_FRAC * w):
            mask_bar[lbls == i] = True

    if np.any(mask_bar):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask_bar = cv2.dilate(mask_bar.astype(np.uint8),
                              kernel, iterations=DILATE_ITER).astype(bool)

    full_mask = np.zeros((h, w), dtype=bool)
    full_mask[y0:, :] = mask_bar
    return full_mask

def merge_overlapping_circles(circles, overlap_threshold=0.5):
    if circles is None or len(circles) == 0:
        return []

    merged_circles = []
    for c in circles:
        is_new_circle = True
        for mc in merged_circles:
            dist = np.linalg.norm(np.array([c[0], c[1]]) - np.array([mc[0], mc[1]]))
            if dist < (c[2] + mc[2]) * overlap_threshold:
                new_radius = max(c[2], mc[2])
                mc[0] = (mc[0] + c[0]) / 2
                mc[1] = (mc[1] + c[1]) / 2
                mc[2] = new_radius
                is_new_circle = False
                break
        if is_new_circle:
            merged_circles.append(c.tolist())
    return merged_circles

def is_border_circle(circle, img_shape):
    """Check if circle touches or crosses image border"""
    x, y, r = circle
    h, w = img_shape[:2]
    
    # Check if circle touches or crosses any image border
    return (x - r <= 0 or  # Left border
            x + r >= w or  # Right border
            y - r <= 0 or  # Top border
            y + r >= h)    # Bottom border

def check_blue_content(img, circle, binary_img):
    """Check if >80% of pixels within circle are blue in the binary image"""
    x, y, r = int(circle[0]), int(circle[1]), int(circle[2])
    h, w = img.shape[:2]
    
    # Create a circle mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (x, y), r, 255, -1)
    
    # Count pixels inside the circle
    total_pixels = np.sum(mask > 0)
    if total_pixels == 0:
        return False
    
    # Count blue pixels (where binary is white/255)
    blue_pixels = np.sum((mask > 0) & (binary_img > 0))
    
    # Calculate percentage
    blue_percentage = (blue_pixels / total_pixels) * 100
    
    return blue_percentage < BLUE_THRESHOLD

def convert_overlapping_circles_to_ovals(circles, img_shape):
    """
    Convert groups of overlapping small circles to ovals
    Returns a mix of circles and ovals with a count of merged shapes
    
    Format for circles: [x, y, radius, type=0]
    Format for ovals: [x, y, width, height, angle, type=1]
    where type 0 = circle, type 1 = oval
    """
    if not circles or len(circles) < 2:
        return circles, 0
    
    # Add type identifier to circles (0 = circle)
    typed_circles = [c + [0] for c in circles]  # [x, y, r, type=0]
    
    # Identify small circles
    small_circles = [c for c in typed_circles if c[2] <= SMALL_CIRCLE_RADIUS]
    large_circles = [c for c in typed_circles if c[2] > SMALL_CIRCLE_RADIUS]
    
    if len(small_circles) < 2:
        return typed_circles, 0  # Not enough small circles to merge
    
    # Find groups of overlapping small circles
    processed = set()
    circle_groups = []
    
    for i in range(len(small_circles)):
        if i in processed:
            continue
            
        c1 = small_circles[i]
        current_group = [i]
        
        for j in range(len(small_circles)):
            if i == j or j in processed:
                continue
                
            c2 = small_circles[j]
            dist = np.linalg.norm(np.array([c1[0], c1[1]]) - np.array([c2[0], c2[1]]))
            
            # Check if they overlap significantly
            if dist < (c1[2] + c2[2]) * OVERLAP_THRESHOLD:
                current_group.append(j)
        
        if len(current_group) > 1:  # Only create groups with multiple circles
            circle_groups.append(current_group)
            processed.update(current_group)
    
    # Process groups to create ovals
    result_shapes = []
    circles_merged = 0
    
    for group in circle_groups:
        group_circles = [small_circles[i] for i in group]
        circles_merged += len(group) - 1  # Count merged circles
        
        if len(group) == 2:  # For pairs, create an oriented oval
            c1, c2 = group_circles[0], group_circles[1]
            
            # Calculate center point between the two circles
            center_x = (c1[0] + c2[0]) / 2
            center_y = (c1[1] + c2[1]) / 2
            
            # Calculate distance between centers
            dx = c2[0] - c1[0]
            dy = c2[1] - c1[1]
            dist = np.sqrt(dx*dx + dy*dy)
            
            # Calculate major axis length (distance between centers + radii)
            major_axis = dist + c1[2] + c2[2]
            
            # Calculate minor axis (average of diameters)
            minor_axis = (c1[2] + c2[2])
            
            # Calculate angle of orientation (in degrees)
            angle = math.degrees(math.atan2(dy, dx))
            
            # Add oval: [x, y, width, height, angle, type=1]
            result_shapes.append([center_x, center_y, major_axis, minor_axis, angle, 1])
            
        else:  # For more than 2 circles, create a best-fit oval
            # Get all centers
            centers = np.array([[c[0], c[1]] for c in group_circles])
            
            # Find the two most distant points to determine major axis
            max_dist = 0
            furthest_pair = (0, 0)
            
            for i in range(len(centers)):
                for j in range(i+1, len(centers)):
                    dist = np.linalg.norm(centers[i] - centers[j])
                    if dist > max_dist:
                        max_dist = dist
                        furthest_pair = (i, j)
            
            # Calculate major axis direction
            i, j = furthest_pair
            dx = centers[j][0] - centers[i][0]
            dy = centers[j][1] - centers[i][1]
            
            # Center of the oval
            center_x = np.mean(centers[:, 0])
            center_y = np.mean(centers[:, 1])
            
            # Major axis = distance between furthest points + their radii
            major_axis = max_dist + group_circles[i][2] + group_circles[j][2]
            
            # Minor axis = average diameter of all circles
            minor_axis = 2 * np.mean([c[2] for c in group_circles])
            
            # Calculate angle
            angle = math.degrees(math.atan2(dy, dx))
            
            # Add oval: [x, y, width, height, angle, type=1]
            result_shapes.append([center_x, center_y, major_axis, minor_axis, angle, 1])
    
    # Add remaining unprocessed small circles
    for i in range(len(small_circles)):
        if i not in processed:
            result_shapes.append(small_circles[i])
    
    # Add large circles
    result_shapes.extend(large_circles)
    
    return result_shapes, circles_merged

input_folder = 'input'
output_folder = 'output'
os.makedirs(output_folder, exist_ok=True)

image_extensions = ('.tiff', '.tif', '.jpeg', '.jpg', '.png')
results = []

for filename in os.listdir(input_folder):
    if not filename.lower().endswith(image_extensions):
        continue

    path = os.path.join(input_folder, filename)
    img_rgb = read_rgb(path)
    original_img = img_rgb.copy()  # Keep a clean copy

    scale_bar_mask = detect_scale_bar(img_rgb)

    gray = img_rgb[:, :, 2]
    blur = cv2.GaussianBlur(gray, (15, 15), 0)

    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = 255 - binary
    binary[scale_bar_mask] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=-10000)

    # Distance Transform and Watershed
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    _, markers = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    markers = np.uint8(markers)
    _, markers = cv2.connectedComponents(markers)
    markers = markers + 1
    markers[binary == 0] = 0
    cv2.watershed(img_rgb, markers)

    # Highlight watershed boundaries (optional visualization)
    img_rgb[markers == -1] = [255, 0, 0]

    circles = cv2.HoughCircles(binary, cv2.HOUGH_GRADIENT, dp=1.7, minDist=15, param1=30, param2=15, minRadius=12, maxRadius=25)
    circle_img = original_img.copy()
    merged_circles = []
    border_circles = []
    non_border_circles = []
    valid_non_border_circles = []  # Circles with >80% blue content
    invalid_non_border_circles = [] # Circles with <80% blue content
    circles_merged = 0

    if circles is not None:
        circles = np.uint16(np.around(circles[0]))
        merged_circles = merge_overlapping_circles(circles, overlap_threshold=0.5)
        
        # Separate border circles from non-border circles
        for c in merged_circles:
            if is_border_circle(c, img_rgb.shape):
                border_circles.append(c)
            else:
                non_border_circles.append(c)
        
        # Check blue content for non-border circles
        for c in non_border_circles:
            if check_blue_content(original_img, c, binary):
                valid_non_border_circles.append(c)
            else:
                invalid_non_border_circles.append(c)
        
        # Convert overlapping small circles to ovals
        mixed_shapes, circles_merged = convert_overlapping_circles_to_ovals(valid_non_border_circles, img_rgb.shape)
        
        # Draw shapes - could be circles or ovals
        for shape in mixed_shapes:
            if shape[-1] == 0:  # Circle
                x, y, r, _ = shape
                cv2.circle(circle_img, (int(x), int(y)), int(r), (0,255,0), 2)
            else:  # Oval
                x, y, width, height, angle, _ = shape
                # Convert to OpenCV ellipse format
                center = (int(x), int(y))
                axes = (int(width/2), int(height/2))
                cv2.ellipse(circle_img, center, axes, angle, 0, 360, (0,255,0), 2)
                
        # Draw border circles in yellow
        for c in border_circles:
            cv2.circle(circle_img, (int(c[0]), int(c[1])), int(c[2]), (255,255,0), 2)

    # Counts
    # Count circles and ovals separately
    circle_count = sum(1 for shape in mixed_shapes if shape[-1] == 0)
    oval_from_circles_count = sum(1 for shape in mixed_shapes if shape[-1] == 1)
    
    # Total valid shapes = circles + ovals (including those created from merged circles)
    valid_shape_count = circle_count + oval_from_circles_count
    
    invalid_circle_count = len(invalid_non_border_circles)  # Count of invalid circles
    border_circle_count = len(border_circles)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_oval_count = 0
    border_oval_count = 0
    
    # Use all circle centers to avoid duplicate detection between circles and ovals
    # Need to extract centers from mixed_shapes
    shape_centers = []
    for shape in mixed_shapes:
        if shape[-1] == 0:  # Circle
            shape_centers.append((int(shape[0]), int(shape[1])))
        else:  # Oval
            shape_centers.append((int(shape[0]), int(shape[1])))
    
    shape_centers.extend([(int(c[0]), int(c[1])) for c in border_circles])

    for cnt in contours:
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            (x, y), (MA, ma), angle = ellipse
            aspect_ratio = max(MA, ma) / min(MA, ma) if min(MA, ma) > 0 else 0
            area = np.pi * (MA/2) * (ma/2)

            if area > MIN_NUCLEUS_AREA and aspect_ratio > 0.2:
                center = (int(x), int(y))
                
                if not any(np.linalg.norm(np.array(center) - np.array(cc)) < 0.6 * max(MA, ma) for cc in shape_centers):
                    # Check if ellipse touches the border
                    touches_border = (x - MA/2 <= 0 or x + MA/2 >= img_rgb.shape[1] or 
                                     y - ma/2 <= 0 or y + ma/2 >= img_rgb.shape[0])
                    
                    if touches_border:
                        cv2.ellipse(circle_img, ellipse, (255,128,0), 2)  # Orange for border ovals
                        border_oval_count += 1
                    else:
                        # For ovals, also check blue content
                        # Create a temporary mask for the ellipse
                        temp_mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
                        cv2.ellipse(temp_mask, ellipse, 255, -1)
                        
                        total_pixels = np.sum(temp_mask > 0)
                        blue_pixels = np.sum((temp_mask > 0) & (binary > 0))
                        blue_percentage = (blue_pixels / total_pixels) * 100 if total_pixels > 0 else 0
                        
                        if blue_percentage > BLUE_THRESHOLD:
                            cv2.ellipse(circle_img, ellipse, (0,0,255), 2)  # Blue for valid ovals
                            detected_oval_count += 1
                        else:
                            cv2.ellipse(circle_img, ellipse, (128,0,128), 2)  # Purple for invalid ovals

    # Calculate total count including ovals created from merged circles
    total_count = circle_count + oval_from_circles_count + detected_oval_count
    total_border_count = border_circle_count + border_oval_count
    
    results.append({
        'filename': filename, 
        'nuclei_count': total_count,
        'border_nuclei_count': total_border_count,
        'circles_merged': circles_merged,
        'ovals_from_circles': oval_from_circles_count,
        'total_valid': total_count + total_border_count
    })

    # Create a legend with information about merged circles -> ovals
    legend_img = np.ones((120, circle_img.shape[1], 3), dtype=np.uint8) * 255
    cv2.putText(legend_img, f"Green: Valid circles ({circle_count}) & ovals from merged circles ({oval_from_circles_count})", 
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    cv2.putText(legend_img, f"Yellow: Border circles ({border_circle_count})", 
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
    cv2.putText(legend_img, f"Blue: Detected ovals ({detected_oval_count})", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    cv2.putText(legend_img, f"Orange: Border ovals ({border_oval_count})", 
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,128,0), 1)
    cv2.putText(legend_img, f"Purple: Invalid ovals", 
                (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128,0,128), 1)
    cv2.putText(legend_img, f"Circles merged into ovals: {circles_merged}", 
                (400, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    
    # Combine with the main visualization
    combined_img = np.vstack([circle_img, legend_img])

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(original_img)
    ax[0].set_title('Original')
    ax[0].axis('off')
    
    ax[1].imshow(binary, cmap='gray')
    ax[1].set_title('Binary')
    ax[1].axis('off')
    
    ax[2].imshow(circle_img)
    ax[2].set_title(f'Valid: {total_count+circles_merged} (Interior), {total_border_count} (Border)')
    ax[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"nuclei_circles_ovals_{filename}.png"), dpi=150)
    plt.close(fig)
    

pd.DataFrame(results).to_csv(os.path.join(output_folder, 'nuclei_counts_circles_ovals.csv'), index=False)

print(pd.DataFrame(results))
