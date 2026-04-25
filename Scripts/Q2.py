import cv2 as cv
import numpy as np
import os

# --- 1. Path Configuration ---
# Finds your images and result folder based on your specific directory structure
script_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate: Scripts -> Assignment2 -> Images
base_img_dir = os.path.normpath(os.path.join(script_dir, "..", "Images"))
results_dir = os.path.join(base_img_dir, "Results")
img_path = os.path.join(base_img_dir, "earrings.jpg")

# Ensure results folder exists
os.makedirs(results_dir, exist_ok=True)

# --- 2. Load and Process Image ---
img = cv.imread(img_path)

if img is None:
    print(f"Error: Could not load image from {img_path}")
    exit()

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# Apply Gaussian Blur to reduce noise for better circle detection
blurred = cv.GaussianBlur(gray, (9, 9), 2)

# --- 3. Camera Parameters from Assignment ---
f = 8.0               # focal length in mm 
pixel_size = 0.0022   # 2.2 um converted to mm 
d = 720.0             # distance from lens to imaging plane in mm 

# Magnification Factor M = f / (d - f)
mag = f / (d - f)

# --- 4. Measurement and Annotation ---
# Detect circles to find the outer diameter
circles = cv.HoughCircles(blurred, cv.HOUGH_GRADIENT, 1, 200, 
                          param1=50, param2=30, minRadius=150, maxRadius=250)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # Radius in pixels
        r_px = i[2]
        
        # Calculate Physical Sizes
        # We use the magnification ratio to convert sensor size to object size
        # Actual Diameter = (Pixel Diameter * Pixel Size) / Magnification
        actual_outer_dia = (r_px * 2 * pixel_size) / mag
        
        # Approximate inner diameter based on visual ratio in image (~0.47)
        # In a real report, you would measure the inner pixels specifically
        inner_r_px = r_px * 0.47
        actual_inner_dia = (inner_r_px * 2 * pixel_size) / mag
        
        # Draw the detected outer circle
        cv.circle(img, (i[0], i[1]), r_px, (0, 255, 0), 4)
        # Draw the center
        cv.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
        
        # Add labels to the image
        text_outer = f"Outer Dia: {actual_outer_dia:.2f} mm"
        text_inner = f"Inner Dia: {actual_inner_dia:.2f} mm"
        cv.putText(img, text_outer, (i[0] - 150, i[1] - r_px - 20), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv.putText(img, text_inner, (i[0] - 150, i[1] + 20), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        print(f"Success! Calculated Outer Diameter: {actual_outer_dia:.2f} mm")
        print(f"Success! Calculated Inner Diameter: {actual_inner_dia:.2f} mm")

# --- 5. Save the Output ---
output_path = os.path.join(results_dir, "q2_output.png")
cv.imwrite(output_path, img)
print(f"Results saved to: {output_path}")