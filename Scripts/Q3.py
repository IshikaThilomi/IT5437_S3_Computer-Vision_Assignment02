import cv2 as cv
import numpy as np
import os

# --- 1. Path Configuration (Matches your VS Code sidebar) ---
script_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate: Scripts -> Assignment2 -> Images -> Results
results_dir = os.path.normpath(os.path.join(script_dir, "..", "Images", "Results"))
img1_path = os.path.normpath(os.path.join(script_dir, "..", "Images", "c1.jpg"))
img2_path = os.path.normpath(os.path.join(script_dir, "..", "Images", "c2.jpg"))

# Verify directory exists
os.makedirs(results_dir, exist_ok=True)

im1 = cv.imread(img1_path)
im2 = cv.imread(img2_path)

if im1 is None or im2 is None:
    print(f"Error: Could not find c1.jpg or c2.jpg in {os.path.join(script_dir, '..', 'Images')}")
    exit()

# --- 2. Manual Selection (Part a & b) ---
N = 6
all_pts = []

def click_event(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        cv.circle(param[1], (x, y), 5, (0, 0, 255), -1)
        param[0].append((x, y))
        cv.imshow("Point Selection", param[1])

for i, img in enumerate([im1, im2]):
    curr_pts = []
    temp_view = img.copy()
    cv.imshow("Point Selection", temp_view)
    cv.setMouseCallback("Point Selection", click_event, [curr_pts, temp_view])
    print(f"Select {N} points on Image {i+1}...")
    while len(curr_pts) < N:
        if cv.waitKey(1) & 0xFF == 27: break
    all_pts.append(np.float32(curr_pts))

cv.destroyAllWindows()

# Process and Save Manual Results
H_m, _ = cv.findHomography(all_pts[1], all_pts[0])
warp_m = cv.warpPerspective(im2, H_m, (im1.shape[1], im1.shape[0]))
diff_m = cv.absdiff(im1, warp_m)

cv.imwrite(os.path.join(results_dir, "q3_warped_manual.png"), warp_m)
cv.imwrite(os.path.join(results_dir, "q3_diff_manual.png"), diff_m)

# --- 3. Automated SIFT (Part c & d) ---
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(im1, None)
kp2, des2 = sift.detectAndCompute(im2, None)
bf = cv.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
good = [m for m, n in matches if m.distance < 0.75 * n.distance]

if len(good) > 4:
    src = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    H_a, _ = cv.findHomography(src, dst, cv.RANSAC, 5.0)
    warp_a = cv.warpPerspective(im2, H_a, (im1.shape[1], im1.shape[0]))
    diff_a = cv.absdiff(im1, warp_a)
    
    cv.imwrite(os.path.join(results_dir, "q3_warped_auto.png"), warp_a)
    cv.imwrite(os.path.join(results_dir, "q3_diff_auto.png"), diff_a)
    print(f"Successfully saved images to {results_dir}")