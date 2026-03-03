import cv2
import numpy as np

# 1. Load your calibration data
with np.load('calibration_data.npz') as data:
    mtx = data['mtx']
    dist = data['dist']

def undistort_frame(img, mtx, dist):
    h, w = img.shape[:2]
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
    undistorted = cv2.undistort(img, mtx, dist, None, new_mtx)
    x, y, w_roi, h_roi = roi
    return undistorted[y:y+h_roi, x:x+w_roi]

# 2. Load and Undistort your two frames
# (Take two photos with about 30% overlap)
img_left = cv2.imread('left_frame.jpg')
img_right = cv2.imread('right_frame.jpg')

img1 = undistort_frame(img_left, mtx, dist)
img2 = undistort_frame(img_right, mtx, dist)

# 3. Initialize SIFT detector
sift = cv2.SIFT_create()

# Find keypoints and descriptors
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# 4. Match features using FLANN (Fast Library for Approximate Nearest Neighbors)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# 5. Apply Lowe's Ratio Test (filters out 'BS' matches)
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# 6. Visualize the matches
match_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, 
                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imwrite('feature_matches.jpg', match_img)
print(f"Found {len(good_matches)} good matches. Check 'feature_matches.jpg'")