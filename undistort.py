import cv2
import numpy as np

# 1. Load your hard-earned calibration data
with np.load('calibration_data.npz') as data:
    mtx = data['mtx']
    dist = data['dist']

# 2. Load the distorted image
img = cv2.imread('right_frame.jpg') # Replace with your target image
h, w = img.shape[:2]

# 3. Refine the camera matrix (This handles the 'black edges' issue)
# alpha=0: All pixels in the result are valid (crops out black edges)
# alpha=1: All pixels from the original are retained (shows black edges)
new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))

# 4. Apply the correction
undistorted_img = cv2.undistort(img, mtx, dist, None, new_camera_mtx)

# 5. Crop the image based on the ROI (Region of Interest)
x, y, w_roi, h_roi = roi
undistorted_img = undistorted_img[y:y+h_roi, x:x+w_roi]

# 6. Save the clean result
cv2.imwrite('right_frame_ud.jpg', undistorted_img)
print("Image undistorted and saved as 'undistorted_result.jpg'")
