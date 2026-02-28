import cv2
import numpy as np

# Load your undistorted images (or undistort them first using your mtx/dist)
img1 = cv2.imread('left_frame_ud.jpg')
img2 = cv2.imread('right_frame_ud.jpg')

# Create the stitcher object
# Use mode 1 for Scans/Panoramas
stitcher = cv2.Stitcher_create(mode=cv2.Stitcher_PANORAMA)

# Stitch the images
status, stitched = stitcher.stitch([img1, img2])

if status == cv2.Stitcher_OK:
    cv2.imwrite('stitched_result_ud.jpg', stitched)
    print("Stitching successful!")
else:
    print(f"Stitching failed with status code {status}")
