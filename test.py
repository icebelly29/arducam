import cv2
import numpy as np

# Load your hard-earned data
with np.load('calibration_data.npz') as data:
    mtx = data['mtx']
    dist = data['dist']

# Capture one fresh image of the board
import subprocess
subprocess.run(["rpicam-still", "--width", "1456", "--height", "1088", "-o", "verify.jpg"])

img = cv2.imread('verify.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

if ret:
    # Calculate the distance to the board (solvePnP)
    objp = np.zeros((54, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * 22.0 # 22mm squares
    
    _, rvec, tvec = cv2.solvePnP(objp, corners, mtx, dist)
    distance = np.linalg.norm(tvec)
    
    print(f"Distance to board: {distance/10:.2f} cm")
