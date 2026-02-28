import numpy as np
import cv2
import glob

# --- 1. HARDWARE CONFIGURATION ---
CHECKERBOARD = (9, 6)
SQUARE_SIZE = 22.0  # Actual size of one black square in mm
W, H = 1456, 1088   # Raspberry Pi Global Shutter resolution

# --- 2. THE LENS PHYSICS (12mm + Spacer) ---
# Theoretical f = 12mm / 0.00345mm pixel pitch = 3478.26
F_PIXEL_TRUE = 3478.26 

# Termination criteria for sub-pixel accuracy
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare 3D object points (0,0,0), (22,0,0), (44,0,0) ...
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = [] # 3D points in real world space
imgpoints = [] # 2D points in image plane

# --- 3. LOAD AND PROCESS IMAGES ---
images = glob.glob('calib*.jpg')
if not images:
    print("Error: No 'calib*.jpg' files found.")
    exit()

valid_count = 0
for fname in images:
    img = cv2.imread(fname)
    if img is None: continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        objpoints.append(objp)
        # Refine corner locations for sub-pixel precision
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        valid_count += 1
        print(f"[{valid_count}] Found corners in: {fname}")

if valid_count == 0:
    print("Error: Could not find corners in any images. Check lighting/focus.")
    exit()

print(f"\nProcessing {valid_count} images for 12mm Lens + 2.5mm Spacer...")

# --- 4. THE STABILIZED MATH ---

# Manually define the center of the 1456x1088 sensor
cx_guess = W / 2 # 728.0
cy_guess = H / 2 # 544.0

# Initial Intrinsic Matrix based on hardware reality
initial_mtx = np.array([
    [F_PIXEL_TRUE, 0, cx_guess],
    [0, F_PIXEL_TRUE, cy_guess],
    [0, 0, 1]
], dtype=np.float32)

# FLAGS: 
# - USE_INTRINSIC_GUESS: Starts at 3478 instead of guessing randomly.
# - FIX_PRINCIPAL_POINT: Locks the lens center to the middle (728, 544).
# - FIX_ASPECT_RATIO: Forces square pixels (fx = fy).
# - ZERO_TANGENT_DIST: Assumes lens is perfectly parallel to sensor (p1=0, p2=0).
# - FIX_K2 & FIX_K3: Forces higher-order distortion to 0.0 to prevent 'black holes'.
flags = (cv2.CALIB_USE_INTRINSIC_GUESS | 
         cv2.CALIB_FIX_PRINCIPAL_POINT | 
         cv2.CALIB_FIX_ASPECT_RATIO | 
         cv2.CALIB_ZERO_TANGENT_DIST |
         cv2.CALIB_FIX_K2 | 
         cv2.CALIB_FIX_K3)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, (W, H), initial_mtx, None, flags=flags
)

# --- 5. RE-PROJECTION ERROR CHECK ---
print("\nPer-Image Projection Error:")
total_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    print(f"{images[i]}: {error:.4f}")
    total_error += error

# --- 6. FINAL RESULTS ---
print(f"\n" + "="*40)
print(f"FINAL RMS ERROR: {ret:.4f}")
print(f"Mean Re-projection Error: {total_error/len(objpoints):.4f}")
print("-" * 40)
print(f"Effective Focal Length: {mtx[0,0]:.2f} pixels")
print(f"Distortion (k1): {dist.ravel()[0]:.4f}")
print("="*40)

# Save the hardware-verified data
np.savez("calibration_data.npz", mtx=mtx, dist=dist)
print("\nSUCCESS: Calibration data saved to 'calibration_data.npz'")
