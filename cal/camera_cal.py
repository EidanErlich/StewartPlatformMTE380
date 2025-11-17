#!/usr/bin/env python3
import cv2
import numpy as np
import time
import argparse
from collections import deque

"""
One time cal to get camera matrix and distortion coeffs
"""

def main():
    ap = argparse.ArgumentParser(description="Live camera intrinsics calibration with a checkerboard.")
    ap.add_argument("--cam", type=int, default=0, help="Camera index (default: 0)")
    ap.add_argument("--width", type=int, default=0, help="Force capture width (0 = leave default)")
    ap.add_argument("--height", type=int, default=0, help="Force capture height (0 = leave default)")
    # Your board: 11x8 squares -> 10x7 inner corners
    ap.add_argument("--corners_x", type=int, default=10, help="Number of inner corners along X (columns)")
    ap.add_argument("--corners_y", type=int, default=7,  help="Number of inner corners along Y (rows)")
    ap.add_argument("--square_mm", type=float, default=25.0, help="Checker square size in millimeters")
    ap.add_argument("--save", type=str, default="camera_calib.npz", help="Output file (.npz)")
    args = ap.parse_args()

    CHECKERBOARD = (args.corners_x, args.corners_y)
    square_size = args.square_mm  # mm

    # Prepare a single "object points" template for this board
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= square_size  # in mm (units don't matter as long as consistent)

    # Storage for calibration sets
    objpoints = []  # 3D points in world coords
    imgpoints = []  # 2D points in image plane

    cap = cv2.VideoCapture(args.cam)
    if args.width  > 0: cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    if args.height > 0: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print("ERROR: Cannot open camera", args.cam)
        return

    print("\nControls:")
    print("  SPACE  = capture current view")
    print("  c      = calibrate with captured views")
    print("  u      = toggle undistortion preview")
    print("  r      = reset captured views")
    print("  s      = save calibration to file (after calibrate)")
    print("  q/ESC  = quit")
    print("\nMake sure the checkerboard is flat and captures cover the WHOLE image (edges & corners).")

    # Visualization helpers
    font = cv2.FONT_HERSHEY_SIMPLEX
    undistort_preview = False
    has_calib = False
    K = None
    dist = None
    newK = None
    roi = None
    per_view_errors = []
    last_status = deque(maxlen=2)

    criteria_subpix = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
    find_flags = (cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE |
                  cv2.CALIB_CB_FAST_CHECK)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Stream ended / camera read failed.")
            break

        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Try to find the checkerboard every frame for visual feedback
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags=find_flags)
        if ret:
            # Refine to subpixel
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria_subpix)
            cv2.drawChessboardCorners(display, CHECKERBOARD, corners, ret)
            status_text = f"Checkerboard detected: {CHECKERBOARD[0]}x{CHECKERBOARD[1]}"
            last_status.append(status_text)
        else:
            last_status.append("No checkerboard")

        # If we have calibration, optionally show undistorted preview
        if undistort_preview and has_calib:
            und = cv2.undistort(frame, K, dist, None, newK)
            cv2.putText(und, "UNDISTORTED PREVIEW (u to toggle)", (18, 28), font, 0.6, (255, 255, 255), 2)
            cv2.imshow("undistorted", und)
        else:
            # Close the window if it was open
            try:
                cv2.destroyWindow("undistorted")
            except:
                pass

        # HUD overlay
        cv2.putText(display, f"Captures: {len(objpoints)}", (15, 25), font, 0.7, (0, 200, 255), 2)
        if has_calib:
            cv2.putText(display, "Calibrated", (15, 55), font, 0.7, (0, 255, 0), 2)
        if last_status:
            cv2.putText(display, last_status[-1], (15, 85), font, 0.6, (255, 255, 255), 2)
        cv2.imshow("calibrate_live", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):  # ESC or q
            break

        elif key == ord(' '):  # capture
            if ret:
                objpoints.append(objp.copy())
                imgpoints.append(corners.copy())
                print(f"Captured view #{len(objpoints)}")
            else:
                print("Capture skipped: checkerboard not detected.")

        elif key == ord('r'):  # reset captures
            objpoints.clear()
            imgpoints.clear()
            has_calib = False
            per_view_errors = []
            print("Reset all captured views.")

        elif key == ord('u'):  # toggle undistort preview
            undistort_preview = not undistort_preview

        elif key == ord('c'):  # calibrate
            if len(objpoints) < 8:
                print("Need at least ~8 good views from varied poses before calibrating.")
                continue

            img_size = (gray.shape[1], gray.shape[0])
            print("Calibrating…")
            start = time.time()

            # Initial calibration
            ret_calib, K, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, img_size, None, None,
                flags=cv2.CALIB_RATIONAL_MODEL
            )

            # Compute per-view reprojection error
            per_view_errors = []
            for i in range(len(objpoints)):
                proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
                err = cv2.norm(imgpoints[i], proj, cv2.NORM_L2) / len(proj)
                per_view_errors.append(float(err))

            mean_err = float(np.mean(per_view_errors))
            dur = time.time() - start
            print(f"RMS (OpenCV): {ret_calib:.4f} pixels")
            print(f"Mean per-view reprojection error: {mean_err:.4f} px (n={len(per_view_errors)})")
            print(f"Camera matrix K:\n{K}")
            print(f"Distortion coeffs (k1,k2,p1,p2,k3,k4,k5,k6):\n{dist.ravel()}")
            print(f"Took {dur:.2f} s")
            has_calib = True

            # Build a nicer new camera matrix for undistortion preview
            newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, img_size, alpha=0.0)

        elif key == ord('s'):  # save
            if not has_calib:
                print("Calibrate first (press 'c').")
                continue
            np.savez(
                args.save,
                K=K, dist=dist, image_size=(gray.shape[1], gray.shape[0]),
                corners_x=CHECKERBOARD[0], corners_y=CHECKERBOARD[1],
                square_mm=square_size,
                per_view_errors=np.array(per_view_errors, dtype=np.float32)
            )
            print(f"Saved calibration to {args.save}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
