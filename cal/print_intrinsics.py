#!/usr/bin/env python3
import numpy as np
import argparse
import os
import math

"""
One time scrit to print camera intrinsics from a saved calibration file.
"""

def main():
    ap = argparse.ArgumentParser(description="Print K (camera matrix) from a saved OpenCV calibration .npz")
    ap.add_argument("--file", "-f", type=str, default="camera_calib.npz",
                    help="Path to .npz file saved by the calibration script")
    args = ap.parse_args()

    if not os.path.isfile(args.file):
        raise FileNotFoundError(f"Could not find: {args.file}")

    data = np.load(args.file, allow_pickle=True)
    # Expected keys from the calibration script
    K      = data.get("K", None)
    dist   = data.get("dist", None)
    imsize = data.get("image_size", None)

    if K is None:
        raise KeyError("No 'K' found in file. Are you sure this is the calibration npz?")
    if imsize is None:
        print("WARNING: 'image_size' missing; FOV computation will be skipped.")

    print("\n=== Camera Intrinsics ===")
    print("K (camera matrix):")
    print(K)

    # Unpack common parameters
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    skew = float(K[0, 1])

    print(f"\nfx (pixels): {fx:.6f}")
    print(f"fy (pixels): {fy:.6f}")
    print(f"cx (px):     {cx:.6f}")
    print(f"cy (px):     {cy:.6f}")
    print(f"skew:        {skew:.6f} (should be ~0)")

    if dist is not None:
        print("\nDistortion coefficients (OpenCV order):")
        print(dist.ravel())

    if imsize is not None:
        w, h = int(imsize[0]), int(imsize[1])
        fov_x = 2.0 * math.degrees(math.atan2(w, 2.0 * fx))
        fov_y = 2.0 * math.degrees(math.atan2(h, 2.0 * fy))
        print(f"\nImage size:  {w} x {h} px")
        print(f"FOVx / FOVy: {fov_x:.2f}° / {fov_y:.2f}°")

    # Quick copy-paste line for your next script
    print("\nCopy-paste for your code:")
    print(f"K = np.array([{K[0].tolist()}, {K[1].tolist()}, {K[2].tolist()}], dtype=np.float64)")

if __name__ == "__main__":
    main()
