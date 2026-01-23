"""
Helper script to run OpenFace via Docker (optimized)
"""

import subprocess
import os
import tempfile
import hashlib
from functools import lru_cache


# Cache OpenFace results for identical frames
@lru_cache(maxsize=32)
def _cached_openface_run(image_hash, image_path, output_dir):
    """Cached OpenFace execution"""
    abs_image_path = os.path.abspath(image_path)
    abs_output_dir = os.path.abspath(output_dir)

    cmd = [
        "docker", "run", "--rm",
        "-v", f"{abs_image_path}:/input.jpg:ro",
        "-v", f"{abs_output_dir}:/output",
        "algebr/openface:latest",
        "/home/openface-build/build/bin/FeatureExtraction",
        "-f", "/input.jpg",
        "-out_dir", "/output",
        "-quiet"
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=5,
        check=False
    )

    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "output_dir": output_dir,
        "returncode": result.returncode
    }


def run_openface_docker(image_path, output_dir=None):
    """
    Run OpenFace Docker container on an image file (optimized with caching)

    Args:
        image_path: Path to input image
        output_dir: Directory for output CSV (defaults to temp)

    Returns:
        Dictionary with parsed facial features
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp()

    # Create hash of image for caching
    try:
        with open(image_path, 'rb') as f:
            image_hash = hashlib.md5(f.read()).hexdigest()
    except Exception:
        image_hash = None

    if image_hash:
        return _cached_openface_run(image_hash, image_path, output_dir)

    # Fallback without caching
    abs_image_path = os.path.abspath(image_path)
    abs_output_dir = os.path.abspath(output_dir)

    cmd = [
        "docker", "run", "--rm",
        "-v", f"{abs_image_path}:/input.jpg:ro",
        "-v", f"{abs_output_dir}:/output",
        "algebr/openface:latest",
        "/home/openface-build/build/bin/FeatureExtraction",
        "-f", "/input.jpg",
        "-out_dir", "/output",
        "-quiet"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=False)

    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "output_dir": output_dir,
        "returncode": result.returncode
    }


def parse_openface_csv(csv_path):
    """Parse OpenFace CSV output"""
    import csv

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        row = next(reader)

    action_units = {k: float(v) for k, v in row.items() if k.startswith('AU')}

    return {
        "action_units": action_units,
        "gaze_angle_x": float(row.get('gaze_angle_x', 0)),
        "gaze_angle_y": float(row.get('gaze_angle_y', 0)),
        "head_pose_x": float(row.get('pose_Tx', 0)),
        "head_pose_y": float(row.get('pose_Ty', 0)),
        "head_pose_z": float(row.get('pose_Tz', 0)),
        "confidence": float(row.get('confidence', 0))
    }


if __name__ == "__main__":
    print("OpenFace Docker helper loaded")
    print("Use: run_openface_docker(image_path)")
