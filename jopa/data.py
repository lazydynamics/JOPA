"""Data utilities: MNIST loading and rotating-digit dataset generation."""
from __future__ import annotations
import gzip
import os
import struct
import urllib.request

import numpy as np

_MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
_CACHE = os.path.expanduser("~/.cache/jopa")


def load_mnist():
    """Download (if needed) and return MNIST training set.

    Returns
    -------
    images : (N, 28, 28) float32 in [0, 1]
    labels : (N,) int
    """
    os.makedirs(_CACHE, exist_ok=True)

    def _get(name):
        path = os.path.join(_CACHE, name)
        if not os.path.exists(path):
            print(f"  downloading {name} …")
            urllib.request.urlretrieve(_MNIST_URL + name, path)
        return path

    with gzip.open(_get("train-images-idx3-ubyte.gz"), "rb") as f:
        _, n, h, w = struct.unpack(">4I", f.read(16))
        imgs = np.frombuffer(f.read(), np.uint8).reshape(n, h, w)
    with gzip.open(_get("train-labels-idx1-ubyte.gz"), "rb") as f:
        struct.unpack(">2I", f.read(8))
        labs = np.frombuffer(f.read(), np.uint8)
    return imgs.astype(np.float32) / 255.0, labs.astype(int)


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------

def rotate_image(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate a 2-D image (nearest-neighbour, zero-padded)."""
    h, w = image.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)

    ys, xs = np.mgrid[0:h, 0:w]
    xc, yc = xs - cx, ys - cy
    # inverse rotation → source coords
    x_src = np.round(cos_a * xc + sin_a * yc + cx).astype(int)
    y_src = np.round(-sin_a * xc + cos_a * yc + cy).astype(int)

    valid = (x_src >= 0) & (x_src < w) & (y_src >= 0) & (y_src < h)
    x_src = np.clip(x_src, 0, w - 1)
    y_src = np.clip(y_src, 0, h - 1)
    return image[y_src, x_src] * valid


def rotating_mnist(
    n_digits: int = 10,
    n_rotations: int = 36,
    digits: tuple[int, ...] = (0, 1),
    binarize: bool = True,
    threshold: float = 0.5,
):
    """Create a dataset of rotated MNIST digits.

    Returns
    -------
    images : (N, 28, 28) float32
    labels : (N,) int
    """
    all_imgs, all_labs = load_mnist()

    images, labels = [], []
    for digit in digits:
        idxs = np.where(all_labs == digit)[0][:n_digits]
        for idx in idxs:
            img = all_imgs[idx]
            for r in range(n_rotations):
                angle = r * 360.0 / n_rotations
                rot = rotate_image(img, angle)
                images.append(rot)
                labels.append(digit)

    images = np.stack(images).astype(np.float32)
    labels = np.array(labels, dtype=int)

    if binarize:
        images = (images > threshold * images.max()).astype(np.float32)

    return images, labels


def make_controlled_sequence(
    digit_idx: int = 0,
    n_frames: int = 100,
    binarize: bool = True,
    threshold: float = 0.5,
    seed: int = 0,
):
    """Generate a rotation sequence driven entirely by control actions.

    Each action u[t] is an angular velocity in degrees per step. The digit
    angle is the cumulative sum of u[t]; there is no autonomous rotation.

    Returns
    -------
    frames : list of (28, 28) float32 arrays
    actions : list of (1,) float32 arrays — angular velocities (deg/step)
    angles : (n_frames,) float64 array of cumulative angles in degrees
    """
    rng = np.random.RandomState(seed)
    imgs, _ = load_mnist()
    img = imgs[digit_idx]

    n_trans = n_frames - 1

    # Piecewise constant velocities (deg/step) with distinct regimes
    segment_len = n_trans // 5
    raw_speeds = [5.0, -3.0, 8.0, 0.0, -5.0]
    velocities = np.zeros(n_trans)
    for i, speed in enumerate(raw_speeds):
        start = i * segment_len
        end = min(start + segment_len, n_trans)
        velocities[start:end] = speed + rng.randn(end - start) * 0.3

    angle_deg = np.zeros(n_frames)
    for i in range(1, n_frames):
        angle_deg[i] = angle_deg[i - 1] + velocities[i - 1]

    frames = []
    for angle in angle_deg:
        rot = rotate_image(img, angle)
        if binarize:
            rot = (rot > threshold * rot.max()).astype(np.float32) if rot.max() > 0 else rot
        frames.append(rot.astype(np.float32))

    actions = [np.array([v], dtype=np.float32) for v in velocities]

    return frames, actions, angle_deg
