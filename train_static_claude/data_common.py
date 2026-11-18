"""Crop helpers shared by both datasets (exact continuous-pixel bookkeeping)."""
import numpy as np


def crop_square(image, k, x0, y0, side, size):
    """Crop the square [x0, x0+side)^2 (continuous pixels, may exceed the image),
    resize to ``size`` and return (crop uint8, valid float mask, K_crop).

    Downscaling first uses INTER_AREA on the whole image (anti-aliasing); the
    remaining affine is then close to a pure translation.
    """
    import cv2

    height, width = image.shape[:2]
    scale = size / side
    if scale < 0.95:
        new_w, new_h = max(1, round(width * scale)), max(1, round(height * scale))
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        sx, sy = new_w / width, new_h / height
    else:
        new_w, new_h, sx, sy = width, height, 1., 1.
    # continuous mapping from (possibly resized) source to crop
    ax, ay = scale / sx, scale / sy
    bx, by = -x0 * scale, -y0 * scale
    m = np.array([[ax, 0, bx + 0.5 * ax - 0.5], [0, ay, by + 0.5 * ay - 0.5]], np.float64)
    crop = cv2.warpAffine(image, m, (size, size), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = cv2.warpAffine(np.ones((new_h, new_w), np.uint8), m, (size, size),
                           flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                           borderValue=0)
    a = np.array([[scale, 0, -x0 * scale], [0, scale, -y0 * scale], [0, 0, 1.]])
    return crop, valid.astype(np.float32), (a @ k).astype(np.float32)
