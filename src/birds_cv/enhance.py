"""Image enhancement for improved detection quality."""

import cv2


def enhance_frame(frame, clip_limit=2.0, tile_size=(8, 8)):
    """
    Apply image enhancement to improve detection quality.

    Uses CLAHE (Contrast Limited Adaptive Histogram Equalization) for contrast
    enhancement and unsharp masking for sharpening.

    Args:
        frame: Input BGR frame
        clip_limit: CLAHE clip limit (higher = more contrast)
        tile_size: CLAHE grid size

    Returns:
        Enhanced BGR frame
    """
    # Convert to LAB color space for better contrast manipulation
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to the L channel (luminance)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    cl = clahe.apply(l)

    # Merge channels and convert back to BGR
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Apply sharpening using unsharp masking
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 3)
    sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)

    return sharpened
