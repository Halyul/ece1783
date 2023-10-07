import math
import numpy as np

"""
    Get the padding of the frame.

    Parameters:
        width (int): The width of the frame.
        height (int): The height of the frame.
        n (int): The length of the block.
    
    Returns:
        (width, height) (tuple): The padded width and height of the frame.
"""
def get_padding(width: int, height: int, n: int) -> tuple:
    pad_width = math.ceil(width / n) * n if width > 0 else -1
    pad_height = math.ceil(height / n) * n if height > 0 else -1
    return pad_width, pad_height

"""
    Convert YUV to RGB.

    Parameters:
        y (np.ndarray): The Y channel.
        u (np.ndarray): The U channel.
        v (np.ndarray): The V channel.
    
    Returns:
        (r, g, b, rgb) (tuple): The R, G, B channels and the RGB array.
"""
def yuv2rgb(y: np.ndarray, u: np.ndarray = None, v: np.ndarray = None) -> np.ndarray:
    height, width = y.shape
    u = np.array([128] * (height * width)).reshape(height, width) if u is None else u
    v = np.array([128] * (height * width)).reshape(height, width) if v is None else v
    r = 1.164 * (y - 16) + 1.596 * (v - 128)
    g = 1.164 * (y - 16) - 0.813 * (v - 128) - 0.392 * (u - 128)
    b = 1.164 * (y - 16) + 2.017 * (u - 128)
    r = np.clip(r, 0, 255).astype(np.uint8)
    g = np.clip(g, 0, 255).astype(np.uint8)
    b = np.clip(b, 0, 255).astype(np.uint8)
    return (r, g, b, np.stack((r, g, b), axis=-1))
    