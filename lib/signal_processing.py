import math
import numpy as np
from scipy import signal

"""
    fspecial('gaussian', size, sigma) in MATLAB

    Parameters:
        size (int): The size of the filter
        sigma (float): The standard deviation of the filter

    Returns:
        window (np.ndarray): The Gaussian filter
"""
def __fspecial_gauss(size: int, sigma: float) -> np.ndarray:
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()

"""
    Calculate PSNR. 
    https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Parameters:
        y_padded (np.ndarray): The original image
        y_averaged (np.ndarray): The averaged image
    
    Returns:
        psnr (float): The PSNR value
"""
def psnr(y_padded: np.ndarray, y_averaged: np.ndarray) -> float:
    y_padded_float64 = y_padded.astype(np.float64)
    y_averaged_float64 = y_averaged.astype(np.float64)
    mse = np.mean((y_padded_float64 - y_averaged_float64) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

"""
    Calculate SSIM using the following MATLAB code:
    https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m

    The following code is a port of the MATLAB code to Python

    Parameters:
        y_padded (np.ndarray): The original image
        y_averaged (np.ndarray): The averaged image

    Returns:
        ssim_map (np.ndarray): The SSIM map value
"""
def ssim(y_padded: np.ndarray, y_averaged: np.ndarray) -> np.ndarray:
    y_padded_float64 = y_padded.astype(np.float64)
    y_averaged_float64 = y_averaged.astype(np.float64)

    size = 11
    sigma = 1.5
    K = [0.01, 0.03]
    L = 255
    C1 = (K[0] * L) ** 2
    C2 = (K[1] * L) ** 2
    window = __fspecial_gauss(size, sigma)
    window = window/np.sum(np.sum(window))
    mu1 = signal.convolve2d(window, y_padded_float64, mode='valid')
    mu2 = signal.convolve2d(window, y_averaged_float64, mode='valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = signal.convolve2d(window, y_padded_float64 * y_padded_float64, mode='valid') - mu1_sq
    sigma2_sq = signal.convolve2d(window, y_averaged_float64 * y_averaged_float64, mode='valid') - mu2_sq
    sigma12 = signal.convolve2d(window, y_padded_float64 * y_averaged_float64, mode='valid') - mu1_mu2
    return ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import pathlib

    with open(pathlib.Path.cwd().joinpath("y_only", "0.y-only-padded"), 'rb') as f:
        file_bytes = f.read()
    file_bytes = np.frombuffer(file_bytes, dtype=np.uint8).reshape(288, 352)
    with open(pathlib.Path.cwd().joinpath("y_only", "0.y-only-averaged"), 'rb') as f:
        x = f.read()
    x = np.frombuffer(x, dtype=np.uint8).reshape(288, 352)
    ssim_map = ssim(file_bytes, x)
    print(ssim_map.mean())
    print(psnr(file_bytes, x))
    plt.imshow(ssim_map, interpolation='nearest', cmap='gray')
    plt.show()