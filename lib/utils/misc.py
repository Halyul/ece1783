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

"""
    Transform the i x i blocked-based frame into a pixel-based frame.
    Y component only.

    Parameters:
        np_array (np.ndarray): The block-based frame.
        shape (height, width) (tuple): The shape of the pixel-based frame.
        params_i (int): The block size.
    
    Returns:
        np_pixel_array (np.ndarray): The pixel-based frame.
"""
def pixel_create(np_array: np.ndarray, shape: tuple, params_i: int) -> np.ndarray:
    offset = shape[1] // params_i

    d = np_array.reshape(-1, offset, params_i, params_i)
    e = []
    for i in range(offset):
        e.append(d[:, i].reshape(-1, params_i))
    
    np_pixel_array = np.block(e)
    if not shape == np_pixel_array.shape:
        raise Exception('Shape mismatch.')
    return np_pixel_array

"""
    Transform the pixel-based frame into an i x i sized block-based frame.
    Y component only.

    Parameters:
        np_array (np.ndarray): The pixel-based frame.
        params_i (int): The block size.

    Returns:
        (np_block_array, offset, block_size, np_array_padded) (tuple): The block-based frame, the offset, the block size, and the padded pixel-based frame.
"""
def block_create(np_array: np.ndarray, params_i: int) -> tuple:
    width = np_array.shape[1]
    height = np_array.shape[0]
    paded_width, paded_height = get_padding(width, height, params_i)
    pad_width = paded_width - width
    pad_height = paded_height - height
    np_array_padded = np.pad(np_array, ((0, pad_height), (0, pad_width)), 'constant', constant_values=128)

    offset = np_array_padded.shape[1] // params_i
    block_size = params_i ** 2
    # combine i rows into 1 row
    # group into i x 1, with <x> channels
    a = np_array_padded.reshape(params_i, -1, params_i, 1) # (i, -1, i, <x>)
    b = []
    # for loop width // i
    # select every i-th column
    # group into one array, size = i**2 * 3
    # = one block in a row, raster order
    for i in range(offset):
        b.append(a[:, i::offset].reshape(-1, block_size * 1)) # [all rows, start::step = width // i] (-1, i**2 * <x>))

    # combine into 1 array
    # group into i x i, with <x> channels
    # each row has width // i blocks
    np_block_array = np.block(b).reshape(-1, offset, block_size) # (-1,  width // i, i**2)
    return (np_block_array, offset, block_size, np_array_padded)

def rounding(x, base):
    return base * round(x / base)

def convert_within_range(np_array, dtype=np.uint8):
    return np.clip(np_array, 0, 255).astype(dtype)

def reconstruct_frame(mv_dump, prev_frame, residual_frame, params_i, height, width):
    reconstructed_frame = []
    reconstructed_counter = 0
    y_counter = 0
    x_counter = 0
    for mv_row in mv_dump:
        reconstructed_frame.append([])
        for mv in mv_row:
            yx = mv
            block_coor = (y_counter, x_counter)
            block_in_prev_frame = prev_frame[yx[0]:yx[0] + params_i, yx[1]:yx[1] + params_i]
            residual_block = residual_frame[block_coor[0]:block_coor[0] + params_i, block_coor[1]:block_coor[1] + params_i]
            if residual_block.shape != block_in_prev_frame.shape:
                raise Exception('Shape mismatch. Current frame: {}, Previous frame: {}, Residual block: {}. YX: {}'.format(residual_frame.shape, prev_frame.shape, residual_block.shape, yx))
            reconstructed_block = block_in_prev_frame + residual_block
            reconstructed_frame[reconstructed_counter].append(reconstructed_block)
            x_counter += params_i
        reconstructed_counter += 1
        y_counter += params_i
        x_counter = 0
    current_reconstructed_frame = pixel_create(np.array(reconstructed_frame), (height, width), params_i)
    return current_reconstructed_frame