import math
import numpy as np
from scipy.fftpack import dct, idct

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
        np_array (np.ndarray): The block-based frame. The format should match the output @ block_create
        shape (height, width) (tuple): The shape of the pixel-based frame.
        params_i (int): The block size.
    
    Returns:
        np_pixel_array (np.ndarray): The pixel-based frame.
"""
def pixel_create(np_array: np.ndarray, shape: tuple, params_i: int) -> np.ndarray:
    offset = shape[1] // params_i

    e = []
    for i in range(offset):
        e.append(np_array[:, i].reshape(-1, params_i))
    
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
    np_block_array = np.block(b).reshape(-1, offset, params_i, params_i) # (-1,  width // i, i**2)
    return (np_block_array, offset, block_size, np_array_padded)

"""
    Round a number to the nearest multiple of the base.

    Parameters:
        x (int): The number to round.
        base (int): The base to round to.

    Returns:
        (int): The rounded number.
"""
def rounding(x: int, base: int) -> int:
    return base * round(x / base)

"""
    Convert a numpy array to a specific data type, and clip the values to within the range.

    Parameters:
        np_array (np.ndarray): The numpy array.
        dtype (np.dtype): The data type to convert to.

    Returns:
        (np.ndarray): The converted numpy array.
"""
def convert_within_range(np_array: np.ndarray, dtype: np.dtype=np.uint8) -> np.ndarray:
    return np.clip(np_array, 0, 255).astype(dtype)

"""
    Construct the predicted frame from the motion vector dump.

    Parameters:
        mv_dump (list): The motion vector dump.
        prev_frame (np.ndarray): The previous frame.
        params_i (int): The block size.
    
    Returns:
        (np.ndarray): The predicted frame.
"""
def construct_predicted_frame(mv_dump: list, prev_frame: np.ndarray, params_i: int) -> np.ndarray:
    predicted_frame_dump = []
    y_counter = 0
    x_counter = 0
    for i in range(len(mv_dump)):
        predicted_frame_dump.append([])
        for j in range(len(mv_dump[i])):
            top_left = mv_dump[i][j]
            predicted_frame_dump[i].append(prev_frame[y_counter + top_left[0]:y_counter + top_left[0] + params_i, x_counter + top_left[1]:x_counter + top_left[1] + params_i])
            x_counter += params_i
        y_counter += params_i
        x_counter = 0
    return pixel_create(np.array(predicted_frame_dump), prev_frame.shape, params_i)

"""
    2D DCT.

    Parameters:
        np_array (np.array): The numpy array.
    
    Returns:
        (np.array): The DCT-ed numpy array.
"""
def dct2(np_array: np.array) -> np.array:
    return dct(dct(np_array.T, norm='ortho').T, norm='ortho')

"""
    2D IDCT.

    Parameters:
        np_array (np.array): The numpy array.
    
    Returns:
        (np.array): The IDCT-ed numpy array.
"""
def idct2(np_array: np.array) -> np.array:
    return idct(idct(np_array.T, norm='ortho').T, norm='ortho')

"""
    Transform the residual frame into residual coefficients.

    Parameters:
        residual_frame (np.ndarray): The residual frame. The format should match the output @ block_create
        params_i (int): The block size.

    Returns:
        (np.ndarray): The residual coefficients. The format matches the output @ block_create
"""
def frame_dct2(np_array, params_i):
    new_np_array = np_array.reshape(-1, params_i, params_i)
    a = []
    for item in new_np_array:
        a.append(dct2(item).astype(int))
    return np.array(a).reshape(-1, np_array.shape[1], params_i, params_i)

"""
    Transform residual coefficients into the residual frame.

    Parameters:
        residual_coefficients (np.ndarray): The residual coefficients. The format should match the output @ block_create
        params_i (int): The block size.
    
    Returns:
        (np.ndarray): The residual frame. The format matches the output @ block_create
"""
def frame_idct2(np_array: np.ndarray, params_i: int) -> np.ndarray:
    new_np_array = np_array.reshape(-1, params_i, params_i)
    a = []
    for item in new_np_array:
        a.append(idct2(item).astype(int))
    return np.array(a).reshape(-1, np_array.shape[1], params_i, params_i)

"""
    Transform the residual frame into residual coefficients.

    Parameters:
        residual_frame (np.ndarray): The residual frame. The format should match the output @ block_create
        params_i (int): The block size.
    
    Returns:
        (np.ndarray): The residual coefficients.
"""
def residual_coefficients_to_residual_frame(residual_coefficients: np.ndarray, params_i: int, shape: tuple) -> np.ndarray:
    return pixel_create(frame_idct2(residual_coefficients, params_i), shape, params_i)

"""
    Generate quantization matrix.

    Parameters:
        params_i (int): The block size.
        params_qp (int): The quantization parameter.
    
    Returns:
        (np.ndarray): The quantization matrix.
"""
def quantization_matrix(params_i: int, params_qp: int) -> np.ndarray:
    np_array = np.empty((params_i, params_i))
    for x in range(params_i):
        for y in range(params_i):
            if x + y < params_i - 1:
                np_array[x][y] = 2 ** params_qp
            elif x + y == params_i - 1:
                np_array[x][y] = 2 ** (params_qp + 1)
            else:
                np_array[x][y] = 2 ** (params_qp + 2)
    return np_array.astype(int)

"""
    Transform coefficients into quantized coefficients.

    Parameters:
        coefficients (np.ndarray): The coefficients.
        q_matrix (np.ndarray): The quantization matrix.
    
    Returns:
        (np.ndarray): The quantized coefficients.
"""
def tc_to_qtc(block: np.ndarray, q_matrix: np.ndarray) -> np.ndarray:
    return np.round(block / q_matrix).astype(int)

"""
    Transform quantized coefficients into coefficients.

    Parameters:
        qtc (np.ndarray): The quantized coefficients.
        q_matrix (np.ndarray): The quantization matrix.
    
    Returns:
        (np.ndarray): The coefficients.
"""
def qtc_to_tc(block: np.ndarray, q_matrix: np.ndarray) -> np.ndarray:
    return block * q_matrix

def frame_qtc_to_tc(frame_qtc: np.ndarray, q_matrix: np.ndarray) -> np.ndarray:
    new_array = np.empty(frame_qtc.shape)
    for y in range(frame_qtc.shape[0]):
        for x in range(frame_qtc.shape[1]):
            new_array[y, x] = qtc_to_tc(frame_qtc[y, x], q_matrix)
    return new_array

def frame_tc_to_qtc(frame_tc: np.ndarray, q_matrix: np.ndarray) -> np.ndarray:
    new_array = np.empty(frame_tc.shape)
    for y in range(frame_tc.shape[0]):
        for x in range(frame_tc.shape[1]):
            new_array[y, x] = tc_to_qtc(frame_tc[y, x], q_matrix)
    return new_array