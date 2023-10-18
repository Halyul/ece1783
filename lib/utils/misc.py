import math
import numpy as np
from scipy.fftpack import dct, idct
from lib.utils.enums import Intraframe

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
        prev_frame (np.ndarray): The previous frame. If None, then the frame is intraframe.
        residual_frame (np.ndarray): The residual frame.
        params_i (int): The block size.
    
    Returns:
        (np.ndarray): The reconstructed frame.
"""
def construct_reconstructed_frame(mv_dump: list, prev_frame: np.ndarray, residual_frame: np.ndarray, params_i: int) -> np.ndarray:
    y_counter = 0
    x_counter = 0
    if prev_frame is None:
        # is intraframe
        height, width = residual_frame.shape
        reconstructed_block_dump = np.empty(residual_frame.shape, dtype=int)
        for y in range(len(mv_dump)):
            for x in range(len(mv_dump[y])):
                current_coor = (y_counter, x_counter)
                predictor = mv_dump[y][x]
                if x == 0 and predictor == Intraframe.HORIZONTAL.value:
                    # first column of blocks in horizontal
                    predictor_block = np.full((params_i, 1), 128)
                    repeat_value = Intraframe.HORIZONTAL.value
                elif y == 0 and predictor == Intraframe.VERTICAL.value:
                    # first row of blocks in vertical
                    predictor_block = np.full((1, params_i), 128)
                    repeat_value = Intraframe.VERTICAL.value
                elif predictor == Intraframe.HORIZONTAL.value:
                    # horizontal
                    hor_top_left, _ = extend_block(current_coor, params_i, (0, 0, 0, 1), (height, width))
                    predictor_block = reconstructed_block_dump[hor_top_left[0]:hor_top_left[0] + params_i, hor_top_left[1]:hor_top_left[1] + 1]
                    repeat_value = Intraframe.HORIZONTAL.value
                elif predictor == Intraframe.VERTICAL.value:
                    # vertical
                    ver_top_left, _ = extend_block(current_coor, params_i, (1, 0, 0, 0), (height, width))
                    predictor_block = reconstructed_block_dump[ver_top_left[0]:ver_top_left[0] + 1, ver_top_left[1]:ver_top_left[1] + params_i]
                    repeat_value = Intraframe.VERTICAL.value
                else:
                    raise Exception('Invalid predictor.')
                predictor_block = predictor_block.repeat(params_i, repeat_value)
                residual_block = residual_frame[y_counter:y_counter + params_i, x_counter:x_counter + params_i]
                reconstructed_block_dump[y_counter:y_counter + params_i, x_counter:x_counter + params_i] = predictor_block + residual_block
                x_counter += params_i
            y_counter += params_i
            x_counter = 0
    else:
        predicted_frame_dump = []
        for i in range(len(mv_dump)):
            predicted_frame_dump.append([])
            for j in range(len(mv_dump[i])):
                top_left = mv_dump[i][j]
                predicted_frame_dump[i].append(prev_frame[y_counter + top_left[0]:y_counter + top_left[0] + params_i, x_counter + top_left[1]:x_counter + top_left[1] + params_i])
                x_counter += params_i
            y_counter += params_i
            x_counter = 0
        reconstructed_block_dump = pixel_create(np.array(predicted_frame_dump), prev_frame.shape, params_i) + residual_frame
    return reconstructed_block_dump

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

"""
    Process full residual frame quantized coefficients to coefficients.

    Parameters:
        frame_qtc (np.ndarray): The quantized coefficients.
        q_matrix (np.ndarray): The quantization matrix.
    
    Returns:
        (np.ndarray): The coefficients.
"""
def frame_qtc_to_tc(frame_qtc: np.ndarray, q_matrix: np.ndarray) -> np.ndarray:
    new_array = np.empty(frame_qtc.shape)
    for y in range(frame_qtc.shape[0]):
        for x in range(frame_qtc.shape[1]):
            new_array[y, x] = qtc_to_tc(frame_qtc[y, x], q_matrix)
    return new_array

"""
    Process full residual frame coefficients to quantized coefficients.

    Parameters:
        frame_tc (np.ndarray): The coefficients.
        q_matrix (np.ndarray): The quantization matrix.

    Returns:
        (np.ndarray): The quantized coefficients.
"""
def frame_tc_to_qtc(frame_tc: np.ndarray, q_matrix: np.ndarray) -> np.ndarray:
    new_array = np.empty(frame_tc.shape)
    for y in range(frame_tc.shape[0]):
        for x in range(frame_tc.shape[1]):
            new_array[y, x] = tc_to_qtc(frame_tc[y, x], q_matrix)
    return new_array

"""
    Extend the block to include the margin.

    Parameters:
        original_top_left (tuple): The top left corner of the block.
        params_i (int): The block size.
        margin (tuple): The margin (top, right, bottom, left).
        shape (tuple): The shape of the frame (height, width).

    Returns:
        (tuple): The extended block (top_left, bottom_right).
"""
def extend_block(original_top_left: tuple, params_i: int, margin: tuple, shape: tuple) -> tuple:
    top, right, bottom, left = margin
    max_height, max_width = shape
    top_left = (original_top_left[0] - top, original_top_left[1] - left)
    if top_left[0] < 0:
        top_left = (0, top_left[1])
    if top_left[1] < 0:
        top_left = (top_left[0], 0)

    bottom_right = (original_top_left[0] + params_i + bottom, original_top_left[1] + params_i + right)
    if bottom_right[0] > max_height:
        bottom_right = (max_height, bottom_right[1])
    if bottom_right[1] > max_width:
        bottom_right = (bottom_right[0], max_width)
    
    return top_left, bottom_right

def get_qtc_and_reconstructed_block(current_block, predicted_block, q_matrix):
    residual_block = current_block - predicted_block
    residual_block_transformed = dct2(residual_block).astype(int)
    qtc_dump = tc_to_qtc(residual_block_transformed, q_matrix)

    residual_block_transformed = qtc_to_tc(qtc_dump, q_matrix)
    residual_block = idct2(residual_block_transformed).astype(int)
    reconstructed_block = predicted_block + residual_block
    return qtc_dump, reconstructed_block