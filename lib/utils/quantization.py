import numpy as np
from scipy.fftpack import dct, idct
from lib.utils.misc import pixel_create

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

def get_qtc_and_reconstructed_block(current_block, predicted_block, q_matrix):
    residual_block = current_block - predicted_block
    residual_block_transformed = dct2(residual_block).astype(int)
    qtc_dump = tc_to_qtc(residual_block_transformed, q_matrix)

    residual_block_transformed = qtc_to_tc(qtc_dump, q_matrix)
    residual_block = idct2(residual_block_transformed).astype(int)
    reconstructed_block = predicted_block + residual_block
    return qtc_dump, reconstructed_block