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

def binstr_to_bytes(s: str) -> bytearray:
    """
        Convert a binary string to bytes.

        Parameters:
            s (str): The binary string.

        Returns:
            bytearray: The bytes.
    """
    padding = ''.join('0' for _ in range(8 - len(s) % 8))
    s += padding
    byte = bytearray()
    while len(s) > 0:
        x = int(s[-8:], 2)
        byte.append(x)
        s = s[:-8]
    return byte

def bytes_to_binstr(bytes: bytearray) -> str:
    """
        Convert bytes to a binary string.

        Parameters:
            bytes (bytearray): The bytes.

        Returns:
            str: The binary string.
    """
    s = ''
    for byte in bytes:
        s = bin(byte)[2:].zfill(8) + s
    return s

"""
    Encode a number using exponential-Golomb encoding extension to negative numbers.

    Parameters:
        number (int): The number to encode.

    Returns:
        str: The encoded number.
"""
def exp_golomb_encoding(number: int) -> str:
    if number <= 0:
        number = -2 * number
    else:
        number = 2 * number - 1
    number += 1
    binary = bin(number)[2:]
    padding = '0' * (len(binary) - 1)
    return padding + binary

"""
    Decode a number using exponential-Golomb encoding extension to negative numbers.

    Parameters:
        number (str): The number to decode.

    Returns:
        int: The decoded number.
"""  
def exp_golomb_decoding(number: str) -> int:
    padding = number.index('1')
    binary = number[padding:]
    number = int(binary, 2)
    number -= 1
    if number % 2 == 0:
        return -number // 2
    else:
        return (number + 1) // 2

def array_exp_golomb_decoding(number: str) -> list:
    """
        Decode an array of numbers using exponential-Golomb encoding extension to negative numbers.

        Parameters:
            number (str): The number to decode.

        Returns:
            list: The decoded numbers.
    """
    numbers = []
    counter = 0
    pending = ''
    while len(number) > 0:
        current = number[0]
        number = number[1:]
        pending += current
        if current == '0':
            counter += 1
        else:
            for _ in range(counter):
                pending += number[0]
                number = number[1:]
            numbers.append(exp_golomb_decoding(pending))
            pending = ''
            counter = 0
    return numbers
