import pathlib, math
from typing import Literal
import numpy as np

def yuv2rgb(y: np.ndarray, u: np.ndarray = None, v: np.ndarray = None) -> np.ndarray:
    """
        Convert YUV to RGB.

        Parameters:
            y (np.ndarray): The Y channel.
            u (np.ndarray): The U channel.
            v (np.ndarray): The V channel.
        
        Returns:
            (r, g, b, rgb) (tuple): The R, G, B channels and the RGB array.
    """
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

def get_padding(width: int, height: int, n: int) -> tuple:
    """
        Get the padding of the frame.

        Parameters:
            width (int): The width of the frame.
            height (int): The height of the frame.
            n (int): The length of the block.
        
        Returns:
            (width, height) (tuple): The padded width and height of the frame.
    """
    pad_width = math.ceil(width / n) * n if width > 0 else -1
    pad_height = math.ceil(height / n) * n if height > 0 else -1
    return pad_width, pad_height

def pixel_create(np_array: np.ndarray, shape: tuple, params_i: int) -> np.ndarray:
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
    offset = shape[1] // params_i

    e = []
    for i in range(offset):
        e.append(np_array[:, i].reshape(-1, params_i))
    
    np_pixel_array = np.block(e)
    if not shape == np_pixel_array.shape:
        raise Exception('Shape mismatch.')
    return np_pixel_array

def block_create(np_array: np.ndarray, params_i: int) -> tuple:
    """
        Transform the pixel-based frame into an i x i sized block-based frame.
        Y component only.

        Parameters:
            np_array (np.ndarray): The pixel-based frame.
            params_i (int): The block size.

        Returns:
            (np_block_array, offset, block_size, np_array_padded) (tuple): The block-based frame, the offset, the block size, and the padded pixel-based frame.
    """
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

def convert_within_range(np_array: np.ndarray, dtype: np.dtype=np.uint8) -> np.ndarray:
    """
        Convert a numpy array to a specific data type, and clip the values to within the range.

        Parameters:
            np_array (np.ndarray): The numpy array.
            dtype (np.dtype): The data type to convert to.

        Returns:
            (np.ndarray): The converted numpy array.
    """
    return np.clip(np_array, 0, 255).astype(dtype)

def extend_block(original_top_left: tuple, params_i: int, margin: tuple, shape: tuple) -> tuple:
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

class Frame:

    def __init__(self, index: int=-1, height: int=0, width: int=0, params_i: int=1, is_intraframe: bool=False, data: np.ndarray=None, prev: Literal['Frame']=None, frame=None):
        if frame is None:
            self.index: int = index
            self.height = height
            self.width = width
            self.shape = (height, width)
            self.prev = prev
            self.raw: np.array = data
            self.params_i = params_i
            self.is_intraframe = is_intraframe
        else:
            self.__copy(frame)

    def __copy(self, frame):
        self.index = frame.index
        self.height = frame.height
        self.width = frame.width
        self.shape = frame.shape
        self.prev = frame.prev
        self.raw: np.array = frame.raw.copy()
        self.params_i = frame.params_i
        self.is_intraframe = frame.is_intraframe

    def copy(self):
        return Frame(frame=self)

    def read_from_file(self, file_path: pathlib.Path, dtype=np.uint8):
        """
            Read frame from file

            Parameters:
                file_path (pathlib.Path): The path to read from.
                dtype (np.dtype): The data type to read as.
        """
        self.raw = np.fromfile(file_path, dtype=dtype).reshape(self.height, self.width)

    def read_prev_from_file(self, file_path: pathlib.Path, index: int, dtype=np.uint8):
        """
            Read previous frame from file

            Parameters:
                file_path (pathlib.Path): The path to read from.
                index (int): The index of the previous frame.
                dtype (np.dtype): The data type to read as.
        """
        self.prev = Frame(index, self.height, self.width, params_i=self.params_i)
        self.prev.read_from_file(file_path, dtype=dtype)

    def convert_type(self, dtype: np.dtype):
        """
            Convert the frame to a different data type.

            Parameters:
                dtype (np.dtype): The data type to convert to.
        """
        self.raw = self.raw.astype(dtype)

    def pixel_to_block(self) -> np.ndarray:
        """
            Convert the frame to block.

            Returns:
                np.ndarray: The block array.
        """
        np_block_array, _, _, _ = block_create(self.raw, self.params_i)
        return np_block_array
    
    def block_to_pixel(self, array) -> np.ndarray:
        """
            Convert the frame to pixel.

            Parameters:
                np_array (np.ndarray): The block array.
        """
        self.raw = pixel_create(np.array(array), self.shape, self.params_i).astype(np.int16)
        return self.raw
    
    def convert_within_range(self, dtype: np.dtype=np.uint8):
        """
            Convert the frame within range.

            Parameters:
                dtype (np.dtype): The data type to convert to.
        """
        self.raw = convert_within_range(self.raw, dtype=dtype)

    def set(self, coor, data):
        """
            Set the value of a pixel.

            Parameters:
                coor (tuple): The coordinate of the pixel.
                data (int): The value to set.
        """
        self.raw[coor[0]:coor[0] + self.params_i, coor[1]:coor[1] + self.params_i] = data

    def dump(self, path: pathlib.Path):
        """
            Dump the frame to file.

            Parameters:
                path (pathlib.Path): The path to dump to.
        """
        path.write_bytes(self.raw.tobytes())

    def __add__(self, other):
        return Frame(self.index, self.height, self.width, data=self.raw + other.raw)
    
    def __sub__(self, other):
        return Frame(self.index, self.height, self.width, data=self.raw - other.raw)
