import pathlib
from typing import Literal
import numpy as np
from lib.utils.misc import block_create, pixel_create, convert_within_range

class Frame:

    def __init__(self, index: int, height: int, width: int, params_i: int=1, is_intraframe: bool=False, data: np.ndarray=None, prev: Literal['Frame']=None):
        self.index = index
        self.height = height
        self.width = width
        self.shape = (height, width)
        self.prev = prev
        self.raw = data
        self.params_i = params_i
        self.is_intraframe = is_intraframe

    def read_from_file(self, file_path: pathlib.Path, dtype=np.uint8):
        """
            Read frame from file

            Parameters:
                file_path (pathlib.Path): The path to read from.
                dtype (np.dtype): The data type to read as.
        """
        self.raw = np.fromfile(file_path, dtype=dtype).reshape(self.height, self.width)
        self.prev = Frame(self.index - 1, self.height, self.width, data=np.full(self.height*self.width, 128).reshape(self.height, self.width))

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
    
    def block_to_pixel(self, np_array: np.ndarray) -> np.ndarray:
        """
            Convert the frame to pixel.

            Parameters:
                np_array (np.ndarray): The block array.
        """
        self.raw = pixel_create(np_array, self.shape, self.params_i).astype(np.int16)
        return self.raw
    
    def convert_within_range(self, dtype: np.dtype=np.uint8):
        """
            Convert the frame within range.

            Parameters:
                dtype (np.dtype): The data type to convert to.
        """
        self.raw = convert_within_range(self.raw, dtype=dtype)

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