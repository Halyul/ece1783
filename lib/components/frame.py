import numpy as np
from lib.utils.misc import block_create, pixel_create, convert_within_range

class Frame:

    def __init__(self, index, height, width, params_i=1, is_intraframe=False, data=None, prev=None):
        self.index = index
        self.height = height
        self.width = width
        self.shape = (height, width)
        self.prev = prev
        self.raw = data
        self.params_i = params_i
        self.is_intraframe = is_intraframe

    def read_from_file(self, file_path, dtype=np.uint8):
        self.raw = np.fromfile(file_path, dtype=dtype).reshape(self.height, self.width)
        self.prev = Frame(self.index - 1, self.height, self.width, data=np.full(self.height*self.width, 128).reshape(self.height, self.width))

    def read_prev_from_file(self, file_path, index, dtype=np.uint8):
        self.prev = Frame(index, self.height, self.width, params_i=self.params_i)
        self.prev.read_from_file(file_path, dtype=dtype)

    def convert_type(self, dtype):
        self.raw = self.raw.astype(dtype)

    def pixel_to_block(self):
        np_block_array, _, _, _ = block_create(self.raw, self.params_i)
        return np_block_array
    
    def block_to_pixel(self, np_array: np.ndarray):
        self.raw = pixel_create(np_array, self.shape, self.params_i).astype(np.int16)
        return self.raw
    
    def convert_within_range(self, dtype: np.dtype=np.uint8):
        self.raw = convert_within_range(self.raw, dtype=dtype)

    def dump(self, path):
        path.write_bytes(self.raw.tobytes())

    def __add__(self, other):
        return Frame(self.index, self.height, self.width, data=self.raw + other.raw)
    
    def __sub__(self, other):
        return Frame(self.index, self.height, self.width, data=self.raw - other.raw)