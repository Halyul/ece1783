import pathlib
import numpy as np
from scipy.fftpack import dct, idct
from lib.utils.misc import binstr_to_bytes, bytes_to_binstr, array_exp_golomb_decoding, exp_golomb_encoding
from lib.components.frame import Frame


def quantization_matrix(params_i: int, params_qp: int) -> np.ndarray:
    """
        Generate quantization matrix.

        Parameters:
            params_i (int): The block size.
            params_qp (int): The quantization parameter.
        
        Returns:
            (np.ndarray): The quantization matrix.
    """
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

def reordering_helper(shape: tuple, current: tuple, row: int) -> tuple:
    """
        Reordering helper

        Parameters:
            shape (tuple): The shape of the matrix.
            current (tuple): The current position.
            row (int): The current row.
        
        Returns:
            tuple: The new position and row.
    """
    if current[1] == row:
        if current[0] == shape[0] - 1:
            row += 1
        current = (row, current[0] + 1)
        if current[1] >= shape[1] - 1:
            current = (row, shape[1] - 1)
    else:
        current = (current[0] + 1, current[1] - 1)
    return current, row

class QTCBlock:

    def __init__(self, block=None, qtc_block=None, q_matrix=None):
        self.qtc = None
        self.block = block
        self.q_matrix = q_matrix
        self.qtc_block = qtc_block

    def tc(self) -> np.ndarray:
        """
            Pixel value to transform coefficient.

            Returns:
                np.ndarray: The transform coefficient.
        """
        return dct(dct(self.block.T, norm='ortho').T, norm='ortho').astype(int)
    
    def __block_to_qtc(self):
        """
            Transform coefficient to qtc values.
        """
        self.qtc_block = np.round(self.tc() / self.q_matrix).astype(int)
    
    def __qtc_to_block(self):
        """
            QTC values to pixel values.
        """
        self.block = idct(idct((self.qtc_block * self.q_matrix).T, norm='ortho').T, norm='ortho').astype(int)
    
    def block_to_qtc(self):
        """
            Pixel values to qtc values.
        """
        self.__block_to_qtc()
        self.__qtc_to_block()

    def qtc_to_block(self):
        """
            QTC values to pixel values.
        """
        self.__qtc_to_block()

    """
        Return the block by default for numpy operations.

        Returns:
            np.ndarray: The block.
    """
    def __array__(self):
        return self.block

class QTCFrame:
    
    def __init__(self, params_i: int= 1, length=0):
        self.blocks = [None] * length
        self.shape = None
        self.params_i = params_i

    def new_row(self) -> None:
        """
            Add a new row.
        """
        self.blocks.append([])

    def append(self, block: QTCBlock) -> None:
        """
            Append a block to the current row.

            Parameters:
                block (QTCBlock): The block to append.
        """
        self.blocks[-1].append(block)

    def append_list(self, index: int, blocks: list) -> None:
        """
            Append a list of blocks to the current row.

            Parameters:
                index (int): The index of the row.
                blocks (list): The block list to append.
        """
        self.blocks[index] = blocks

    def tobytes(self) -> bytes:
        """
            Convert the QTC frame to bytes.

            Returns:
                bytes: The QTC frame as bytes.
        """
        text = ''
        for object in self.blocks:
            for item in object:
                text += ''.join(exp_golomb_encoding(x) for x in self.rle_encoding(self.reording_encoding(item.qtc_block)))
        return binstr_to_bytes(text)
    
    def read_from_file(self, path: pathlib.Path, q_matrix: np.ndarray, width: int) -> None:
        """
            Read QTC Frame from file

            Parameters:
                path (pathlib.Path): The path to read from.
                q_matrix (np.ndarray): The quantization matrix.
                width (int): The width of the frame.
        """
        qtc = path.read_bytes()
        qtc = bytes_to_binstr(qtc)
        qtc = array_exp_golomb_decoding(qtc)
        qtc_counter = 0
        qtc_pending = []
        for item in qtc:
            qtc_pending.append(item)
            if item == 0:
                if qtc_counter == 0:
                    self.new_row()
                qtc_block = QTCBlock(qtc_block=np.array(self.reording_decoding(self.rle_decoding(qtc_pending, q_matrix.shape), q_matrix.shape)).astype(int), q_matrix=q_matrix)
                qtc_block.qtc_to_block()
                self.append(qtc_block)
                qtc_pending = []
                qtc_counter += 1
                if qtc_counter == width // self.params_i:
                    qtc_counter = 0
        self.shape = (len(self.blocks) * self.params_i, width)

    def to_residual_frame(self) -> Frame:
        """
            Convert QTC Frame to Residual Frame

            Returns:
                Frame: The residual frame.
        """
        frame = Frame(-1, self.shape[0], self.shape[1], self.params_i)
        frame.block_to_pixel(np.array(self.blocks))
        return frame
    
    @staticmethod
    def rle_encoding(array: list) -> list:
        """
            Run-length encoding.

            Parameters:
                array (list): The array to encode.

            Returns:
                list: The encoded array.
        """
        new_array = []
        pending = []
        counter = 0
        for item in array:
            if item == 0:
                if counter < 0:
                    new_array.append(counter)
                    new_array += pending
                    pending = []
                    counter = 0
                counter += 1
            else:
                if counter > 0:
                    new_array.append(counter)
                    counter = 0
                counter -= 1
                pending.append(item)
        if len(pending) > 0:
            new_array.append(counter)
            new_array += pending
        new_array.append(0)
        return new_array

    @staticmethod
    def rle_decoding(array: list, shape: tuple) -> list:
        """
            Run-length decoding.

            Parameters:
                array (list): The array to decode.
                shape (tuple): The shape of the matrix.

            Returns:
                list: The decoded array.
        """
        max_length = shape[0] * shape[1]
        new_array = []
        counter = 0
        for i in range(len(array)):
            item = array[i]
            if counter == 0:
                counter = item
                if counter == 0:
                    offset = max_length - len(new_array)
                    if offset > 0:
                        counter = offset
                if counter > 0:
                    for i in range(counter):
                        new_array.append(0)
                    counter = 0
                continue
            if counter < 0:
                new_array.append(item)
                counter += 1
            else:
                new_array.append(0)
                counter -= 1
        
        return new_array
    
    @staticmethod
    def reording_encoding(qtc_dump: list) -> list:
        """
            Matrix reording encoding.

            Parameters:
                qtc_dump (list): The qtc dump to encode.

            Returns:
                list: The encoded qtc dump.
        """
        array = []
        current = (0, 0)
        row = 0
        end = (len(qtc_dump), len(qtc_dump[0])) # (rows, columns)
        for _ in range(end[0] * end[1]):
            array.append(qtc_dump[current[0]][current[1]])
            current, row = reordering_helper(end, current, row)
        return array

    @staticmethod
    def reording_decoding(reordered_dump: list, shape: tuple) -> list:
        """
            Matrix reording decoding.

            Parameters:
                reordered_dump (list): The reordered dump to decode.
                shape (tuple): The shape of the matrix.
            
            Returns:
                list: The decoded qtc dump.
        """
        matrix = np.empty(shape, dtype=int)
        current = (0, 0)
        row = 0
        for item in reordered_dump:
            matrix[current[0], current[1]] = item
            current, row = reordering_helper(shape, current, row)
        return matrix