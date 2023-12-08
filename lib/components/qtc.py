import pathlib
import numpy as np
from scipy.fftpack import dct, idct
from lib.utils.misc import binstr_to_bytes, bytes_to_binstr, array_exp_golomb_decoding, exp_golomb_encoding
from lib.components.frame import Frame
from lib.enums import VBSMarker

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

    def __init__(self, block=None, qtc_block=None, q_matrix=None, qp = 0):
        self.qtc = None
        self.block = block
        self.q_matrix = q_matrix
        self.qtc_block = qtc_block
        self.str = None
        self.qp = qp

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
    
    def to_str(self) -> str:
        if self.str is None:
            self.str = '{}{}'.format(''.join(exp_golomb_encoding(x) for x in rle_encoding(reording_encoding(self.qtc_block))), exp_golomb_encoding(self.qp))
        return self.str

class QTCFrame:
    
    def __init__(self, params_i: int= 1, length=0, vbs_enable=False, shape=None):
        self.blocks = [None] * length if length > 0 else []
        if shape is not None:
            self.blocks = [[None] * shape[1] for _ in range(shape[0])]
        self.shape = None
        self.params_i = params_i
        self.vbs_enable = vbs_enable

    def get_average_qp(self) -> int:
        """
            Get the average qp of the QTC frame.

            Returns:
                int: The average qp.
        """
        qp_sum = 0
        for row in self.blocks:
            for block in row:
                if self.vbs_enable:
                    block = block['qtc_block']
                qp_sum += block.qp
        return qp_sum // (len(self.blocks) * len(self.blocks[0]))

    def new_row(self) -> None:
        """
            Add a new row.
        """
        self.blocks.append([])

    def set(self, coor: tuple, block: QTCBlock):
        """
            Set a block at a coordinate.
        """
        self.blocks[coor[0]][coor[1]] = block

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
        if self.vbs_enable:
            for i in range(len(self.blocks)):
                for j in range(len(self.blocks[i])):
                    item = self.blocks[i][j]
                    vbs = item['vbs']
                    block = item['qtc_block']
                    text += '{}{}'.format(exp_golomb_encoding(vbs.value), block.to_str())
        else:
            for object in self.blocks:
                for item in object:
                    text += item.to_str()
        return binstr_to_bytes(text)
    
    def read_from_file(self, path: pathlib.Path, width: int) -> None:
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
        index_counter = 0
        while index_counter < len(qtc):
            item = qtc[index_counter]
            index_counter += 1
            if self.vbs_enable:
                vbs = VBSMarker(item)
                item = -1
                while item != 0:
                    item = qtc[index_counter]
                    index_counter += 1
                    qtc_pending.append(item)
                    if item == 0:
                        if qtc_counter == 0:
                            self.new_row()
                        qp = qtc[index_counter]
                        index_counter += 1
                        q_matrix = quantization_matrix(self.params_i, qp)
                        array = np.array(reording_decoding(rle_decoding(qtc_pending, q_matrix.shape), q_matrix.shape)).astype(int)
                        if vbs is VBSMarker.SPLIT:
                            subblock_params_i = self.params_i // 2
                            sub_q_matrix = quantization_matrix(subblock_params_i, qp)
                            top_lefts = [(y, x) for y in range(0, self.params_i, subblock_params_i) for x in range(0, self.params_i, subblock_params_i)]
                            sub_qtc_blocks = []
                            for centered_top_left in top_lefts:
                                centered_subblock = array[centered_top_left[0]:centered_top_left[0] + subblock_params_i, centered_top_left[1]:centered_top_left[1] + subblock_params_i]
                                qtc_block = QTCBlock(qtc_block=centered_subblock, q_matrix=sub_q_matrix, qp=qp)
                                qtc_block.qtc_to_block()
                                sub_qtc_blocks.append(qtc_block)
                            qtc_block = QTCBlock(block=np.concatenate((np.concatenate((sub_qtc_blocks[0], sub_qtc_blocks[1]), axis=1), np.concatenate((sub_qtc_blocks[2], sub_qtc_blocks[3]), axis=1)), axis=0), qtc_block=array)
                        else:
                            qtc_block = QTCBlock(qtc_block=array, q_matrix=q_matrix, qp=qp)
                            qtc_block.qtc_to_block()
                        self.append(qtc_block)
                        qtc_pending = []
                        qtc_counter += 1
                        if qtc_counter == width // self.params_i:
                            qtc_counter = 0
            else:
                qtc_pending.append(item)
                if item == 0:
                    if qtc_counter == 0:
                        self.new_row()
                    qp = qtc[index_counter]
                    index_counter += 1
                    q_matrix = quantization_matrix(self.params_i, qp)
                    qtc_block = QTCBlock(qtc_block=np.array(reording_decoding(rle_decoding(qtc_pending, q_matrix.shape), q_matrix.shape)).astype(int), q_matrix=q_matrix)
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
