import numpy as np
from lib.utils.quantization import dct2, idct2, tc_to_qtc, qtc_to_tc
from lib.utils.entropy import *
from lib.utils.misc import binstr_to_bytes, bytes_to_binstr
from lib.components.frame import Frame

class QTCBlock:

    def __init__(self, block=None, qtc_block=None, q_matrix=None):
        self.qtc = None
        self.block = block
        self.q_matrix = q_matrix
        self.qtc_block = qtc_block

    def tc(self):
        return dct2(self.block).astype(int)
    
    def __block_to_qtc(self):
        self.qtc_block = tc_to_qtc(self.tc(), self.q_matrix)
    
    def __qtc_to_block(self):
        self.block = idct2(qtc_to_tc(self.qtc_block, self.q_matrix)).astype(int)
    
    def block_to_qtc(self):
        self.__block_to_qtc()
        self.__qtc_to_block()

    def qtc_to_block(self):
        self.__qtc_to_block()

class QTCFrame:
    
    def __init__(self, length=0):
        self.blocks = [None] * length

    def new_row(self):
        self.blocks.append([])

    def append(self, block):
        self.blocks[-1].append(block)

    def append_list(self, index, blocks):
        self.blocks[index] = blocks

    def tobytes(self):
        text = ''
        for object in self.blocks:
            for item in object:
                text += ''.join(exp_golomb_encoding(x) for x in rle_encoding(reording_encoding(item.qtc_block)))
        return binstr_to_bytes(text)
    
    def read_from_file(self, path, q_matrix, width, params_i):
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
                qtc_block = QTCBlock(qtc_block=np.array(reording_decoding(rle_decoding(qtc_pending, q_matrix.shape), q_matrix.shape)).astype(int), q_matrix=q_matrix)
                qtc_block.qtc_to_block()
                self.append(qtc_block.block) # CAUTION: the format is different
                qtc_pending = []
                qtc_counter += 1
                if qtc_counter == width // params_i:
                    qtc_counter = 0

    def to_residual_frame(self, height, width, params_i):
        frame = Frame(-1, height, width, params_i)
        frame.block_to_pixel(np.array(self.blocks))
        return frame