from lib.utils.misc import binstr_to_bytes, bytes_to_binstr
from lib.utils.entropy import exp_golomb_encoding
from lib.utils.enums import Intraframe, TypeMarker
from lib.utils.entropy import array_exp_golomb_decoding

class MotionVector:

    def __init__(self, y, x, mae=0):
        self.y = y
        self.x = x
        self.mae = mae
        self.raw = (self.y, self.x)

    def l1_norm(self):
        return abs(self.y) + abs(self.x)

    def __add__(self, other):
        return MotionVector(self.y + other.y, self.x + other.x)
    
    def __sub__(self, other):
        return MotionVector(self.y - other.y, self.x - other.x)
    
class MotionVectorFrame:

    def __init__(self, is_intraframe=False, length=0):
        self.is_intraframe = is_intraframe
        self.raw = [None] * length

    def new_row(self):
        self.raw.append([])

    def append(self, mv):
        self.raw[-1].append(mv)
    
    def append_list(self, index, mvs):
        self.raw[index] = mvs

    def average_mae(self):
        counter = 0
        total = 0
        for object in self.raw:
            for item in object:
                total += item.mae
                counter += 1
        return total / counter

    def tobytes(self):
        text = ''
        if self.is_intraframe:
            prev_mv = MotionVector(Intraframe.HORIZONTAL.value, 0)
            text += '{}'.format(TypeMarker.I_FRAME.value)
        else:
            prev_mv = MotionVector(0, 0)
            text += '{}'.format(TypeMarker.P_FRAME.value)
        
        for i in range(len(self.raw)):
            for j in range(len(self.raw[i])):
                diff_mv = self.raw[i][j] - prev_mv
                prev_mv = self.raw[i][j]
                if self.is_intraframe:
                    text += '{}'.format(exp_golomb_encoding(diff_mv.y))
                else:
                    text += '{}{}'.format(exp_golomb_encoding(diff_mv.y), exp_golomb_encoding(diff_mv.x))
        return binstr_to_bytes(text)

    def read_from_file(self, path, width, params_i):
        mv = path.read_bytes()
        mv = bytes_to_binstr(mv)
        type_marker = int(mv[0])
        mv = mv[1:]
        if type_marker == TypeMarker.I_FRAME.value:
            self.is_intraframe = True
            prev_mv = MotionVector(Intraframe.HORIZONTAL.value, 0)
        else:
            self.is_intraframe = False
            prev_mv = MotionVector(0, 0)
        mv = array_exp_golomb_decoding(mv)
        mv_counter = 0
        if self.is_intraframe:
            for item in mv:
                if mv_counter == 0:
                    self.new_row()
                current_mv = prev_mv + MotionVector(item, 0)
                self.append(current_mv)
                prev_mv = current_mv
                mv_counter += 1
                if mv_counter == width // params_i:
                    mv_counter = 0
        else:
            for j in range(0, len(mv), 2):
                if mv_counter == 0:
                    self.new_row()
                current_mv = prev_mv + MotionVector(mv[j], mv[j + 1])
                self.append(current_mv)
                prev_mv = current_mv
                mv_counter += 1
                if mv_counter == width // params_i:
                    mv_counter = 0
