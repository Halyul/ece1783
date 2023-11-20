import pathlib
from lib.utils.misc import binstr_to_bytes, bytes_to_binstr, exp_golomb_encoding, array_exp_golomb_decoding
from lib.enums import Intraframe, TypeMarker, VBSMarker

class MotionVector:

    def __init__(self, y, x, ref_offset = 0, mae=0):
        self.y = y
        self.x = x
        self.ref_offset = ref_offset
        self.mae = mae
        self.raw = (self.y, self.x)

    def l1_norm(self) -> int:
        """
            L1 norm of the motion vector.

            Returns:
                int: The L1 norm.
        """
        return abs(self.y) + abs(self.x)

    def __add__(self, other):
        return MotionVector(self.y + other.y, self.x + other.x, self.ref_offset + other.ref_offset)
    
    def __sub__(self, other):
        return MotionVector(self.y - other.y, self.x - other.x, self.ref_offset - other.ref_offset)
    
    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, MotionVector):
            return False
        return (self.y == __value.y and self.x == __value.x and self.ref_offset == __value.ref_offset)
    
    def to_str(self, is_intraframe=False, fme_enabled=False) -> str:
        multiplier = 2 if fme_enabled else 1
        return exp_golomb_encoding(int(self.y * multiplier)) if is_intraframe else '{}{}{}'.format(exp_golomb_encoding(int(self.y * multiplier)), exp_golomb_encoding(int(self.x * multiplier)), exp_golomb_encoding(self.ref_offset))
    
class MotionVectorFrame:

    def __init__(self, is_intraframe=False, length=0, vbs_enable=False, fme_enable=False, shape=None):
        self.is_intraframe = is_intraframe
        self.raw = [None] * length if length > 0 else []
        if shape is not None:
            self.raw = [[None] * shape[1] for _ in range(shape[0])]
        self.vbs_enable = vbs_enable
        self.fme_enable = fme_enable

    def set(self, coor: tuple, mv: MotionVector):
        """
            Set a mv at a coordinate.
        """
        self.raw[coor[0]][coor[1]] = mv

    def new_row(self):
        """
            Add a new row.
        """
        self.raw.append([])

    def append(self, mv: MotionVector):
        """
            Append a motion vector to the last row.

            Parameters:
                mv (MotionVector): The motion vector.
        """
        self.raw[-1].append(mv)
    
    def append_list(self, index, mvs: list):
        """
            Append a list of motion vectors to the last row.

            Parameters:
                index (int): The index of the row.
                mvs (list): The motion vector list.
        """
        self.raw[index] = mvs

    def average_mae(self) -> float:
        """
            Average MAE of the motion vectors.

            Returns:
                float: The average MAE.
        """
        counter = 0
        total = 0
        if self.vbs_enable:
            for i in range(len(self.raw)):
                for j in range(len(self.raw[i])):
                    item = self.raw[i][j]
                    vbs = item['vbs']
                    block = item['predictor']
                    if vbs is VBSMarker.SPLIT:
                        for predictor in block:
                            total += predictor.mae
                            counter += 1
                    elif vbs is VBSMarker.UNSPLIT:
                        total += block.mae
                        counter += 1
                    else:
                        raise Exception('Invalid VBS Marker')
        else:
            for object in self.raw:
                for item in object:
                    total += item.mae
                    counter += 1
        return total / counter

    def tobytes(self) -> bytes:
        """
            Convert the motion vector frame to bytes.

            Returns:
                bytes: The motion vector frame as bytes.
        """
        text = ''
        if self.is_intraframe:
            prev_mv = MotionVector(Intraframe.HORIZONTAL.value, 0)
            text += '{}'.format(TypeMarker.I_FRAME.value)
        else:
            prev_mv = MotionVector(0, 0)
            text += '{}'.format(TypeMarker.P_FRAME.value)

        if self.vbs_enable:
            for i in range(len(self.raw)):
                for j in range(len(self.raw[i])):
                    item = self.raw[i][j]
                    vbs = item['vbs']
                    block = item['predictor']
                    if vbs is VBSMarker.SPLIT:
                        text += '{}'.format(exp_golomb_encoding(VBSMarker.SPLIT.value))

                        for predictor in block:
                            diff_mv = predictor - prev_mv
                            prev_mv = predictor
                            text += '{}'.format(diff_mv.to_str(self.is_intraframe, fme_enabled=self.fme_enable))

                    elif vbs is VBSMarker.UNSPLIT:
                        diff_mv = block - prev_mv
                        prev_mv = block
                        text += '{}{}'.format(exp_golomb_encoding(VBSMarker.UNSPLIT.value), diff_mv.to_str(self.is_intraframe, fme_enabled=self.fme_enable))

                    else:
                        raise Exception('Invalid VBS Marker')
        else:
            for i in range(len(self.raw)):
                for j in range(len(self.raw[i])):
                    diff_mv = self.raw[i][j] - prev_mv
                    prev_mv = self.raw[i][j]
                    text += diff_mv.to_str(self.is_intraframe, fme_enabled=self.fme_enable)
        return binstr_to_bytes(text)

    def read_from_file(self, path: pathlib.Path, width: int, params_i: int):
        """
            Read motion vector frame from file

            Parameters:
                path (pathlib.Path): The path to read from.
                width (int): The width of the frame.
                params_i (int): The I parameter.
        """
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
            index_counter = 0
            while index_counter < len(mv):
                if mv_counter == 0:
                    self.new_row()
                if self.vbs_enable:
                    vbs = VBSMarker(int(mv[index_counter]))
                    index_counter += 1
                    if vbs is VBSMarker.SPLIT:
                        current_mv = dict(
                            vbs=VBSMarker.SPLIT,
                            predictor=[]
                        )
                        for _ in range(4):
                            mv_a = prev_mv + MotionVector(mv[index_counter], 0)
                            if mv_a.y > 1:
                                raise Exception('Invalid motion vector')
                            prev_mv = mv_a
                            current_mv['predictor'].append(mv_a)
                            index_counter += 1
                    elif vbs is VBSMarker.UNSPLIT:
                        mv_a = prev_mv + MotionVector(mv[index_counter], 0)
                        current_mv = dict(
                            vbs=VBSMarker.UNSPLIT,
                            predictor=mv_a
                        )
                        prev_mv = mv_a
                        index_counter += 1
                    else:
                        raise Exception('Invalid VBS Marker')
                else:
                    current_mv = prev_mv + MotionVector(mv[index_counter], 0)
                    prev_mv = current_mv
                    index_counter += 1
                self.append(current_mv)
                mv_counter += 1
                if mv_counter == width // params_i:
                    mv_counter = 0
        else:
            index_counter = 0
            while index_counter < len(mv):
                if mv_counter == 0:
                    self.new_row()
                if self.vbs_enable:
                    vbs = VBSMarker(int(mv[index_counter]))
                    index_counter += 1
                    if vbs is VBSMarker.SPLIT:
                        current_mv = dict(
                            vbs=VBSMarker.SPLIT,
                            predictor=[]
                        )
                        for _ in range(4):
                            if self.fme_enable:
                                mv_a = prev_mv + MotionVector(mv[index_counter] / 2, mv[index_counter + 1] / 2, mv[index_counter + 2])
                            else:
                                mv_a = prev_mv + MotionVector(mv[index_counter], mv[index_counter + 1], mv[index_counter + 2])
                            prev_mv = mv_a
                            current_mv['predictor'].append(mv_a)
                            index_counter += 3
                    elif vbs is VBSMarker.UNSPLIT:
                        if self.fme_enable:
                            mv_a = prev_mv + MotionVector(mv[index_counter] / 2, mv[index_counter + 1] / 2, mv[index_counter + 2])
                        else:
                            mv_a = prev_mv + MotionVector(mv[index_counter], mv[index_counter + 1], mv[index_counter + 2])
                        prev_mv = mv_a
                        current_mv = dict(
                            vbs=VBSMarker.UNSPLIT,
                            predictor=mv_a
                        )
                        index_counter += 3
                    else:
                        raise Exception('Invalid VBS Marker')
                else:
                    if self.fme_enable:
                        current_mv = prev_mv + MotionVector(mv[index_counter] / 2, mv[index_counter + 1] / 2, mv[index_counter + 2])
                    else:
                        current_mv = prev_mv + MotionVector(mv[index_counter], mv[index_counter + 1], mv[index_counter + 2])
                    prev_mv = current_mv
                    index_counter += 3
                self.append(current_mv)
                mv_counter += 1
                if mv_counter == width // params_i:
                    mv_counter = 0
