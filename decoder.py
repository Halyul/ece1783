#!/usr/bin/env python3
from lib.config.config import Config
from lib.enums import Intraframe, VBSMarker
from lib.components.frame import Frame, extend_block
from lib.components.qtc import QTCFrame, quantization_matrix
from lib.components.mv import MotionVectorFrame
import numpy as np
import time, math

def get_fme_block(top_left, params_i, current_frame, y_counter, x_counter):
    mv_y: float = top_left[0]
    mv_x: float = top_left[1]
    is_y_interpolate = not mv_y.is_integer()
    is_x_interpolate = not mv_x.is_integer()
    if is_y_interpolate or is_x_interpolate:
        if is_y_interpolate and not is_x_interpolate:
            top_y = math.floor(mv_y)
            bottom_y = math.ceil(mv_y)
            mv_x = int(mv_x)
            top_block = current_frame.raw[y_counter + top_y:y_counter + top_y + params_i, x_counter + mv_x:x_counter + mv_x + params_i].astype(float)
            bottom_block = current_frame.raw[y_counter + bottom_y:y_counter + bottom_y + params_i, x_counter + mv_x:x_counter + mv_x + params_i].astype(float)
            y_middle_block = (top_block + bottom_block) / 2
            return y_middle_block.astype(np.uint8)
        elif is_x_interpolate and not is_y_interpolate:
            left_x = math.floor(mv_x)
            right_x = math.ceil(mv_x)
            mv_y = int(mv_y)
            left_block = current_frame.raw[y_counter + mv_y:y_counter + mv_y + params_i, x_counter + left_x:x_counter + left_x + params_i].astype(float)
            right_block = current_frame.raw[y_counter + mv_y:y_counter + mv_y + params_i, x_counter + right_x:x_counter + right_x + params_i].astype(float)
            x_middle_block = (left_block + right_block) / 2
            return x_middle_block.astype(np.uint8)
        else:
            top_y = math.floor(mv_y)
            bottom_y = math.ceil(mv_y)
            left_x = math.floor(mv_x)
            right_x = math.ceil(mv_x)
            top_left_block = current_frame.raw[y_counter + top_y:y_counter + top_y + params_i, x_counter + left_x:x_counter + left_x + params_i].astype(float)
            top_right_block = current_frame.raw[y_counter + top_y:y_counter + top_y + params_i, x_counter + right_x:x_counter + right_x + params_i].astype(float)
            bottom_left_block = current_frame.raw[y_counter + bottom_y:y_counter + bottom_y + params_i, x_counter + left_x:x_counter + left_x + params_i].astype(float)
            bottom_right_block = current_frame.raw[y_counter + bottom_y:y_counter + bottom_y + params_i, x_counter + right_x:x_counter + right_x + params_i].astype(float)
            y_middle_left_block = (top_left_block + bottom_left_block) / 2
            y_middle_right_block = (top_right_block + bottom_right_block) / 2
            y_middle_block = ((y_middle_left_block + y_middle_right_block) / 2)
            x_middle_top_block = (top_left_block + top_right_block) / 2
            x_middle_bottom_block = (bottom_left_block + bottom_right_block) / 2
            x_middle_block = ((x_middle_top_block + x_middle_bottom_block) / 2)
            return ((y_middle_block + x_middle_block) / 2).astype(np.uint8)
    else:
        mv_y = int(mv_y)
        mv_x = int(mv_x)
        return current_frame.raw[y_counter + mv_y:y_counter + mv_y + params_i, x_counter + mv_x:x_counter + mv_x + params_i]

"""
    Construct the predicted frame from the motion vector dump.

    Parameters:
        mv_dump (list): The motion vector dump.
        prev_frame (np.ndarray): The previous frame. If None, then the frame is intraframe.
        residual_frame (np.ndarray): The residual frame.
        params_i (int): The block size.
    
    Returns:
        (np.ndarray): The reconstructed frame.
"""
def construct_reconstructed_frame(mv_dump, frame, residual_frame, vbs_enable=False, fme_enable=False) -> np.ndarray:
    y_counter = 0
    x_counter = 0
    if frame.is_intraframe:
        # is intraframe
        height, width = residual_frame.shape
        reconstructed_block_dump = Frame(frame.index, height, width, frame.params_i, frame.is_intraframe, data=np.empty(residual_frame.shape, dtype=int))
        for y in range(len(mv_dump.raw)):
            for x in range(len(mv_dump.raw[y])):
                current_coor = (y_counter, x_counter)
                local_vbs_enable = vbs_enable and mv_dump.raw[y][x]['vbs'] is VBSMarker.SPLIT
                if local_vbs_enable:
                    subblock_params_i = frame.params_i // 2
                    top_lefts = [(_y + y_counter, _x + x_counter) for _y in range(0, frame.params_i, subblock_params_i) for _x in range(0, frame.params_i, subblock_params_i)]
                    for centered_top_left_index in range(len(top_lefts)):
                        centered_top_left = top_lefts[centered_top_left_index]
                        predictor = mv_dump.raw[y][x]['predictor'][centered_top_left_index].y
                        if centered_top_left[1] == 0 and predictor == Intraframe.HORIZONTAL.value:
                            # first column of blocks in horizontal
                            predictor_block = np.full((subblock_params_i, 1), 128)
                            repeat_value = Intraframe.HORIZONTAL.value
                        elif centered_top_left[0] == 0 and predictor == Intraframe.VERTICAL.value:
                            # first row of blocks in vertical
                            predictor_block = np.full((1, subblock_params_i), 128)
                            repeat_value = Intraframe.VERTICAL.value
                        elif predictor == Intraframe.HORIZONTAL.value:
                            # horizontal
                            hor_top_left, _ = extend_block(centered_top_left, subblock_params_i, (0, 0, 0, 1), (height, width))
                            predictor_block = reconstructed_block_dump.raw[hor_top_left[0]:hor_top_left[0] + subblock_params_i, hor_top_left[1]:hor_top_left[1] + 1]
                            repeat_value = Intraframe.HORIZONTAL.value
                        elif predictor == Intraframe.VERTICAL.value:
                            # vertical
                            ver_top_left, _ = extend_block(centered_top_left, subblock_params_i, (1, 0, 0, 0), (height, width))
                            predictor_block = reconstructed_block_dump.raw[ver_top_left[0]:ver_top_left[0] + 1, ver_top_left[1]:ver_top_left[1] + subblock_params_i]
                            repeat_value = Intraframe.VERTICAL.value
                        else:
                            raise Exception('Invalid predictor.')
                        predictor_block = predictor_block.repeat(subblock_params_i, repeat_value)
                        residual_block = residual_frame.raw[centered_top_left[0]:centered_top_left[0] + subblock_params_i, centered_top_left[1]:centered_top_left[1] + subblock_params_i]
                        reconstructed_block_dump.raw[centered_top_left[0]:centered_top_left[0] + subblock_params_i, centered_top_left[1]:centered_top_left[1] + subblock_params_i] = predictor_block + residual_block
                    x_counter += params_i
                else:
                    predictor = mv_dump.raw[y][x]['predictor'].y if vbs_enable else mv_dump.raw[y][x].y
                    if x == 0 and predictor == Intraframe.HORIZONTAL.value:
                        # first column of blocks in horizontal
                        predictor_block = np.full((params_i, 1), 128)
                        repeat_value = Intraframe.HORIZONTAL.value
                    elif y == 0 and predictor == Intraframe.VERTICAL.value:
                        # first row of blocks in vertical
                        predictor_block = np.full((1, params_i), 128)
                        repeat_value = Intraframe.VERTICAL.value
                    elif predictor == Intraframe.HORIZONTAL.value:
                        # horizontal
                        hor_top_left, _ = extend_block(current_coor, params_i, (0, 0, 0, 1), (height, width))
                        predictor_block = reconstructed_block_dump.raw[hor_top_left[0]:hor_top_left[0] + params_i, hor_top_left[1]:hor_top_left[1] + 1]
                        repeat_value = Intraframe.HORIZONTAL.value
                    elif predictor == Intraframe.VERTICAL.value:
                        # vertical
                        ver_top_left, _ = extend_block(current_coor, params_i, (1, 0, 0, 0), (height, width))
                        predictor_block = reconstructed_block_dump.raw[ver_top_left[0]:ver_top_left[0] + 1, ver_top_left[1]:ver_top_left[1] + params_i]
                        repeat_value = Intraframe.VERTICAL.value
                    else:
                        raise Exception('Invalid predictor.')
                    predictor_block = predictor_block.repeat(params_i, repeat_value)
                    residual_block = residual_frame.raw[y_counter:y_counter + params_i, x_counter:x_counter + params_i]
                    reconstructed_block_dump.raw[y_counter:y_counter + params_i, x_counter:x_counter + params_i] = predictor_block + residual_block
                    x_counter += params_i
            y_counter += params_i
            x_counter = 0
    else:
        predicted_frame_dump = []
        for i in range(len(mv_dump.raw)):
            predicted_frame_dump.append([])
            for j in range(len(mv_dump.raw[i])):
                if j == 21 and i == 1:
                    print('')
                current_frame = frame.prev
                if vbs_enable:
                    item = mv_dump.raw[i][j]
                    vbs = item['vbs']
                    block = item['predictor']
                    if vbs is VBSMarker.SPLIT:
                        subblock_params_i = frame.params_i // 2
                        top_lefts = [(_y + y_counter, _x + x_counter) for _y in range(0, frame.params_i, subblock_params_i) for _x in range(0, frame.params_i, subblock_params_i)]
                        frame_data = []
                        for top_left_index in range(len(top_lefts)):
                            top_left_coor = top_lefts[top_left_index]
                            top_left = block[top_left_index].raw
                            local_current_frame = current_frame
                            for _ in range(block[top_left_index].ref_offset):
                                local_current_frame = local_current_frame.prev
                            if fme_enable:
                                d = get_fme_block(top_left, subblock_params_i, local_current_frame, top_left_coor[0], top_left_coor[1])
                                if d.shape != (subblock_params_i, subblock_params_i):
                                    raise Exception('Invalid FME block shape.')
                                frame_data.append(d)
                            else:
                                frame_data.append(local_current_frame.raw[top_left_coor[0] + top_left[0]:top_left_coor[0] + top_left[0] + subblock_params_i, top_left_coor[1] + top_left[1]:top_left_coor[1] + top_left[1] + subblock_params_i])
                        frame_stack = np.concatenate((np.concatenate((frame_data[0], frame_data[1]), axis=1), np.concatenate((frame_data[2], frame_data[3]), axis=1)), axis=0)
                        predicted_frame_dump[i].append(frame_stack)
                    elif vbs is VBSMarker.UNSPLIT:
                        top_left = block.raw
                        for _ in range(block.ref_offset):
                            current_frame = current_frame.prev
                        if fme_enable:
                            d = get_fme_block(top_left, params_i, current_frame, y_counter, x_counter)
                            if d.shape != (params_i, params_i):
                                raise Exception('Invalid FME block shape. y={}, x={}'.format(y_counter, x_counter))
                            predicted_frame_dump[i].append(get_fme_block(top_left, params_i, current_frame, y_counter, x_counter))
                        else:
                            predicted_frame_dump[i].append(current_frame.raw[y_counter + top_left[0]:y_counter + top_left[0] + params_i, x_counter + top_left[1]:x_counter + top_left[1] + params_i])
                else:
                    top_left = mv_dump.raw[i][j].raw
                    for _ in range(mv_dump.raw[i][j].ref_offset):
                        current_frame = current_frame.prev
                    if fme_enable:
                        predicted_frame_dump[i].append(get_fme_block(top_left, params_i, current_frame, y_counter, x_counter))
                    else:
                        predicted_frame_dump[i].append(current_frame.raw[y_counter + top_left[0]:y_counter + top_left[0] + params_i, x_counter + top_left[1]:x_counter + top_left[1] + params_i])
                x_counter += params_i
            y_counter += params_i
            x_counter = 0
        frame.block_to_pixel(np.array(predicted_frame_dump))
        frame += residual_frame
        reconstructed_block_dump = frame

    return reconstructed_block_dump

if __name__ == '__main__':
    start = time.time()

    config = Config('config.yaml')

    mv_path = config.decoder.input_path.mv_folder
    residual_path = config.decoder.input_path.residual_folder
    meta_file = config.decoder.input_path.meta_file

    output_path = config.decoder.output_path.main_folder

    l = meta_file.read_text().split(',')
    total_frames = int(l[0])
    height, width = int(l[1]), int(l[2])
    params_i = int(l[3])
    params_qp = int(l[4])
    nRefFrames = int(l[5])
    VBSEnabled = bool(int(l[6]))
    FMEEnabled = bool(int(l[7]))
    q_matrix = quantization_matrix(params_i, params_qp)
    read_frame_counter = 0
    prev_frame = None
    while read_frame_counter < total_frames:
        mv_file = mv_path.joinpath('{}'.format(read_frame_counter))
        mv_dump = MotionVectorFrame(vbs_enable=VBSEnabled, fme_enable=FMEEnabled)
        mv_dump.read_from_file(mv_file, width, params_i)

        frame = Frame(read_frame_counter, height, width, params_i, mv_dump.is_intraframe)
        if not mv_dump.is_intraframe:
            frame.prev = prev_frame

        qtc_file = residual_path.joinpath('{}'.format(read_frame_counter))
        qtc_frame = QTCFrame(params_i=params_i, vbs_enable=VBSEnabled)
        qtc_frame.read_from_file(qtc_file, q_matrix, width, params_qp)
        residual_frame = qtc_frame.to_residual_frame()

        if read_frame_counter == 3:
            print('')

        frame = construct_reconstructed_frame(mv_dump, frame, residual_frame, vbs_enable=VBSEnabled, fme_enable=FMEEnabled)
        frame.convert_within_range()
        frame.dump(output_path.joinpath('{}'.format(read_frame_counter)))

        frame.prev = prev_frame
        prev_frame = frame
        if not frame.is_intraframe:
            prev_pointer = prev_frame
            prev_counter = 0
            while prev_pointer.prev is not None:
                if prev_counter == nRefFrames - 1:
                    prev_pointer.prev = None
                    break
                else:
                    prev_counter += 1
                prev_pointer = prev_pointer.prev


        print("reconstructed frame {} written".format(read_frame_counter))
        read_frame_counter += 1

    end = time.time()
    print('Time: {}s'.format(end - start))
