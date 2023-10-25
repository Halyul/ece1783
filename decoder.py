#!/usr/bin/env python3
from lib.config.config import Config
from lib.utils.misc import extend_block
from lib.utils.enums import Intraframe
from lib.utils.quantization import quantization_matrix
from lib.utils.differential import frame_differential_decoding
from lib.utils.entropy import array_exp_golomb_decoding
from lib.utils.enums import TypeMarker
from lib.utils.misc import bytes_to_binstr
from lib.components.frame import Frame
from lib.components.qtc import QTCFrame
import numpy as np
import time

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
def construct_reconstructed_frame(mv_dump: list, frame: np.ndarray, residual_frame: np.ndarray) -> np.ndarray:
    y_counter = 0
    x_counter = 0
    if frame.is_intraframe:
        # is intraframe
        height, width = residual_frame.shape
        reconstructed_block_dump = Frame(frame.index, height, width, frame.params_i, frame.is_intraframe, data=np.empty(residual_frame.shape, dtype=int))
        for y in range(len(mv_dump)):
            for x in range(len(mv_dump[y])):
                current_coor = (y_counter, x_counter)
                predictor = mv_dump[y][x]
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
        for i in range(len(mv_dump)):
            predicted_frame_dump.append([])
            for j in range(len(mv_dump[i])):
                top_left = mv_dump[i][j]
                predicted_frame_dump[i].append(frame.prev.raw[y_counter + top_left[0]:y_counter + top_left[0] + params_i, x_counter + top_left[1]:x_counter + top_left[1] + params_i])
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
    q_matrix = quantization_matrix(params_i, params_qp)

    for i in range(total_frames):
        mv_file = mv_path.joinpath('{}'.format(i))
        mv_file_lines = mv_file.read_bytes()
        mv_file_lines = bytes_to_binstr(mv_file_lines)
        mv_dump = []
        mv_counter = 0
        type_marker = int(mv_file_lines[0])
        mv_file_lines = mv_file_lines[1:]
        if type_marker == TypeMarker.I_FRAME.value:
            is_intraframe = True
        else:
            is_intraframe = False
        mv_single_array = array_exp_golomb_decoding(mv_file_lines)
        if is_intraframe:
            for item in mv_single_array:
                if mv_counter == 0:
                    mv_dump.append([])
                mv_dump[-1].append(item)
                mv_counter += 1
                if mv_counter == width // params_i:
                    mv_counter = 0
        else:
            for j in range(0, len(mv_single_array), 2):
                min_motion_vector = (mv_single_array[j], mv_single_array[j+1])
                if mv_counter == 0:
                    mv_dump.append([])
                mv_dump[-1].append(min_motion_vector)
                mv_counter += 1
                if mv_counter == width // params_i:
                    mv_counter = 0

        frame = Frame(i, height, width, params_i, is_intraframe)
        if not is_intraframe:
            prev_index = i - 1
            frame.read_prev_from_file(output_path.joinpath('{}'.format(prev_index)), prev_index)
            frame.prev.convert_type(np.int16)

        qtc_file = residual_path.joinpath('{}'.format(i))
        qtc_frame = QTCFrame()
        qtc_frame.read_from_file(qtc_file, q_matrix, width, params_i)
        residual_frame = qtc_frame.to_residual_frame(height, width, params_i)
        
        mv_dump = frame_differential_decoding(mv_dump, is_intraframe)
        frame = construct_reconstructed_frame(mv_dump, frame, residual_frame)
        frame.convert_within_range()
        frame.dump(output_path.joinpath('{}'.format(i)))

        print("reconstructed frame {} written".format(i))

    end = time.time()
    print('Time: {}s'.format(end - start))