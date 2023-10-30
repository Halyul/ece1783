#!/usr/bin/env python3
from lib.config.config import Config
from lib.enums import Intraframe
from lib.components.frame import Frame, extend_block
from lib.components.qtc import QTCFrame, quantization_matrix
from lib.components.mv import MotionVectorFrame
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
def construct_reconstructed_frame(mv_dump, frame, residual_frame) -> np.ndarray:
    y_counter = 0
    x_counter = 0
    if frame.is_intraframe:
        # is intraframe
        height, width = residual_frame.shape
        reconstructed_block_dump = Frame(frame.index, height, width, frame.params_i, frame.is_intraframe, data=np.empty(residual_frame.shape, dtype=int))
        for y in range(len(mv_dump.raw)):
            for x in range(len(mv_dump.raw[y])):
                current_coor = (y_counter, x_counter)
                predictor = mv_dump.raw[y][x].y
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
                top_left = mv_dump.raw[i][j].raw
                current_frame = frame.prev
                for _ in range(mv_dump.raw[i][j].ref_offset):
                    current_frame = current_frame.prev
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
    q_matrix = quantization_matrix(params_i, params_qp)
    read_frame_counter = 0
    prev_frame = None
    while read_frame_counter < total_frames:
        mv_file = mv_path.joinpath('{}'.format(read_frame_counter))
        mv_dump = MotionVectorFrame()
        mv_dump.read_from_file(mv_file, width, params_i)

        frame = Frame(read_frame_counter, height, width, params_i, mv_dump.is_intraframe)
        if not mv_dump.is_intraframe:
            frame.prev = prev_frame

        qtc_file = residual_path.joinpath('{}'.format(read_frame_counter))
        qtc_frame = QTCFrame(params_i=params_i)
        qtc_frame.read_from_file(qtc_file, q_matrix, width)
        residual_frame = qtc_frame.to_residual_frame()

        frame = construct_reconstructed_frame(mv_dump, frame, residual_frame)
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
