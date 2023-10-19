#!/usr/bin/env python3
from lib.utils.config import Config
from lib.utils.misc import construct_reconstructed_frame, convert_within_range
from lib.utils.quantization import quantization_matrix, frame_qtc_to_tc, residual_coefficients_to_residual_frame
from lib.utils.differential import frame_differential_decoding
from lib.utils.entropy import reording_decoding, rle_decoding, array_exp_golomb_decoding
from lib.utils.enums import TypeMarker
from lib.utils.misc import bytes_to_binstr
import pathlib
import numpy as np
import time

start = time.time()

config_class = Config('config.yaml')
config = config_class.config

mv_path = pathlib.Path.cwd().joinpath(config['decoder']['input_path']['mv_folder'])
residual_path = pathlib.Path.cwd().joinpath(config['decoder']['input_path']['residual_folder'])
meta_file = pathlib.Path.cwd().joinpath(config['decoder']['input_path']['meta_file'])

output_path = pathlib.Path.cwd().joinpath(config['decoder']['output_path']['main_folder'])
if not output_path.exists():
    output_path.mkdir()

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

    if is_intraframe:
        prev_frame = None
    else:
        prev_index = i - 1
        if prev_index == -1:
            prev_frame = np.full(height*width, 128).reshape(height, width)
        else:
            prev_file = output_path.joinpath('{}'.format(prev_index))
            prev_file_bytes = prev_file.read_bytes()
            prev_frame_uint8 = np.frombuffer(prev_file_bytes, dtype=np.uint8).reshape(height, width)
            prev_frame = np.array(prev_frame_uint8, dtype=np.int16)

    qtc_file = residual_path.joinpath('{}'.format(i))
    qtc_file_lines = qtc_file.read_bytes()
    qtc_file_lines = bytes_to_binstr(qtc_file_lines)

    qtc_dump = []
    qtc_counter = 0
    qtc_single_array = array_exp_golomb_decoding(qtc_file_lines)
    qtc_pending = []
    for item in qtc_single_array:
        qtc_pending.append(item)
        if item == 0:
            if qtc_counter == 0:
                qtc_dump.append([])
            qtc_dump[-1].append(reording_decoding(rle_decoding(qtc_pending, q_matrix.shape), q_matrix.shape))
            qtc_pending = []
            qtc_counter += 1
            if qtc_counter == width // params_i:
                qtc_counter = 0

    residual_frame_qtc = np.array(qtc_dump, dtype=np.int16)
    residual_frame_coefficients = frame_qtc_to_tc(residual_frame_qtc, q_matrix)
    residual_frame = residual_coefficients_to_residual_frame(residual_frame_coefficients, params_i, (height, width))
    
    mv_dump = frame_differential_decoding(mv_dump, is_intraframe)
    current_reconstructed_frame = construct_reconstructed_frame(mv_dump, prev_frame, residual_frame, params_i)
    current_reconstructed_frame = convert_within_range(current_reconstructed_frame)

    output_path.joinpath('{}'.format(i)).write_bytes(current_reconstructed_frame)
    print("reconstructed frame {} written".format(i))

end = time.time()
print('Time: {}s'.format(end - start))