#!/usr/bin/env python3
from lib.config.config import Config
from lib.utils.misc import convert_within_range, construct_predicted_frame
import numpy as np
import time

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

for i in range(total_frames):
    prev_index = i - 1
    if prev_index == -1:
        prev_frame = np.full(height*width, 128).reshape(height, width)
    else:
        prev_file = output_path.joinpath('{}'.format(prev_index))
        prev_file_bytes = prev_file.read_bytes()
        prev_frame_uint8 = np.frombuffer(prev_file_bytes, dtype=np.uint8).reshape(height, width)
        prev_frame = np.array(prev_frame_uint8, dtype=np.int16)
    residual_file = residual_path.joinpath('{}'.format(i))
    residual_file_bytes = residual_file.read_bytes()
    residual_frame = np.frombuffer(residual_file_bytes, dtype=np.int16).reshape(height, width)
    
    mv_file = mv_path.joinpath('{}'.format(i))
    mv_file_lines = mv_file.read_text().split('\n')
    mv_dump = []
    mv_counter = 0
    for line in mv_file_lines:
        if line == '':
            continue
        min_motion_vector_y, min_motion_vector_x = line.split(' ')
        min_motion_vector = (int(min_motion_vector_y), int(min_motion_vector_x))
        if mv_counter == 0:
            mv_dump.append([])
        mv_dump[-1].append(min_motion_vector)
        mv_counter += 1
        if mv_counter == width // params_i:
            mv_counter = 0
    
    current_reconstructed_frame = construct_predicted_frame(mv_dump, prev_frame, params_i) + residual_frame
    current_reconstructed_frame = convert_within_range(current_reconstructed_frame)

    output_path.joinpath('{}'.format(i)).write_bytes(current_reconstructed_frame)
    print("reconstructed frame {} written".format(i))

end = time.time()
print("Time taken: {}s".format(end - start))