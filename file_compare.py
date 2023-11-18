#!/usr/bin/env python3
from lib.config.config import Config
import numpy as np

config = Config('config.yaml')

decoder_output_path = config.decoder.output_path.main_folder
reconstructed_output_path = config.output_path.reconstructed_folder

flag = True

for reconstructed_file in reconstructed_output_path.iterdir():
    reconstructed_file_bytes = reconstructed_file.read_bytes()
    reconstructed_file_array = np.frombuffer(reconstructed_file_bytes, dtype=np.uint8)
    decoder_file = decoder_output_path.joinpath(reconstructed_file.name)
    decoder_file_bytes = decoder_file.read_bytes()
    decoder_file_array = np.frombuffer(decoder_file_bytes, dtype=np.uint8)
    for i in range(len(reconstructed_file_array)):
        if reconstructed_file_array[i] != decoder_file_array[i]:
            print("Error in file: {}; index: {}".format(reconstructed_file.name, i))
            print(reconstructed_file_array[i], decoder_file_array[i])
            flag = False

if flag:
    print("All files equal")