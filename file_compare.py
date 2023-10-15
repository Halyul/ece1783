from lib.utils.config import Config
import pathlib
import numpy as np

config_class = Config('config.yaml')
config = config_class.config

decoder_output_path = pathlib.Path.cwd().joinpath(config['decoder']['output_path']['main_folder'])
reconstructed_output_path = pathlib.Path.cwd().joinpath(config['output_path']['main_folder'], config['output_path']['reconstructed_folder'])

flag = True

for reconstructed_file in reconstructed_output_path.iterdir():
    reconstructed_file_bytes = reconstructed_file.read_bytes()
    reconstructed_file_array = np.frombuffer(reconstructed_file_bytes, dtype=np.uint8)
    decoder_file = decoder_output_path.joinpath(reconstructed_file.name)
    decoder_file_bytes = decoder_file.read_bytes()
    decoder_file_array = np.frombuffer(decoder_file_bytes, dtype=np.uint8)
    if not np.array_equal(reconstructed_file_array, decoder_file_array):
        print("file {} not equal".format(reconstructed_file.name))
        flag = False

if flag:
    print("All files equal")