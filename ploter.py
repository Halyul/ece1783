#!/usr/bin/env python3
from matplotlib import pyplot as plt
import argparse
import pathlib
import numpy as np
from lib.config import Config
from lib.misc import get_padding, yuv2rgb

CONFIG = Config('config.yaml').config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='File to plot/view')
    parser.add_argument('--v', '--view', dest='view', action='store_true', help='View image')
    parser.add_argument('--h', '--height', dest='height', type=int, default=-1, help='Height of image')
    parser.add_argument('--w', '--width', dest='width', type=int, default=-1,help='Width of image')
    args = parser.parse_args()

    file = pathlib.Path.cwd().joinpath(args.file)

    if not file.exists():
        print("File does not exist")
        exit(1)

    if args.height <= 0 and args.width <= 0:
        print("Height or width must be greater than 0")
        exit(1)

    height = args.height
    width = args.width
    params_i = CONFIG['params']['i']
    pad_width, pad_height = get_padding(width, height, params_i)

    if args.view:
        with open(file, 'rb') as f:
            file_bytes = f.read()
        
        data = np.frombuffer(file_bytes, dtype=np.uint8).reshape(pad_height, pad_width)
        _, _, _, data = yuv2rgb(data)
    else:
        averaged_file = file.parent.joinpath('{}.y-only-averaged'.format(file.stem))
        padded_file = file.parent.joinpath('{}.y-only-padded'.format(file.stem))
        
        with open(averaged_file, 'rb') as f:
            averaged_file_bytes = f.read()

        with open(padded_file, 'rb') as f:
            padded_file_bytes = f.read()

        averaged_data = np.frombuffer(averaged_file_bytes, dtype=np.uint8).reshape(pad_height, pad_width)
        padded_file_data = np.frombuffer(padded_file_bytes, dtype=np.uint8).reshape(pad_height, pad_width)
        
        data = (averaged_data - padded_file_data) * CONFIG['output']['y_only']['diff_factor']
    
    plt.imshow(data, interpolation='nearest')
    plt.show()