#!/usr/bin/env python3

from matplotlib import pyplot as plt
import argparse
import pathlib
import math
import numpy as np
from lib.config import Config

CONFIG = Config('config.yaml').config

def get_padding(width, height, i):
    pad_width = math.ceil(width / i) * i if width > 0 else -1
    pad_height = math.ceil(height / i) * i if height > 0 else -1
    return pad_width, pad_height

def yuv2rgb(y):
    height, width = y.shape
    u = np.array([128] * (height * width)).reshape(height, width)
    v = np.array([128] * (height * width)).reshape(height, width)
    r = 1.164 * (y - 16) + 1.596 * (v - 128)
    g = 1.164 * (y - 16) - 0.813 * (v - 128) - 0.392 * (u - 128)
    b = 1.164 * (y - 16) + 2.017 * (u - 128)
    r = np.clip(r, 0, 255).astype(np.uint8)
    g = np.clip(g, 0, 255).astype(np.uint8)
    b = np.clip(b, 0, 255).astype(np.uint8)
    return np.stack((r, g, b), axis=-1)

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
        data = yuv2rgb(data)
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