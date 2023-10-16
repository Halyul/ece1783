#!/usr/bin/env python3
from matplotlib import pyplot as plt
import argparse
import pathlib
import numpy as np
from lib.utils.config import Config
from lib.utils.misc import get_padding, yuv2rgb, convert_within_range

CONFIG = Config('config.yaml').config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file1', type=str, help='File to plot/view')
    parser.add_argument('--c', '--compare', dest='file2', default='', type=str, help='File to compare with')
    parser.add_argument('--d', '--diff-factor', dest='diff_factor', type=int, default=1,help='Diff factor')
    parser.add_argument('--v', '--view', dest='view', action='store_true', help='View image')
    parser.add_argument('--h', '--height', dest='height', type=int, default=-1, help='Height of image')
    parser.add_argument('--w', '--width', dest='width', type=int, default=-1,help='Width of image')
    parser.add_argument('--r', '--residual', dest='residual', action='store_true' ,help='if is residual file')
    args = parser.parse_args()

    file1 = pathlib.Path.cwd().joinpath(args.file1)
    file2 = pathlib.Path.cwd().joinpath(args.file2)

    if not file1.exists():
        print("File 1 does not exist")
        exit(1)

    if not file2.exists() and not args.view:
        print("File 2 does not exist")
        exit(1)

    if args.height <= 0 and args.width <= 0:
        print("Height or width must be greater than 0")
        exit(1)

    height = args.height
    width = args.width
    params_i = CONFIG['params']['i']
    pad_width, pad_height = get_padding(width, height, params_i)

    if args.view:
        with open(file1, 'rb') as f:
            file_bytes = f.read()
        
        if args.residual:
            data = np.frombuffer(file_bytes, dtype=np.int16).reshape(pad_height, pad_width)
            data = np.abs(data) * args.diff_factor
        else:
            data = np.frombuffer(file_bytes, dtype=np.uint8).reshape(pad_height, pad_width)
    else:
        
        with open(file1, 'rb') as f:
            averaged_file_bytes = f.read()

        with open(file2, 'rb') as f:
            padded_file_bytes = f.read()

        averaged_data = np.frombuffer(averaged_file_bytes, dtype=np.uint8).reshape(pad_height, pad_width)
        padded_file_data = np.frombuffer(padded_file_bytes, dtype=np.uint8).reshape(pad_height, pad_width)
        
        data = (averaged_data - padded_file_data) * args.diff_factor
    _, _, _, data = yuv2rgb(data)
    
    plt.imshow(data, interpolation='nearest')
    plt.show()