#!/usr/bin/env python3

from matplotlib import pyplot as plt
import argparse
import pathlib
import math
import numpy as np

def get_padding(width, height, i):
        pad_width = math.ceil(width / i) * i if width > 0 else -1
        pad_height = math.ceil(height / i) * i if height > 0 else -1
        return pad_width, pad_height

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='File to plot')
    parser.add_argument('--i', dest='i', type=int, default=-1, help='Block size')
    parser.add_argument('--h', '--height', dest='height', type=int, default=-1, help='Height of image')
    parser.add_argument('--w', '--width', dest='width', type=int, default=-1,help='width of image')
    args = parser.parse_args()

    file = pathlib.Path.cwd().joinpath(args.file)

    if not file.exists():
        print("File does not exist")
        exit(1)

    if args.height <= 0 and args.width <= 0:
        print("Height or width must be greater than 0")
        exit(1)
    
    if args.i <= 0:
        print("Block size must be greater than 0")
        exit(1)
    
    height = args.height
    width = args.width
    pad_width, pad_height = get_padding(width, height, args.i)
    
    with open(file, 'rb') as f:
        raw_bytes = f.read()
    
    data = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(pad_height, pad_width)
    
    plt.imshow(data, interpolation='nearest')
    plt.show()