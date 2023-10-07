#!/usr/bin/env python3
from matplotlib import pyplot as plt
import pathlib
import numpy as np
import argparse
from lib.signal_processing import psnr, ssim

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file1', type=str, help='File 1')
    parser.add_argument('file2', type=str, help='File 2')
    parser.add_argument('--h', '--height', dest='height', type=int, default=-1, help='Height of image')
    parser.add_argument('--w', '--width', dest='width', type=int, default=-1,help='Width of image')
    args = parser.parse_args()

    file1 = pathlib.Path.cwd().joinpath(args.file1)
    file2 = pathlib.Path.cwd().joinpath(args.file2)

    if not file1.exists():
        print("File 1 does not exist")
        exit(1)
    if not file2.exists():
        print("File 2 does not exist")
        exit(1)

    if args.height <= 0 and args.width <= 0:
        print("Height or width must be greater than 0")
        exit(1)
    
    height = args.height
    width = args.width

    with open(pathlib.Path.cwd().joinpath(file1), 'rb') as f:
        f1 = f.read()
    f1 = np.frombuffer(f1, dtype=np.uint8).reshape(height, width)
    with open(pathlib.Path.cwd().joinpath(file2), 'rb') as f:
        f2 = f.read()
    f2 = np.frombuffer(f2, dtype=np.uint8).reshape(height, width)

    ssim_map = ssim(f1, f2)
    print(ssim_map.mean())
    print(psnr(f1, f2), "dB")
    plt.imshow(ssim_map, interpolation='nearest', cmap='gray')
    plt.show()