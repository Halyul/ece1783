#!/usr/bin/env python3
from yuv_processor import YUVProcessor

if __name__ == '__main__':
    reader = YUVProcessor('config.yaml')
    print(reader.info)