#!/usr/bin/env python3
from lib.yuv_processor import YUVProcessor
import time

if __name__ == '__main__':
    start = time.time()
    reader = YUVProcessor('config.yaml')
    end = time.time()
    print(reader.info)
    print('Time: {}s'.format(end - start))