#!/usr/bin/env python3
from lib.yuv_processor import YUVProcessor
import time

if __name__ == '__main__':
    start = time.time()
    reader = YUVProcessor('config.yaml', 
                        config_override=dict(
                            # params=dict(
                            #     stop_at=5,
                            #     i=64,
                            # )
                        )
                        )
    end = time.time()
    print(reader.info)
    print('Time: {}s'.format(end - start))