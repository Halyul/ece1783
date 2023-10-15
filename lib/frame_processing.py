from lib.utils.enums import YUVFormat
import numpy as np
from multiprocessing import Queue
from typing import Callable as function

"""
    Upscale the frame to YUV444.

    Parameters:
        data (tuple): The data to upscale.
        q (Queue): The queue to write to.
"""
def upscale(data: tuple, q: Queue):
    (width, frame_index, format_tuple, yuv_components, mode) = data

    pixel_list = []

    for i in range(format_tuple[0]):
        y_component = yuv_components[i]
        current_row = i // width
        current_col = i % width
        if mode == YUVFormat.YUV420:
            # YUV420
            u_component = yuv_components[format_tuple[0] + current_row // 2 * width // 2 + current_col // 2]
            v_component = yuv_components[format_tuple[0] + format_tuple[1] + current_row // 2 * width // 2 + current_col // 2]
        elif mode == YUVFormat.YUV422:
            # YUV422
            u_component = yuv_components[format_tuple[0] + i // 2]
            v_component = yuv_components[format_tuple[0] + format_tuple[1] + i // 2]
        else:
            # YUV444
            u_component = yuv_components[format_tuple[0] + i]
            v_component = yuv_components[format_tuple[0] + format_tuple[1] + i]
            break
        if i % width == 0:
            pixel_list.append([])
        pixel_list[current_row].append([y_component, u_component, v_component])

    q.put((np.array(pixel_list), frame_index))
