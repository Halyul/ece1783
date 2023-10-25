from lib.utils.enums import YUVFormat
import numpy as np
from lib.utils.misc import get_padding, convert_within_range
from lib.config.config import Config

"""
    Upscale the frame to YUV444.

    Parameters:
        data (tuple): The data to upscale.
        config (Config): The config object.
"""
def upscale(data: tuple, config: Config) -> None:
    (width, frame_index, format_tuple, yuv_components, mode) = data
    print("Upscaling", frame_index)

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
    np_pixel_array = np.array(pixel_list)

    np_y_array = np_pixel_array[:, :, 0]
    width = np_pixel_array.shape[1]
    height = np_pixel_array.shape[0]
    paded_width, paded_height = get_padding(width, height, config.params.i)
    pad_width = paded_width - width
    pad_height = paded_height - height
    np_y_array_padded = np.pad(np_y_array, ((0, pad_height), (0, pad_width)), 'constant', constant_values=128)

    # simply write to a file, so the execution order is guaranteed?
    # q.put((np.array(pixel_list), frame_index))
    np_y_array_padded = convert_within_range(np_y_array_padded)
    config.output_path.original_folder.joinpath('{}'.format(frame_index)).write_bytes(np_y_array_padded)

    np_uv_array = convert_within_range(np.stack((np_pixel_array[:, :, 1], np_pixel_array[:, :, 2]), axis=2))
    config.output_path.uv_folder.joinpath('{}'.format(frame_index)).write_bytes(np_uv_array)