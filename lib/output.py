import numpy as np
from multiprocessing import Queue

from lib.utils.enums import YUVFormat, Identifier
from lib.utils.misc import yuv2rgb, pixel_create
from lib.utils.misc import block_create

"""
    Output the frame to y-only files.
    
    Parameters:
        data (np_array, frame_index) (tuple): Data to process.
        args (params_i, diff_factor) (tuple): Arguments.
        q (mp.Queue): The queue to put data to.
"""
def to_y_only_files(data: tuple, args: tuple, q: Queue) -> None:
    np_array, frame_index = data
    params_i, diff_factor = args
    # padding
    np_array_uint8 = np.array(np_array, dtype=np.uint8)
    q.put((np_array_uint8[:, :, 0], "{}.y-only".format(frame_index)))

    # create block-based frame
    block_frame, offset, block_size, y_only_padded = block_create(np_array[:, :, 0], params_i)

    # average block
    average_block = block_frame.mean(2).round().astype(int).reshape(-1, offset, 1).repeat(block_size, 2)

    # reshape to original form
    y_only_averaged_array = pixel_create(average_block, y_only_padded.shape, params_i)
    
    y_only_averaged_array_uint8 = np.array(y_only_averaged_array, dtype=np.uint8)
    q.put((y_only_averaged_array_uint8, "{}.y-only-averaged".format(frame_index)))

    # difference
    if diff_factor > 0:
        y_only_padded_uint8 = np.array(y_only_padded, dtype=np.uint8)
        q.put((y_only_padded_uint8, "{}.y-only-padded".format(frame_index)))

"""
    Output the frame to video.

    Parameters:
        data (np_array, frame_index) (tuple): Data to process.
        args (upscale) (tuple): Arguments.
        q (mp.Queue): The queue to put data to.
"""
def to_video(data: tuple, args: tuple, q: Queue) -> None:
    np_array, frame_index = data
    upscale, = args

    np_array_uint8 = np.array(np_array, dtype=np.uint8)
    yuv_frame = bytearray()
    yuv_frame.extend(Identifier.FRAME.value)

    yuv_frame.extend(np_array_uint8[:, :, 0].tobytes())
    if upscale == YUVFormat.YUV420:
        yuv_frame.extend(np_array_uint8[::2, ::2, 1].tobytes())
        yuv_frame.extend(np_array_uint8[::2, ::2, 2].tobytes())
    elif upscale == YUVFormat.YUV422:
        yuv_frame.extend(np_array_uint8[:, ::2, 1].tobytes())
        yuv_frame.extend(np_array_uint8[:, ::2, 2].tobytes())
    else:
        yuv_frame.extend(np_array_uint8[:, :, 1].tobytes())
        yuv_frame.extend(np_array_uint8[:, :, 2].tobytes())
    q.put((frame_index, yuv_frame))

"""
    Output the frame to pngs.

    Parameters:
        data (np_array, frame_index) (tuple): Args.
        args (noise_arg) (tuple): Arguments.
        q (mp.Queue): The queue to put data to.
"""
def to_pngs(data: tuple, args: tuple, q: Queue) -> None:
    np_array, frame_index = data
    noise_arg, = args
    
    y = np_array[:, :, 0]
    u = np_array[:, :, 1]
    v = np_array[:, :, 2]

    if noise_arg is not None:
        noise = np.random.normal(0, 50, y.shape)
        if noise_arg == 'y':
            y = y + noise
        elif noise_arg == 'u':
            u = u + noise
        elif noise_arg == 'v':
            v = v + noise

    r, g, b, _ = yuv2rgb(y, u, v)

    if noise_arg is not None:
        if noise_arg == 'r':
            r = r + noise
        elif noise_arg == 'g':
            g = g + noise
        elif noise_arg == 'b':
            b = b + noise

    r = np.clip(r, 0, 255).astype(np.uint8)
    g = np.clip(g, 0, 255).astype(np.uint8)
    b = np.clip(b, 0, 255).astype(np.uint8)
    rgb = np.stack((r, g, b), axis=-1)
    q.put((rgb, frame_index))