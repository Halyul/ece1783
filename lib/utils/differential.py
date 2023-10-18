from lib.utils.enums import Intraframe

"""
    Differential encoding.

    Parameters:
        current_mv (tuple): The current motion vector.
        prev_mv (tuple): The previous motion vector.
        is_intraframe (bool): Whether the current frame is an intraframe.

    Returns:
        (tuple) or int: The differential encoded motion vector.
"""
def differential_encoding(current_mv: tuple, prev_mv: tuple, is_intraframe: bool):
    if is_intraframe:
        return current_mv - prev_mv
    else:
        return (current_mv[0] - prev_mv[0], current_mv[1] - prev_mv[1])

"""
    Differential decoding.

    Parameters:
        diff_mv (tuple): The differential encoded motion vector.
        prev_mv (tuple): The previous motion vector.
        is_intraframe (bool): Whether the current frame is an intraframe.
    
    Returns:
        (tuple) or int: The differential decoded motion vector.
"""
def differential_decoding(diff_mv: tuple, prev_mv: tuple, is_intraframe: bool):
    if is_intraframe:
        return diff_mv + prev_mv
    else:
        return (diff_mv[0] + prev_mv[0], diff_mv[1] + prev_mv[1])

"""
    Process full frame motion vectors to differential encoded motion vectors.

    Parameters:
        mv_dump (list): The motion vectors.
        is_intraframe (bool): Whether the current frame is an intraframe.

    Returns:
        (list): The differential encoded motion vectors.
"""
def frame_differential_encoding(mv_dump: list, is_intraframe: bool) -> list:
    if is_intraframe:
        prev_mv = Intraframe.HORIZONTAL.value
    else:
        prev_mv = (0, 0)
    encoded_mv_dump = []

    for i in range(len(mv_dump)):
        encoded_mv_dump.append([])
        for j in range(len(mv_dump[i])):
            diff_mv = differential_encoding(mv_dump[i][j], prev_mv, is_intraframe)
            encoded_mv_dump[-1].append(diff_mv)
            prev_mv = mv_dump[i][j]
    return encoded_mv_dump

"""
    Process full frame differential encoded motion vectors to motion vectors.

    Parameters:
        mv_dump (list): The differential encoded motion vectors.
        is_intraframe (bool): Whether the current frame is an intraframe.

    Returns:
        (list): The motion vectors.
"""
def frame_differential_decoding(mv_dump: list, is_intraframe: bool) -> list:
    if is_intraframe:
        prev_mv = Intraframe.HORIZONTAL.value
    else:
        prev_mv = (0, 0)
    decoded_mv_dump = []

    for i in range(len(mv_dump)):
        decoded_mv_dump.append([])
        for j in range(len(mv_dump[i])):
            diff_mv = differential_decoding(mv_dump[i][j], prev_mv, is_intraframe)
            decoded_mv_dump[-1].append(diff_mv)
            prev_mv = diff_mv
    return decoded_mv_dump
