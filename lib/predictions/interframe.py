import math
import numpy as np
from lib.config.config import Params
from lib.components.frame import Frame, extend_block, convert_within_range
from lib.components.qtc import QTCBlock, quantization_matrix
from lib.components.mv import MotionVector
from lib.enums import VBSMarker
from lib.predictions.misc import rdo

def interframe_vbs(original_block_coor: tuple, original_block: np.ndarray, original_search_windows: list, original_search_window_coor, reconstructed_block: np.ndarray, qtc_block: QTCBlock, diff_mv: MotionVector, prev_motion_vector: MotionVector, params: Params, qp_rc_vbs = None):
    if params.RCflag != 0:
        block_rdo_cost = rdo(original_block, reconstructed_block, qtc_block, diff_mv, qp_rc_vbs, is_intraframe=False)
    else:
        block_rdo_cost = rdo(original_block, reconstructed_block, qtc_block, diff_mv, params.qp, is_intraframe=False)
    subblock_params_i = params.i // 2
    if params.RCflag != 0:
        qp = qp_rc_vbs - 1 if qp_rc_vbs > 0 else 0
        q_matrix = quantization_matrix(subblock_params_i, qp)
    else:
        qp = params.qp - 1 if params.qp > 0 else 0
        q_matrix = quantization_matrix(subblock_params_i, qp)
    top_lefts = [(y, x) for y in range(0, original_block.shape[0], subblock_params_i) for x in range(0, original_block.shape[1], subblock_params_i)]
    top_lefts_in_search_window = [(y + original_block_coor[0], x + original_block_coor[1]) for y, x in top_lefts]
    subblock_rdo_cost = 0
    qtc_subblocks = []
    reconstructed_subblocks = []
    mv_subblocks = []
    residual_subblocks = []
    for centered_top_left_index in range(len(top_lefts)):
        centered_top_left = top_lefts[centered_top_left_index]
        centered_subblock = original_block[centered_top_left[0]:centered_top_left[0] + subblock_params_i, centered_top_left[1]:centered_top_left[1] + subblock_params_i]
        top_left_in_search_window = top_lefts_in_search_window[centered_top_left_index]

        if params.FastME:
            min_motion_vector, min_block = calc_fast_motion_vector(centered_subblock, top_left_in_search_window, original_search_windows, (original_search_window_coor, (original_search_window_coor[0] + original_search_windows[0].shape[0], original_search_window_coor[1] + original_search_windows[0].shape[1])), subblock_params_i, params, prev_motion_vector)
        else:
            min_motion_vector, min_block = calc_full_range_motion_vector(centered_subblock, top_left_in_search_window, original_search_windows, original_search_window_coor, subblock_params_i, params.FMEEnable)

        qtc_subblock = QTCBlock(block=centered_subblock - min_block, q_matrix=q_matrix, qp = qp)
        qtc_subblock.block_to_qtc()
        residual_subblocks.append(qtc_subblock.block)
        reconstructed_subblock = qtc_subblock.block + min_block
        diff_submv = min_motion_vector - prev_motion_vector if prev_motion_vector is not None else min_motion_vector
        prev_motion_vector = min_motion_vector
        subblock_rdo_cost += rdo(centered_subblock, reconstructed_subblock, qtc_subblock, diff_submv, qp, is_intraframe=False)
        qtc_subblocks.append(qtc_subblock.qtc_block)
        reconstructed_subblocks.append(reconstructed_subblock)
        mv_subblocks.append(min_motion_vector)

    if subblock_rdo_cost < block_rdo_cost:
        qtc_stack = np.concatenate((np.concatenate((qtc_subblocks[0], qtc_subblocks[1]), axis=1), np.concatenate((qtc_subblocks[2], qtc_subblocks[3]), axis=1)), axis=0)
        temp_stack = np.concatenate((np.concatenate((residual_subblocks[0], residual_subblocks[1]), axis=1), np.concatenate((residual_subblocks[2], residual_subblocks[3]), axis=1)), axis=0)
        reconstructed_stack = np.concatenate((np.concatenate((reconstructed_subblocks[0], reconstructed_subblocks[1]), axis=1), np.concatenate((reconstructed_subblocks[2], reconstructed_subblocks[3]), axis=1)), axis=0)
        qtc_block = QTCBlock(qtc_block=qtc_stack, block=temp_stack, qp = qp)
        return qtc_block, reconstructed_stack, mv_subblocks
    else:
        return qtc_block, reconstructed_block, None

def get_interpolated_block(current_block, current_coor, location, og_search_window, og_search_window_top_left, params_i):
    # location = [0, 1, 2, 3] #top, bottom, left, right
    current_block = current_block.astype(float)
    top_left = current_coor
    max_size = og_search_window.shape
    search_window = None
    is_y_interpolate = not float(current_coor[0]).is_integer()
    is_x_interpolate = not float(current_coor[1]).is_integer()
    if is_y_interpolate or is_x_interpolate:
        if is_y_interpolate and not is_x_interpolate:
            # y is interpolated, top and bottom blocks exist
            # x is not interpolated, left and right blocks may exist
            top_block_top_left = (math.floor(top_left[0]), int(top_left[1]))
            bottom_block_top_left = (math.ceil(top_left[0]), int(top_left[1]))
            top_block_top_left_relative_coor = (top_block_top_left[0] - og_search_window_top_left[0], top_block_top_left[1] - og_search_window_top_left[1])
            bottom_block_top_left_relative_coor = (bottom_block_top_left[0] - og_search_window_top_left[0], bottom_block_top_left[1] - og_search_window_top_left[1])
            top_block = og_search_window[top_block_top_left_relative_coor[0]:top_block_top_left_relative_coor[0] + params_i, top_block_top_left_relative_coor[1]:top_block_top_left_relative_coor[1] + params_i].astype(float)
            bottom_block = og_search_window[bottom_block_top_left_relative_coor[0]:bottom_block_top_left_relative_coor[0] + params_i, bottom_block_top_left_relative_coor[1]:bottom_block_top_left_relative_coor[1] + params_i].astype(float)
            if location == 0:
                search_window = top_block
                top_left = top_block_top_left
            elif location == 1:
                search_window = bottom_block
                top_left = bottom_block_top_left
            elif location == 2:
                top_left_block_top_left = (top_block_top_left[0], top_block_top_left[1] - 1)
                top_left_block_top_left_relative_coor = (top_left_block_top_left[0] - og_search_window_top_left[0], top_left_block_top_left[1] - og_search_window_top_left[1])
                if top_left_block_top_left[1] >= 0:
                    bottom_left_block_top_left = (bottom_block_top_left[0], bottom_block_top_left[1] - 1)
                    bottom_left_block_top_left_relative_coor = (bottom_left_block_top_left[0] - og_search_window_top_left[0], bottom_left_block_top_left[1] - og_search_window_top_left[1])
                    top_left_block = og_search_window[top_left_block_top_left_relative_coor[0]:top_left_block_top_left_relative_coor[0] + params_i, top_left_block_top_left_relative_coor[1]:top_left_block_top_left_relative_coor[1] + params_i].astype(float)
                    bottom_left_block = og_search_window[bottom_left_block_top_left_relative_coor[0]:bottom_left_block_top_left_relative_coor[0] + params_i, bottom_left_block_top_left_relative_coor[1]:bottom_left_block_top_left_relative_coor[1] + params_i].astype(float)
                    top_center_block = (top_left_block + top_block) / 2
                    bottom_center_block = (bottom_left_block + bottom_block) / 2
                    left_center_block = (top_left_block + bottom_left_block) / 2
                    search_window = (top_center_block + bottom_center_block + left_center_block + current_block) / 4
                    top_left = (top_left[0] - 0.5, top_left[1])
            elif location == 3:
                top_right_block_top_left = (top_block_top_left[0], top_block_top_left[1] + 1)
                top_right_block_top_left_relative_coor = (top_right_block_top_left[0] - og_search_window_top_left[0], top_right_block_top_left[1] - og_search_window_top_left[1])
                if top_right_block_top_left[1] + params_i <= max_size[1]:
                    bottom_right_block_top_left = (bottom_block_top_left[0], bottom_block_top_left[1] + 1)
                    bottom_right_block_top_left_relative_coor = (bottom_right_block_top_left[0] - og_search_window_top_left[0], bottom_right_block_top_left[1] - og_search_window_top_left[1])
                    top_right_block = og_search_window[top_right_block_top_left_relative_coor[0]:top_right_block_top_left_relative_coor[0] + params_i, top_right_block_top_left_relative_coor[1]:top_right_block_top_left_relative_coor[1] + params_i].astype(float)
                    bottom_right_block = og_search_window[bottom_right_block_top_left_relative_coor[0]:bottom_right_block_top_left_relative_coor[0] + params_i, bottom_right_block_top_left_relative_coor[1]:bottom_right_block_top_left_relative_coor[1] + params_i].astype(float)
                    top_center_block = (top_right_block + top_block) / 2
                    bottom_center_block = (bottom_right_block + bottom_block) / 2
                    right_center_block = (top_right_block + bottom_right_block) / 2
                    search_window = (top_center_block + bottom_center_block + right_center_block + current_block) / 4
                    top_left = (top_left[0] + 0.5, top_left[1])
        elif is_x_interpolate and not is_y_interpolate:
            # x is interpolated, left and right blocks exist
            # y is not interpolated, top and bottom blocks may exist
            left_block_top_left = (int(top_left[0]), math.floor(top_left[1]))
            right_block_top_left = (int(top_left[0]), math.ceil(top_left[1]))
            left_block_top_left_relative_coor = (left_block_top_left[0] - og_search_window_top_left[0], left_block_top_left[1] - og_search_window_top_left[1])
            right_block_top_left_relative_coor = (right_block_top_left[0] - og_search_window_top_left[0], right_block_top_left[1] - og_search_window_top_left[1])
            left_block = og_search_window[left_block_top_left_relative_coor[0]:left_block_top_left_relative_coor[0] + params_i, left_block_top_left_relative_coor[1]:left_block_top_left_relative_coor[1] + params_i].astype(float)
            right_block = og_search_window[right_block_top_left_relative_coor[0]:right_block_top_left_relative_coor[0] + params_i, right_block_top_left_relative_coor[1]:right_block_top_left_relative_coor[1] + params_i].astype(float)
            if location == 0:
                top_left_block_top_left = (left_block_top_left[0] - 1, left_block_top_left[1])
                top_left_block_top_left_relative_coor = (top_left_block_top_left[0] - og_search_window_top_left[0], top_left_block_top_left[1] - og_search_window_top_left[1])
                if top_left_block_top_left[0] >= 0:
                    top_right_block_top_left = (right_block_top_left[0] - 1, right_block_top_left[1])
                    top_right_block_top_left_relative_coor = (top_right_block_top_left[0] - og_search_window_top_left[0], top_right_block_top_left[1] - og_search_window_top_left[1])
                    top_left_block = og_search_window[top_left_block_top_left_relative_coor[0]:top_left_block_top_left_relative_coor[0] + params_i, top_left_block_top_left_relative_coor[1]:top_left_block_top_left_relative_coor[1] + params_i].astype(float)
                    top_right_block = og_search_window[top_right_block_top_left_relative_coor[0]:top_right_block_top_left_relative_coor[0] + params_i, top_right_block_top_left_relative_coor[1]:top_right_block_top_left_relative_coor[1] + params_i].astype(float)
                    top_center_block = (top_left_block + top_right_block) / 2
                    left_center_block = (top_left_block + left_block) / 2
                    right_center_block = (top_right_block + right_block) / 2
                    search_window = (top_center_block + left_center_block + right_center_block + current_block) / 4
                    top_left = (top_left[0], top_left[1] - 0.5)
            elif location == 1:
                bottom_left_block_top_left = (left_block_top_left[0] + 1, left_block_top_left[1])
                bottom_left_block_top_left_relative_coor = (bottom_left_block_top_left[0] - og_search_window_top_left[0], bottom_left_block_top_left[1] - og_search_window_top_left[1])
                if bottom_left_block_top_left[0] + params_i <= max_size[0]:
                    bottom_right_block_top_left = (right_block_top_left[0] + 1, right_block_top_left[1])
                    bottom_right_block_top_left_relative_coor = (bottom_right_block_top_left[0] - og_search_window_top_left[0], bottom_right_block_top_left[1] - og_search_window_top_left[1])
                    bottom_left_block = og_search_window[bottom_left_block_top_left_relative_coor[0]:bottom_left_block_top_left_relative_coor[0] + params_i, bottom_left_block_top_left_relative_coor[1]:bottom_left_block_top_left_relative_coor[1] + params_i].astype(float)
                    bottom_right_block = og_search_window[bottom_right_block_top_left_relative_coor[0]:bottom_right_block_top_left_relative_coor[0] + params_i, bottom_right_block_top_left_relative_coor[1]:bottom_right_block_top_left_relative_coor[1] + params_i].astype(float)
                    bottom_center_block = (bottom_left_block + bottom_right_block) / 2
                    left_center_block = (bottom_left_block + left_block) / 2
                    right_center_block = (bottom_right_block + right_block) / 2
                    search_window = (bottom_center_block + left_center_block + right_center_block + current_block) / 4
                    top_left = (top_left[0], top_left[1] + 0.5)
            elif location == 2:
                search_window = left_block
                top_left = left_block_top_left
            elif location == 3:
                search_window = right_block
                top_left = right_block_top_left
        else:
            # both x and y are interpolated, which means top left, top right, bottom left, bottom right blocks exist
            top_left_block_top_left = (math.floor(top_left[0]), math.floor(top_left[1]))
            top_right_block_top_left = (math.floor(top_left[0]), math.ceil(top_left[1]))
            bottom_left_block_top_left = (math.ceil(top_left[0]), math.floor(top_left[1]))
            bottom_right_block_top_left = (math.ceil(top_left[0]), math.ceil(top_left[1]))
            top_left_block_top_left_relative_coor = (top_left_block_top_left[0] - og_search_window_top_left[0], top_left_block_top_left[1] - og_search_window_top_left[1])
            top_right_block_top_left_relative_coor = (top_right_block_top_left[0] - og_search_window_top_left[0], top_right_block_top_left[1] - og_search_window_top_left[1])
            bottom_left_block_top_left_relative_coor = (bottom_left_block_top_left[0] - og_search_window_top_left[0], bottom_left_block_top_left[1] - og_search_window_top_left[1])
            bottom_right_block_top_left_relative_coor = (bottom_right_block_top_left[0] - og_search_window_top_left[0], bottom_right_block_top_left[1] - og_search_window_top_left[1])
            top_left_block = og_search_window[top_left_block_top_left_relative_coor[0]:top_left_block_top_left_relative_coor[0] + params_i, top_left_block_top_left_relative_coor[1]:top_left_block_top_left_relative_coor[1] + params_i].astype(float)
            top_right_block = og_search_window[top_right_block_top_left_relative_coor[0]:top_right_block_top_left_relative_coor[0] + params_i, top_right_block_top_left_relative_coor[1]:top_right_block_top_left_relative_coor[1] + params_i].astype(float)
            bottom_left_block = og_search_window[bottom_left_block_top_left_relative_coor[0]:bottom_left_block_top_left_relative_coor[0] + params_i, bottom_left_block_top_left_relative_coor[1]:bottom_left_block_top_left_relative_coor[1] + params_i].astype(float)
            bottom_right_block = og_search_window[bottom_right_block_top_left_relative_coor[0]:bottom_right_block_top_left_relative_coor[0] + params_i, bottom_right_block_top_left_relative_coor[1]:bottom_right_block_top_left_relative_coor[1] + params_i].astype(float)
            if location == 0:
                search_window = (top_left_block + top_right_block) / 2
                top_left = (top_left[0] - 0.5, top_left[1])
            elif location == 1:
                search_window = (bottom_left_block + bottom_right_block) / 2
                top_left = (top_left[0] + 0.5, top_left[1])
            elif location == 2:
                search_window = (top_left_block + bottom_left_block) / 2
                top_left = (top_left[0], top_left[1] - 0.5)
            elif location == 3:
                search_window = (top_right_block + bottom_right_block) / 2
                top_left = (top_left[0], top_left[1] + 0.5)
    else:
        if location == 0:
            top_block_top_left = (int(top_left[0]) - 1, int(top_left[1]))
            top_block_top_left_relative_coor = (top_block_top_left[0] - og_search_window_top_left[0], top_block_top_left[1] - og_search_window_top_left[1])
            if top_block_top_left[0] >= 0:
                top_block = og_search_window[top_block_top_left_relative_coor[0]:top_block_top_left_relative_coor[0] + params_i, top_block_top_left_relative_coor[1]:top_block_top_left_relative_coor[1] + params_i].astype(float)
                search_window = (top_block + current_block) / 2
                top_left = (top_left[0] - 0.5, top_left[1])
        elif location == 1:
            bottom_block_top_left = (int(top_left[0]) + 1, int(top_left[1]))
            bottom_block_top_left_relative_coor = (bottom_block_top_left[0] - og_search_window_top_left[0], bottom_block_top_left[1] - og_search_window_top_left[1])
            if bottom_block_top_left[0] + params_i <= max_size[0]:
                bottom_block = og_search_window[bottom_block_top_left_relative_coor[0]:bottom_block_top_left_relative_coor[0] + params_i, bottom_block_top_left_relative_coor[1]:bottom_block_top_left_relative_coor[1] + params_i].astype(float)
                search_window = (bottom_block + current_block) / 2
                top_left = (top_left[0] + 0.5, top_left[1])
        elif location == 2:
            left_block_top_left = (int(top_left[0]), int(top_left[1]) - 1)
            left_block_top_left_relative_coor = (left_block_top_left[0] - og_search_window_top_left[0], left_block_top_left[1] - og_search_window_top_left[1])
            if left_block_top_left[1] >= 0:
                left_block = og_search_window[left_block_top_left_relative_coor[0]:left_block_top_left_relative_coor[0] + params_i, left_block_top_left_relative_coor[1]:left_block_top_left_relative_coor[1] + params_i].astype(float)
                search_window = (left_block + current_block) / 2
                top_left = (top_left[0], top_left[1] - 0.5)
        elif location == 3:
            right_block_top_left = (int(top_left[0]), int(top_left[1]) + 1)
            right_block_top_left_relative_coor = (right_block_top_left[0] - og_search_window_top_left[0], right_block_top_left[1] - og_search_window_top_left[1])
            if right_block_top_left[1] + params_i <= max_size[1]:
                right_block = og_search_window[right_block_top_left_relative_coor[0]:right_block_top_left_relative_coor[0] + params_i, right_block_top_left_relative_coor[1]:right_block_top_left_relative_coor[1] + params_i].astype(float)
                search_window = (right_block + current_block) / 2
                top_left = (top_left[0], top_left[1] + 0.5)
    return top_left, search_window

def calc_fast_motion_vector(block: np.ndarray, block_coor: tuple, og_search_windows, og_search_window_coors, params_i, params, mvp) -> tuple:
    """
        Nearest Neighbors search is a requirement, with MVP being the MV of the latest encoded block (MVP =(0,0) for first block in every row of (𝑖 × 𝑖) blocks). Note: any candidate block that partially or fully exists outside of the frame is not searched. Lecture 6

        Parameters:
            block (np.ndarray): The block.
            block_coor (tuple): The top left coordinates of block.
            frame (Frame): The current frame.
            params_i (int): The block size.
            mvp (MotionVector): The motion vector predictor.
        
        Returns:
            min_motion_vector (MotionVector): The motion vector object.
            min_block (np.ndarray): The block from the search window.
    """
    min_motion_vector = None
    min_yx = None
    min_block = None
    mvp = MotionVector(0, 0) if mvp is None else mvp

    # baseline
    search_window_offset = (block_coor[0] - og_search_window_coors[0][0], block_coor[1] - og_search_window_coors[0][1])
    search_windows = []
    for search_window in og_search_windows:
        search_window = search_window[search_window_offset[0]:search_window_offset[0] + params_i, search_window_offset[1]:search_window_offset[1] + params_i]
        search_windows.append(search_window)

    baseline_motion_vector, baseline_block = calc_full_range_motion_vector(block, block_coor, search_windows, block_coor, params_i, False)
    min_motion_vector = baseline_motion_vector
    min_yx = block_coor
    min_block = baseline_block

    flag = False
    search_origin = block_coor
    search_window_top_left, search_window_bottom_right = og_search_window_coors
    while not flag:
        # search mvp
        new_coor = (search_origin[0] + mvp.y, search_origin[1] + mvp.x)

        # generate 5 coor
        coors = [
            (-1, 0), # top
            (1, 0), # bottom
            (0, - 1), # left
            (0, 1), # right
        ]
        for coor_index in range(len(coors)):
            current_coor = (new_coor[0] + coors[coor_index][0], new_coor[1] + coors[coor_index][1])
            search_window_offset = (current_coor[0] - search_window_top_left[0], current_coor[1] - search_window_top_left[1])
            if current_coor[0] < 0 or current_coor[1] < 0 or current_coor[0] < search_window_top_left[0] or current_coor[1] < search_window_top_left[1] or current_coor[0] + params_i > search_window_bottom_right[0] or current_coor[1] + params_i > search_window_bottom_right[1]:
                continue
            top_left = current_coor
            search_windows = []
            selection_flag = True
            for og_search_window in og_search_windows:
                if not selection_flag:
                    break
                if params.FMEEnable:
                    # center block's top left in interpolated frame
                    scaled_center_top_left = (new_coor[0] * 2, new_coor[1] * 2)
                    # search block's top left in interpolated frame
                    scaled_top_left = (float(scaled_center_top_left[0] + coors[coor_index][0]), float(scaled_center_top_left[1] + coors[coor_index][1]))
                    # actual top left in the true frame
                    top_left = (scaled_top_left[0] / 2, scaled_top_left[1] / 2)
                    top_left, search_window = get_interpolated_block(baseline_block, top_left, coor_index, og_search_window, search_window_top_left, params_i)
                    if search_window is None:
                        selection_flag = False
                        continue
                else:
                    search_window = og_search_window[search_window_offset[0]:search_window_offset[0] + params_i, search_window_offset[1]:search_window_offset[1] + params_i]
                search_windows.append(search_window)
            
            if not selection_flag:
                continue

            current_min_motion_vector, current_min_block = calc_full_range_motion_vector(block, block_coor, search_windows, top_left, params_i, False)

            if current_min_motion_vector.mae < min_motion_vector.mae:
                min_motion_vector = current_min_motion_vector
                min_yx = current_coor
                min_block = current_min_block
            elif current_min_motion_vector.mae == min_motion_vector.mae:
                current_min_l1_norm = current_min_motion_vector.l1_norm()
                new_min_l1_norm = min_motion_vector.l1_norm()
                if new_min_l1_norm < current_min_l1_norm:
                    min_motion_vector = current_min_motion_vector
                    min_yx = current_coor
                    min_block = current_min_block
                elif new_min_l1_norm == current_min_l1_norm:
                    if current_coor[0] < min_yx[0]:
                        min_motion_vector = current_min_motion_vector
                        min_yx = current_coor
                        min_block = current_min_block
                    elif current_coor[0] == min_yx[0]:
                        if current_coor[1] < min_yx[1]:
                            min_motion_vector = current_min_motion_vector
                            min_yx = current_coor
                            min_block = current_min_block
            
        if min_motion_vector == baseline_motion_vector:
            flag = True
            break
        else:
            baseline_motion_vector = min_motion_vector
            baseline_block = min_block
            search_origin = min_yx
            mvp = min_motion_vector
    return min_motion_vector, min_block

"""
    Calculate the motion vector for a block from the search window.

    Parameters:
        block (np.ndarray): The block.
        block_coor (tuple): The top left coordinates of block.
        search_windows (list): A list of search windows.
        search_window_coor (tuple): The top left coordinates of search window.
        params_i (int): The block size.
    
    Returns:
        min_motion_vector (MotionVector): The motion vector object.
        min_block (np.ndarray): The block from the search window.
"""
def calc_full_range_motion_vector(block: np.ndarray, block_coor: tuple, search_windows: list, search_window_coor: tuple, params_i: int, FMEEnable: bool) -> tuple:
    min_motion_vector = None
    min_yx = None
    block_reshaped = block.reshape(params_i * params_i)
    min_block = None
    for search_windows_index in range(len(search_windows)):
        current_search_window = search_windows[search_windows_index]
        search_window_list = interpolate_search_window(current_search_window, search_window_coor, params_i, block_coor, FMEEnable)
        for search_block_dict in search_window_list:
            actual_y, actual_x = search_block_dict['actual_yx']
            offset = (actual_y - block_coor[0], actual_x - block_coor[1])
            search_block = (search_block_dict['block']).astype(np.uint8)
            motion_vector = MotionVector(offset[0], offset[1], ref_offset=search_windows_index, mae=np.abs(search_block.reshape(params_i * params_i) - block_reshaped).mean())
            if min_motion_vector == None:
                min_motion_vector = motion_vector
                min_yx = (actual_y, actual_x)
                min_block = search_block
            elif motion_vector.mae < min_motion_vector.mae:
                min_motion_vector = motion_vector
                min_yx = (actual_y, actual_x)
                min_block = search_block
            elif motion_vector.mae == min_motion_vector.mae:
                current_min_l1_norm = min_motion_vector.l1_norm()
                new_min_l1_norm = motion_vector.l1_norm()
                if new_min_l1_norm < current_min_l1_norm:
                    min_motion_vector = motion_vector
                    min_yx = (actual_y, actual_x)
                    min_block = search_block
                elif new_min_l1_norm == current_min_l1_norm:
                    if actual_y < min_yx[0]:
                        min_motion_vector = motion_vector
                        min_yx = (actual_y, actual_x)
                        min_block = search_block
                    elif actual_y == min_yx[0]:
                        if actual_x < min_yx[1]:
                            min_motion_vector = motion_vector
                            min_yx = (actual_y, actual_x)
                            min_block = search_block

    return min_motion_vector, min_block

def interpolate_search_window(search_window, search_window_coor, params_i, block_coor, FMEEnable):
    search_window_list = []
    for y in range(0, search_window.shape[0] - params_i + 1):
        actual_y = y + search_window_coor[0]
        search_window_list.append([])
        for x in range(0, search_window.shape[1] - params_i + 1):
            actual_x = x + search_window_coor[1]
            a = search_window[y:params_i + y, x:params_i + x]
            search_window_list[-1].append(dict(
                block=a,
                actual_yx=(actual_y, actual_x),
                offset=(actual_y - block_coor[0], actual_x - block_coor[1])
            ))

    if FMEEnable:
        row_length = len(search_window_list)
        col_length = len(search_window_list[0])

        #insert between columns
        x_counter = 0
        while x_counter < (col_length - 1) * 2:
            for y in range(row_length):
                left_search_block = search_window_list[y][x_counter]
                right_search_block = search_window_list[y][x_counter + 1]
                middle_search_block = (left_search_block['block'].astype(float) + right_search_block['block'].astype(float)) / 2
                search_window_list[y].insert(x_counter + 1, dict(
                    block=middle_search_block,
                    actual_yx=(left_search_block['actual_yx'][0], left_search_block['actual_yx'][1] + 0.5),
                    offset=(left_search_block['offset'][0], left_search_block['offset'][1] + 0.5)
                ))
            x_counter += 2
        
        #insert between rows
        y_counter = 0
        while y_counter < (row_length - 1) * 2:
            search_window_list.insert(y_counter + 1, [])
            for x in range(len(search_window_list[0])):
                top_search_block = search_window_list[y_counter][x]
                bottom_search_block = search_window_list[y_counter + 2][x]
                middle_search_block = (top_search_block['block'].astype(float) + bottom_search_block['block'].astype(float)) / 2
                search_window_list[y_counter + 1].append(dict(
                    block=middle_search_block,
                    actual_yx=(top_search_block['actual_yx'][0] + 0.5, top_search_block['actual_yx'][1]),
                    offset=(top_search_block['offset'][0] + 0.5, top_search_block['offset'][1])
                ))
            y_counter += 2

    return [item for sublist in search_window_list for item in sublist]

def interframe_block_prediction(current_coor, frame, params, buffer, q_matrix, prev_motion_vector=None, data_queue=None, qp = None):
    split_counter = 0
    mv_integrate = ''
    centered_block = frame.raw[current_coor[0]:current_coor[0] + frame.params_i, current_coor[1]:current_coor[1] + frame.params_i]

    buffer_state = False
    top = right = bottom = left = params.r
    if buffer.state('interframe_block_prediction_me', current_coor):
        buffer_state = True
        min_motion_vector = buffer.get('interframe_block_prediction_me', current_coor)
        if min_motion_vector.y == 0:
            top = bottom = params.r // 2
        elif min_motion_vector.y > 0:
            top = 0
        elif min_motion_vector.y < 0:
            bottom = 0
        
        if min_motion_vector.x == 0:
            left = right = params.r // 2
        elif min_motion_vector.x > 0:
            left = 0
        elif min_motion_vector.x < 0:
            right = 0

    top_left, bottom_right = extend_block(current_coor, frame.params_i, (top, right, bottom, left), frame.shape)
    current_frame = frame
    search_windows = []
    while current_frame.prev is not None:
        search_window = current_frame.prev.raw[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
        search_windows.append(search_window)
        current_frame = current_frame.prev
    if params.FastME:
        min_motion_vector, min_block = calc_fast_motion_vector(centered_block, current_coor, search_windows, (top_left, bottom_right), frame.params_i, params, prev_motion_vector)
    else:
        min_motion_vector, min_block = calc_full_range_motion_vector(centered_block, current_coor, search_windows, top_left, frame.params_i, params.FMEEnable)
    if not buffer_state:
        buffer.add('interframe_block_prediction_me', current_coor, min_motion_vector)
    
    if params.RCflag != 0:
        qtc_block = QTCBlock(block=centered_block - min_block, q_matrix=q_matrix, qp=qp)
    else:
        qtc_block = QTCBlock(block=centered_block - min_block, q_matrix=q_matrix, qp=params.qp)
    qtc_block.block_to_qtc()
    reconstructed_block = qtc_block.block + min_block
    diff_mv = min_motion_vector - prev_motion_vector if prev_motion_vector is not None else min_motion_vector
    prev_motion_vector = min_motion_vector
    bitcount_per_block = len(qtc_block.to_str()) + len(min_motion_vector.to_str(is_intraframe=False))

    vbsed = False
    buffer_state = False
    if buffer.state('interframe_block_prediction_vbs', current_coor):
        # buffer state exists
        buffer_state = True
        vbsed = buffer.get('interframe_block_prediction_vbs', current_coor)
    
    if not buffer_state:
        vbsed = True

    if params.VBSEnable:
        if vbsed:
            vbs_qtc_block, vbs_reconstructed_block, vbs_mv = interframe_vbs(current_coor, centered_block, search_windows, top_left, reconstructed_block, qtc_block, diff_mv, prev_motion_vector, params, qp)
            reconstructed_block = vbs_reconstructed_block
            if vbs_mv is not None:
                for mv in vbs_mv:
                    mv_integrate += mv.to_str(is_intraframe=False)
                vbs_mv_length = len(mv_integrate)
                bitcount_per_block = len(vbs_qtc_block.to_str()) + vbs_mv_length
                qtc_block = dict(
                    vbs=VBSMarker.SPLIT,
                    qtc_block=vbs_qtc_block,
                )
                min_motion_vector = dict(
                    vbs=VBSMarker.SPLIT,
                    predictor=vbs_mv,
                )
                prev_motion_vector = vbs_mv[-1]
                print('vbs used in Frame', frame.index, current_coor)
                split_counter += 1
                if not buffer_state:
                    buffer.add('interframe_block_prediction_vbs', current_coor, True)
            else:
                bitcount_per_block = len(qtc_block.to_str()) + len(min_motion_vector.to_str(is_intraframe=False))
                qtc_block = dict(
                    vbs=VBSMarker.UNSPLIT,
                    qtc_block=qtc_block,
                )
                min_motion_vector = dict(
                    vbs=VBSMarker.UNSPLIT,
                    predictor=min_motion_vector,
                )
                if not buffer_state:
                    buffer.add('interframe_block_prediction_vbs', current_coor, False)
        else:
            bitcount_per_block = len(qtc_block.to_str()) + len(min_motion_vector.to_str(is_intraframe=False))
            qtc_block = dict(
                vbs=VBSMarker.UNSPLIT,
                qtc_block=qtc_block,
            )
            min_motion_vector = dict(
                vbs=VBSMarker.UNSPLIT,
                predictor=min_motion_vector,
            )

    if data_queue is not None:
        reconstructed_blocks = [convert_within_range(reconstructed_block)]
        current_frame = frame
        while current_frame.prev is not None:
            reconstructed_blocks.append(convert_within_range(current_frame.prev.raw[current_coor[0]:current_coor[0] + frame.params_i, current_coor[1]:current_coor[1] + frame.params_i]))
            current_frame = current_frame.prev
        data_queue.put((current_coor, reconstructed_blocks))
    return current_coor, qtc_block, min_motion_vector, reconstructed_block, split_counter, bitcount_per_block

def interframe_prediction(index: int, frame: Frame, params: Params, q_matrix: np.ndarray, y: int, qp = None, buffer = None) -> tuple:
    """
        Helper function to calculate the motion vector, residual blocks, and mae values.

        Parameters:
            index (int): The index of the frame.
            frame (Frame): The current frame.
            params_r (int): The search window size.
            q_matrix (np.ndarray): The quantization matrix.
            y (int): The y coordinate of the block.

        Returns:
            index (int): The index of the frame.
            qtc_block_dump (np.ndarray): The quantized transformed coefficients.
            mv_dump (list): The motion vectors.
            mae_dump (list): The mean absolute errors.
            reconstructed_block_dump (np.ndarray): The reconstructed blocks.
    """
    qtc_block_dump = []
    mv_dump = []
    reconstructed_block_dump = []
    prev_motion_vector = None
    split_counter = 0
    bit_count_in_onerow= 0
    for x in range(0, frame.width, frame.params_i):
        _, qtc_block, min_motion_vector, reconstructed_block, current_split_counter , bitcount_per_block= interframe_block_prediction((y, x), frame, params, buffer, q_matrix, prev_motion_vector, qp= qp)
        
        if current_split_counter > 0:
            split_counter += current_split_counter
        if params.VBSEnable:
            if min_motion_vector['vbs'] == VBSMarker.SPLIT:
                prev_motion_vector = min_motion_vector['predictor'][-1]
            else:
                prev_motion_vector = min_motion_vector['predictor']
        else:
            prev_motion_vector = min_motion_vector
        bit_count_in_onerow += bitcount_per_block
        qtc_block_dump.append(qtc_block)
        reconstructed_block_dump.append(reconstructed_block)
        mv_dump.append(min_motion_vector)
    return (index, qtc_block_dump, mv_dump, reconstructed_block_dump, split_counter, bit_count_in_onerow, buffer)
