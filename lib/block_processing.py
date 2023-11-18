import math
import numpy as np
from multiprocessing import Pool
from pathlib import Path

from lib.config.config import Params
from lib.enums import Intraframe
from lib.components.frame import Frame, extend_block
from lib.components.qtc import QTCBlock, QTCFrame, quantization_matrix
from lib.components.mv import MotionVector, MotionVectorFrame
from lib.enums import VBSMarker

def rdo(original_block: np.ndarray, reconstructed_block: np.ndarray, qtc_block: QTCBlock, mv: MotionVector, params_qp: int, is_intraframe=False):
    lambda_value = 0.5 ** ((params_qp - 12) / 3) * 5
    sad_value = np.abs(original_block - reconstructed_block).sum()
    r_vaule = len(qtc_block.to_str()) + len(mv.to_str(is_intraframe))
    return sad_value + lambda_value * r_vaule

def interframe_vbs(coor_offset: tuple, original_block: np.ndarray, original_search_windows: list, reconstructed_block: np.ndarray, qtc_block: QTCBlock, diff_mv: MotionVector, prev_motion_vector: MotionVector, frame: Frame, params: Params):
    """
        Implementation verification pending
    """
    block_rdo_cost = rdo(original_block, reconstructed_block, qtc_block, diff_mv, params.qp, is_intraframe=False)
    subblock_params_i = params.i // 2
    q_matrix = quantization_matrix(subblock_params_i, params.qp - 1 if params.qp > 0 else 0)
    top_lefts = [(y, x) for y in range(0, original_block.shape[0], subblock_params_i) for x in range(0, original_block.shape[1], subblock_params_i)]
    top_lefts_in_search_window = [(y + coor_offset[0], x + coor_offset[1]) for y, x in top_lefts] if coor_offset is not None else None
    subblock_rdo_cost = 0
    qtc_subblocks = []
    reconstructed_subblocks = []
    mv_subblocks = []
    residual_subblocks = []
    for centered_top_left_index in range(len(top_lefts)):
        centered_top_left = top_lefts[centered_top_left_index]
        centered_subblock = original_block[centered_top_left[0]:centered_top_left[0] + subblock_params_i, centered_top_left[1]:centered_top_left[1] + subblock_params_i]
        if params.FastME:
            min_motion_vector, min_block = calc_fast_motion_vector(centered_subblock, centered_top_left, frame, subblock_params_i, params, prev_motion_vector)
        else:
            top_left_in_search_window = top_lefts_in_search_window[centered_top_left_index]
            top_left, bottom_right = extend_block(top_left_in_search_window, subblock_params_i, (params.r, params.r, params.r, params.r), original_block.shape)
            search_windows = []
            for original_search_window in original_search_windows:
                search_window = original_search_window[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
                search_windows.append(search_window)
            min_motion_vector, min_block = calc_full_range_motion_vector(centered_subblock, top_left_in_search_window, search_windows, top_left, subblock_params_i, params.FMEEnable)

        qtc_subblock = QTCBlock(block=centered_subblock - min_block, q_matrix=q_matrix)
        qtc_subblock.block_to_qtc()
        residual_subblocks.append(qtc_subblock.block)
        reconstructed_subblock = qtc_subblock.block + min_block
        diff_submv = min_motion_vector - prev_motion_vector if prev_motion_vector is not None else min_motion_vector
        prev_motion_vector = min_motion_vector
        subblock_rdo_cost += rdo(centered_subblock, reconstructed_subblock, qtc_subblock, diff_submv, params.qp, is_intraframe=False)
        qtc_subblocks.append(qtc_subblock.qtc_block)
        reconstructed_subblocks.append(reconstructed_subblock)
        mv_subblocks.append(min_motion_vector)

    if subblock_rdo_cost < block_rdo_cost:
        qtc_stack = np.concatenate((np.concatenate((qtc_subblocks[0], qtc_subblocks[1]), axis=1), np.concatenate((qtc_subblocks[2], qtc_subblocks[3]), axis=1)), axis=0)
        temp_stack = np.concatenate((np.concatenate((residual_subblocks[0], residual_subblocks[1]), axis=1), np.concatenate((residual_subblocks[2], residual_subblocks[3]), axis=1)), axis=0)
        reconstructed_stack = np.concatenate((np.concatenate((reconstructed_subblocks[0], reconstructed_subblocks[1]), axis=1), np.concatenate((reconstructed_subblocks[2], reconstructed_subblocks[3]), axis=1)), axis=0)
        qtc_block = QTCBlock(qtc_block=qtc_stack, block=temp_stack)
        return qtc_block, reconstructed_stack, mv_subblocks
    else:
        return qtc_block, reconstructed_block, None

def intraframe_vbs(current_coor:tuple, original_block: np.ndarray, reconstructed_block: np.ndarray, reconstructed_block_dump, qtc_block: QTCBlock, diff_predictor: int, params: Params):
    """
        Implementation verification pending
    """
    block_rdo_cost = rdo(original_block, reconstructed_block, qtc_block, diff_predictor, params.qp, is_intraframe=True)
    subblock_params_i = params.i // 2
    q_matrix = quantization_matrix(subblock_params_i, params.qp - 1 if params.qp > 0 else 0)
    top_lefts = [(y, x) for y in range(0, original_block.shape[0], subblock_params_i) for x in range(0, original_block.shape[1], subblock_params_i)]
    subblock_rdo_cost = 0
    prev_predictor = diff_predictor
    subpredictor_dump = []
    qtc_subblocks = []
    reconstructed_subblocks = np.empty(original_block.shape, dtype=int)
    residual_subblocks = []
    for centered_top_left_index in range(len(top_lefts)):
        centered_top_left = top_lefts[centered_top_left_index]
        current_block = original_block[centered_top_left[0]:centered_top_left[0] + subblock_params_i, centered_top_left[1]:centered_top_left[1] + subblock_params_i]
        hor_top_left, _ = extend_block(centered_top_left, subblock_params_i, (0, 0, 0, 1), original_block.shape)
        ver_top_left, _ = extend_block(centered_top_left, subblock_params_i, (1, 0, 0, 0), original_block.shape)
        
        # select vertical edge
        if hor_top_left[Intraframe.HORIZONTAL.value] == centered_top_left[Intraframe.HORIZONTAL.value] + current_coor[1]:
            hor_block = np.full((subblock_params_i, 1), 128)
        else:
            if centered_top_left_index == 0 or centered_top_left_index == 2:
                hor_block = reconstructed_block_dump.raw[current_coor[0] + centered_top_left[0]:current_coor[0] + centered_top_left[0] + subblock_params_i, current_coor[1] + centered_top_left[1] - 1:current_coor[1] + centered_top_left[1]]
            else:
                hor_block = reconstructed_subblocks[hor_top_left[0]:hor_top_left[0] + subblock_params_i, hor_top_left[1]:hor_top_left[1] + 1]
        hor_block = hor_block.repeat(subblock_params_i, Intraframe.HORIZONTAL.value)
        hor_mae = np.abs(hor_block - current_block).mean().astype(int)

        # select horizontal edge
        if ver_top_left[Intraframe.VERTICAL.value] == centered_top_left[Intraframe.VERTICAL.value] + current_coor[0]:
            ver_block = np.full((1, subblock_params_i), 128)
        else:
            if centered_top_left_index == 0 or centered_top_left_index == 1:
                ver_block = reconstructed_block_dump.raw[current_coor[0] + centered_top_left[0] - 1:current_coor[0] + centered_top_left[0], current_coor[1] + centered_top_left[1]:current_coor[1] + centered_top_left[1] + subblock_params_i]
            else:
                ver_block = reconstructed_subblocks[ver_top_left[0]:ver_top_left[0] + 1, ver_top_left[1]:ver_top_left[1] + subblock_params_i]
        ver_block = ver_block.repeat(subblock_params_i, Intraframe.VERTICAL.value)
        ver_mae = np.abs(ver_block - current_block).mean().astype(int)
        predictor_block = None
        if ver_mae < hor_mae:
            current_predictor = MotionVector(Intraframe.VERTICAL.value, -1, mae=ver_mae)
            predictor_block = ver_block
        else:
            current_predictor = MotionVector(Intraframe.HORIZONTAL.value, -1, mae=hor_mae)
            predictor_block = hor_block
        
        qtc_subblock = QTCBlock(block=current_block - predictor_block, q_matrix=q_matrix)
        qtc_subblock.block_to_qtc()
        residual_subblocks.append(qtc_subblock.block)
        reconstructed_subblock = qtc_subblock.block + predictor_block
        diff_subpredictor = current_predictor - prev_predictor if prev_predictor is not None else current_predictor
        prev_predictor = current_predictor
        subblock_rdo_cost += rdo(current_block, reconstructed_subblock, qtc_subblock, diff_subpredictor, params.qp, is_intraframe=True)
        qtc_subblocks.append(qtc_subblock.qtc_block)
        reconstructed_subblocks[centered_top_left[0]:centered_top_left[0] + subblock_params_i, centered_top_left[1]:centered_top_left[1] + subblock_params_i] = reconstructed_subblock
        subpredictor_dump.append(current_predictor)

    if subblock_rdo_cost < block_rdo_cost:
        qtc_stack = np.concatenate((np.concatenate((qtc_subblocks[0], qtc_subblocks[1]), axis=1), np.concatenate((qtc_subblocks[2], qtc_subblocks[3]), axis=1)), axis=0)
        temp_stack = np.concatenate((np.concatenate((residual_subblocks[0], residual_subblocks[1]), axis=1), np.concatenate((residual_subblocks[2], residual_subblocks[3]), axis=1)), axis=0)
        qtc_block = QTCBlock(qtc_block=qtc_stack, block=temp_stack)
        return qtc_block, reconstructed_subblocks, subpredictor_dump
    else:
        return qtc_block, reconstructed_block, None

def get_interpolated_block(current_block, current_coor, location, frame, params_i):
    # location = [0, 1, 2, 3] #top, bottom, left, right
    current_block = current_block.astype(float)
    top_left = current_coor
    max_size = frame.shape
    search_window = None
    is_y_interpolate = not float(current_coor[0]).is_integer()
    is_x_interpolate = not float(current_coor[1]).is_integer()
    if is_y_interpolate or is_x_interpolate:
        if is_y_interpolate and not is_x_interpolate:
            # y is interpolated, top and bottom blocks exist
            # x is not interpolated, left and right blocks may exist
            top_block_top_left = (math.floor(top_left[0]), int(top_left[1]))
            bottom_block_top_left = (math.ceil(top_left[0]), int(top_left[1]))
            top_block = frame.raw[top_block_top_left[0]:top_block_top_left[0] + params_i, top_block_top_left[1]:top_block_top_left[1] + params_i].astype(float)
            bottom_block = frame.raw[bottom_block_top_left[0]:bottom_block_top_left[0] + params_i, bottom_block_top_left[1]:bottom_block_top_left[1] + params_i].astype(float)
            if location == 0:
                search_window = top_block
                top_left = top_block_top_left
            elif location == 1:
                search_window = bottom_block
                top_left = bottom_block_top_left
            elif location == 2:
                top_left_block_top_left = (top_block_top_left[0], top_block_top_left[1] - 1)
                if top_left_block_top_left[1] >= 0:
                    bottom_left_block_top_left = (bottom_block_top_left[0], bottom_block_top_left[1] - 1)
                    top_left_block = frame.raw[top_left_block_top_left[0]:top_left_block_top_left[0] + params_i, top_left_block_top_left[1]:top_left_block_top_left[1] + params_i].astype(float)
                    bottom_left_block = frame.raw[bottom_left_block_top_left[0]:bottom_left_block_top_left[0] + params_i, bottom_left_block_top_left[1]:bottom_left_block_top_left[1] + params_i].astype(float)
                    top_center_block = (top_left_block + top_block) / 2
                    bottom_center_block = (bottom_left_block + bottom_block) / 2
                    left_center_block = (top_left_block + bottom_left_block) / 2
                    search_window = (top_center_block + bottom_center_block + left_center_block + current_block) / 4
                    top_left = (top_left[0] - 0.5, top_left[1])
            elif location == 3:
                top_right_block_top_left = (top_block_top_left[0], top_block_top_left[1] + 1)
                if top_right_block_top_left[1] + params_i <= max_size[1]:
                    bottom_right_block_top_left = (bottom_block_top_left[0], bottom_block_top_left[1] + 1)
                    top_right_block = frame.raw[top_right_block_top_left[0]:top_right_block_top_left[0] + params_i, top_right_block_top_left[1]:top_right_block_top_left[1] + params_i].astype(float)
                    bottom_right_block = frame.raw[bottom_right_block_top_left[0]:bottom_right_block_top_left[0] + params_i, bottom_right_block_top_left[1]:bottom_right_block_top_left[1] + params_i].astype(float)
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
            left_block = frame.raw[left_block_top_left[0]:left_block_top_left[0] + params_i, left_block_top_left[1]:left_block_top_left[1] + params_i].astype(float)
            right_block = frame.raw[right_block_top_left[0]:right_block_top_left[0] + params_i, right_block_top_left[1]:right_block_top_left[1] + params_i].astype(float)
            if location == 0:
                top_left_block_top_left = (left_block_top_left[0] - 1, left_block_top_left[1])
                if top_left_block_top_left[0] >= 0:
                    top_right_block_top_left = (right_block_top_left[0] - 1, right_block_top_left[1])
                    top_left_block = frame.raw[top_left_block_top_left[0]:top_left_block_top_left[0] + params_i, top_left_block_top_left[1]:top_left_block_top_left[1] + params_i].astype(float)
                    top_right_block = frame.raw[top_right_block_top_left[0]:top_right_block_top_left[0] + params_i, top_right_block_top_left[1]:top_right_block_top_left[1] + params_i].astype(float)
                    top_center_block = (top_left_block + top_right_block) / 2
                    left_center_block = (top_left_block + left_block) / 2
                    right_center_block = (top_right_block + right_block) / 2
                    search_window = (top_center_block + left_center_block + right_center_block + current_block) / 4
                    top_left = (top_left[0], top_left[1] - 0.5)
            elif location == 1:
                bottom_left_block_top_left = (left_block_top_left[0] + 1, left_block_top_left[1])
                if bottom_left_block_top_left[0] + params_i <= max_size[0]:
                    bottom_right_block_top_left = (right_block_top_left[0] + 1, right_block_top_left[1])
                    bottom_left_block = frame.raw[bottom_left_block_top_left[0]:bottom_left_block_top_left[0] + params_i, bottom_left_block_top_left[1]:bottom_left_block_top_left[1] + params_i].astype(float)
                    bottom_right_block = frame.raw[bottom_right_block_top_left[0]:bottom_right_block_top_left[0] + params_i, bottom_right_block_top_left[1]:bottom_right_block_top_left[1] + params_i].astype(float)
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
            top_left_block = frame.raw[top_left_block_top_left[0]:top_left_block_top_left[0] + params_i, top_left_block_top_left[1]:top_left_block_top_left[1] + params_i].astype(float)
            top_right_block = frame.raw[top_right_block_top_left[0]:top_right_block_top_left[0] + params_i, top_right_block_top_left[1]:top_right_block_top_left[1] + params_i].astype(float)
            bottom_left_block = frame.raw[bottom_left_block_top_left[0]:bottom_left_block_top_left[0] + params_i, bottom_left_block_top_left[1]:bottom_left_block_top_left[1] + params_i].astype(float)
            bottom_right_block = frame.raw[bottom_right_block_top_left[0]:bottom_right_block_top_left[0] + params_i, bottom_right_block_top_left[1]:bottom_right_block_top_left[1] + params_i].astype(float)
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
            if top_block_top_left[0] >= 0:
                top_block = frame.raw[top_block_top_left[0]:top_block_top_left[0] + params_i, top_block_top_left[1]:top_block_top_left[1] + params_i].astype(float)
                search_window = (top_block + current_block) / 2
                top_left = (top_left[0] - 0.5, top_left[1])
        elif location == 1:
            bottom_block_top_left = (int(top_left[0]) + 1, int(top_left[1]))
            if bottom_block_top_left[0] + params_i <= max_size[0]:
                bottom_block = frame.raw[bottom_block_top_left[0]:bottom_block_top_left[0] + params_i, bottom_block_top_left[1]:bottom_block_top_left[1] + params_i].astype(float)
                search_window = (bottom_block + current_block) / 2
                top_left = (top_left[0] + 0.5, top_left[1])
        elif location == 2:
            left_block_top_left = (int(top_left[0]), int(top_left[1]) - 1)
            if left_block_top_left[1] >= 0:
                left_block = frame.raw[left_block_top_left[0]:left_block_top_left[0] + params_i, left_block_top_left[1]:left_block_top_left[1] + params_i].astype(float)
                search_window = (left_block + current_block) / 2
                top_left = (top_left[0], top_left[1] - 0.5)
        elif location == 3:
            right_block_top_left = (int(top_left[0]), int(top_left[1]) + 1)
            if right_block_top_left[1] + params_i <= max_size[1]:
                right_block = frame.raw[right_block_top_left[0]:right_block_top_left[0] + params_i, right_block_top_left[1]:right_block_top_left[1] + params_i].astype(float)
                search_window = (right_block + current_block) / 2
                top_left = (top_left[0], top_left[1] + 0.5)
    return top_left, search_window

def calc_fast_motion_vector(block: np.ndarray, block_coor: tuple, frame: Frame, params_i, params, mvp) -> tuple:
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
    top_left, bottom_right = extend_block(block_coor, params_i, (0, 0, 0, 0), frame.shape)
    search_windows = []
    current_frame = frame
    while current_frame.prev is not None:
        search_window = current_frame.prev.raw[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
        search_windows.append(search_window)
        current_frame = current_frame.prev
    
    baseline_motion_vector, baseline_block = calc_full_range_motion_vector(block, block_coor, search_windows, top_left, params_i, False)
    min_motion_vector = baseline_motion_vector
    min_yx = block_coor
    min_block = baseline_block

    flag = False
    search_origin = (block_coor[0], block_coor[1])
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
            if current_coor[0] < 0 or current_coor[1] < 0 or current_coor[0] + params_i > frame.shape[0] or current_coor[1] + params_i > frame.shape[1]:
                continue
            top_left, bottom_right = extend_block(current_coor, params_i, (0, 0, 0, 0), frame.shape)
            search_windows = []
            current_frame = frame
            selection_flag = True
            while current_frame.prev is not None and selection_flag:
                if params.FMEEnable:
                    # center block's top left in interpolated frame
                    scaled_center_top_left = (new_coor[0] * 2, new_coor[1] * 2)
                    # search block's top left in interpolated frame
                    scaled_top_left = (float(scaled_center_top_left[0] + coors[coor_index][0]), float(scaled_center_top_left[1] + coors[coor_index][1]))
                    # actual top left in the true frame
                    top_left = (scaled_top_left[0] / 2, scaled_top_left[1] / 2)
                    # TODO: check if this is correct
                    top_left, search_window = get_interpolated_block(baseline_block, top_left, coor_index, current_frame.prev, params_i)
                    if search_window is None:
                        selection_flag = False
                else:
                    search_window = current_frame.prev.raw[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
                search_windows.append(search_window)
                current_frame = current_frame.prev
            
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

"""
    Calculate intra-frame prediction.
    No parallisim due to block dependency.

    Parameters:
        frame (Frame): The current frame.
        q_matrix (np.ndarray): The quantization matrix.

    Returns:
        qtc_block_dump (QTCFrame): The quantized transformed coefficients.
        predictor_dump (MotionVectorFrame): The predictor blocks.
        reconstructed_block_dump (Frame): The reconstructed blocks.
"""
def intraframe_prediction(frame: Frame, q_matrix: np.ndarray, params: Params) -> tuple:
    height, width = frame.shape
    block_frame = frame.pixel_to_block().astype(int)
    reconstructed_block_dump = Frame(frame.index, height, width, params_i=frame.params_i, data=np.empty(frame.shape, dtype=int))
    predictor_dump = MotionVectorFrame(is_intraframe=True, vbs_enable=params.VBSEnable)
    qtc_block_dump = QTCFrame(params_i=frame.params_i, vbs_enable=params.VBSEnable)
    y_counter = 0
    x_counter = 0
    prev_predictor = None
    split_counter = 0
    for y in range(0, height, frame.params_i):
        predictor_dump.new_row()
        qtc_block_dump.new_row()
        for x in range(0, width, frame.params_i):
            current_coor = (y, x)
            current_block = block_frame[y_counter, x_counter]
            hor_top_left, _ = extend_block(current_coor, frame.params_i, (0, 0, 0, 1), (height, width))
            ver_top_left, _ = extend_block(current_coor, frame.params_i, (1, 0, 0, 0), (height, width))
            
            # select vertical edge
            if hor_top_left[Intraframe.HORIZONTAL.value] == current_coor[Intraframe.HORIZONTAL.value]:
                hor_block = np.full((frame.params_i, 1), 128)
            else:
                hor_block = reconstructed_block_dump.raw[hor_top_left[0]:hor_top_left[0] + frame.params_i, hor_top_left[1]:hor_top_left[1] + 1]
            hor_block = hor_block.repeat(frame.params_i, Intraframe.HORIZONTAL.value)
            hor_mae = np.abs(hor_block - current_block).mean().astype(int)

            # select horizontal edge
            if ver_top_left[Intraframe.VERTICAL.value] == current_coor[Intraframe.VERTICAL.value]:
                ver_block = np.full((1, frame.params_i), 128)
            else:
                ver_block = reconstructed_block_dump.raw[ver_top_left[0]:ver_top_left[0] + 1, ver_top_left[1]:ver_top_left[1] + frame.params_i]
            ver_block = ver_block.repeat(frame.params_i, Intraframe.VERTICAL.value)
            ver_mae = np.abs(ver_block - current_block).mean().astype(int)

            predictor_block = None
            if ver_mae < hor_mae:
                current_predictor = MotionVector(Intraframe.VERTICAL.value, -1, mae=ver_mae)
                predictor_block = ver_block
            else:
                current_predictor = MotionVector(Intraframe.HORIZONTAL.value, -1, mae=hor_mae)
                predictor_block = hor_block

            qtc_block = QTCBlock(block=current_block - predictor_block, q_matrix=q_matrix)
            qtc_block.block_to_qtc()
            reconstructed_block = qtc_block.block + predictor_block
            diff_predictor = current_predictor - prev_predictor if prev_predictor is not None else current_predictor
            prev_predictor = current_predictor

            if params.VBSEnable:
                vbs_qtc_block, vbs_reconstructed_block, vbs_predictor = intraframe_vbs(current_coor, current_block, reconstructed_block, reconstructed_block_dump, qtc_block, diff_predictor, params)
                reconstructed_block = vbs_reconstructed_block
                if vbs_predictor is not None:
                    qtc_block = dict(
                        vbs=VBSMarker.SPLIT,
                        qtc_block=vbs_qtc_block,
                    )
                    current_predictor = dict(
                        vbs=VBSMarker.SPLIT,
                        predictor=vbs_predictor,
                    )
                    prev_predictor = vbs_predictor[-1]
                    split_counter += 1
                    print('vbs used in Frame', frame.index, current_coor)
                else:
                    qtc_block = dict(
                        vbs=VBSMarker.UNSPLIT,
                        qtc_block=qtc_block,
                    )
                    current_predictor = dict(
                        vbs=VBSMarker.UNSPLIT,
                        predictor=current_predictor,
                    )

            predictor_dump.append(current_predictor)
            qtc_block_dump.append(qtc_block)
            reconstructed_block_dump.raw[y_counter * frame.params_i:y_counter * frame.params_i + frame.params_i, x_counter * frame.params_i:x_counter * frame.params_i + frame.params_i] = reconstructed_block
            x_counter += 1
        y_counter += 1
        x_counter = 0
    return (qtc_block_dump, predictor_dump, reconstructed_block_dump, split_counter)

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
def mv_parallel_helper(index: int, frame: Frame, params: Params, q_matrix: np.ndarray, y: int) -> tuple:
    qtc_block_dump = []
    mv_dump = []
    reconstructed_block_dump = []
    prev_motion_vector = None
    split_counter = 0
    for x in range(0, frame.width, frame.params_i):
        centered_top_left = (y, x)
        centered_block = frame.raw[centered_top_left[0]:centered_top_left[0] + frame.params_i, centered_top_left[1]:centered_top_left[1] + frame.params_i]

        top_left = None
        search_windows = None

        if params.FastME:
            min_motion_vector, min_block = calc_fast_motion_vector(centered_block, centered_top_left, frame, frame.params_i, params, prev_motion_vector)
        else:
            top_left, bottom_right = extend_block(centered_top_left, frame.params_i, (params.r, params.r, params.r, params.r), frame.shape)
        
            current_frame = frame
            search_windows = []
            while current_frame.prev is not None:
                search_window = current_frame.prev.raw[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
                search_windows.append(search_window)
                current_frame = current_frame.prev
            min_motion_vector, min_block = calc_full_range_motion_vector(centered_block, centered_top_left, search_windows, top_left, frame.params_i, params.FMEEnable)

        qtc_block = QTCBlock(block=centered_block - min_block, q_matrix=q_matrix)
        qtc_block.block_to_qtc()
        reconstructed_block = qtc_block.block + min_block
        diff_mv = min_motion_vector - prev_motion_vector if prev_motion_vector is not None else min_motion_vector
        prev_motion_vector = min_motion_vector

        if params.VBSEnable:
            coor_offset = (centered_top_left[0] - top_left[0], centered_top_left[1] - top_left[1]) if top_left is not None else None
            vbs_qtc_block, vbs_reconstructed_block, vbs_mv = interframe_vbs(coor_offset, centered_block, search_windows, reconstructed_block, qtc_block, diff_mv, prev_motion_vector, frame, params)
            reconstructed_block = vbs_reconstructed_block
            if vbs_mv is not None:
                qtc_block = dict(
                    vbs=VBSMarker.SPLIT,
                    qtc_block=vbs_qtc_block,
                )
                min_motion_vector = dict(
                    vbs=VBSMarker.SPLIT,
                    predictor=vbs_mv,
                )
                prev_motion_vector = vbs_mv[-1]
                print('vbs used in Frame', frame.index, centered_top_left)
                split_counter += 1
            else:
                qtc_block = dict(
                    vbs=VBSMarker.UNSPLIT,
                    qtc_block=qtc_block,
                )
                min_motion_vector = dict(
                    vbs=VBSMarker.UNSPLIT,
                    predictor=min_motion_vector,
                )

        qtc_block_dump.append(qtc_block)
        reconstructed_block_dump.append(reconstructed_block)
        mv_dump.append(min_motion_vector)
    return (index, qtc_block_dump, mv_dump, reconstructed_block_dump, split_counter)

"""
    Calculate the motion vector for a block from the search window in parallel.

    Parameters:
        frame (Frame): The current frame.
        params_r (int): The search window size.
        q_matrix (np.ndarray): The quantization matrix.
        reconstructed_path (Path): The path to write the reconstructed frame to.
        pool (Pool): The pool of processes.

    Returns:
        frame (Frame): The current reconstructed frame.
        mv_dump (MotionVectorFrame): The motion vectors.
        qtc_block_dump (QTCFrame): The quantized transformed coefficients.
"""
def calc_motion_vector_parallel_helper(frame: Frame, params: Params, q_matrix: np.ndarray, reconstructed_path: Path, pool: Pool) -> tuple:
    print("Processing", frame.index)
    if (frame.is_intraframe is False and frame.prev is None) or ((frame.prev is not None) and (frame.prev.index + 1 != frame.index)):
        raise Exception('Frame index mismatch. Current: {}, Previous: {}'.format(frame.index, frame.prev.index))
    
    if frame.is_intraframe:
        qtc_block_dump, mv_dump, current_reconstructed_frame, split_counter = intraframe_prediction(frame, q_matrix, params)
    else:
        jobs = []
        results = []
        counter = 0
        for y in range(0, frame.shape[0], frame.params_i):
            job = pool.apply_async(func=mv_parallel_helper, args=(
                counter,
                frame, 
                params, 
                q_matrix,
                y,
            ))
            jobs.append(job)
            counter += 1

        for job in jobs:
            results.append(job.get())
        
        results.sort(key=lambda x: x[0])
        qtc_block_dump = QTCFrame(length=len(results), vbs_enable=params.VBSEnable)
        mv_dump = MotionVectorFrame(length=len(results), vbs_enable=params.VBSEnable, fme_enable=params.FMEEnable)
        reconstructed_block_dump = [None] * len(results)
        split_counter = 0
        for result in results:
            index = result[0]
            qtc_block_dump.append_list(index, result[1])
            mv_dump.append_list(index, result[2])
            reconstructed_block_dump[index] = result[3]
            split_counter += result[4]
        
        current_reconstructed_frame = Frame(frame=frame)
        current_reconstructed_frame.block_to_pixel(reconstructed_block_dump)

    current_reconstructed_frame.convert_within_range()

    current_reconstructed_frame.dump(reconstructed_path.joinpath('{}'.format(frame.index)))
    
    return current_reconstructed_frame, mv_dump, qtc_block_dump, split_counter
