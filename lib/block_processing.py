from typing import List
import numpy as np
from multiprocessing import Pool, Queue
from pathlib import Path

from lib.config.config import Params, Config
from lib.components.frame import Frame, extend_block
from lib.components.qtc import QTCFrame, quantization_matrix
from lib.components.mv import MotionVectorFrame
from lib.enums import VBSMarker
from lib.predictions.intraframe import intraframe_prediction, intraframe_prediction_mode0
from lib.predictions.interframe import interframe_prediction, interframe_block_prediction
from qp_bitcount import CIF_bitcount_perRow_p, QCIF_bitcount_perRow_p, CIF_bitcount_perRow_i, QCIF_bitcount_perRow_i

def get_next_dispatchable_block(dispatched_array, shape, params_i):
    """
        Get next processing block for intraframe prediction
    """
    dispatchable_list = []
    invalidate_list = []
    true_array = np.argwhere(dispatched_array == True)
    if len(true_array) == 0:
        dispatchable_list.append(dict(
            left=None,
            top=None,
            coor=(0, 0)
        ))
    else:
        for item in true_array:
            item = tuple(item)
            y = item[0] * params_i
            x = item[1] * params_i
            is_right_appended = False
            is_bottom_appended = False
            right_coor = (y, x + params_i)
            bottom_coor = (y  + params_i, x)
            if right_coor[1] < shape[1]:
                    # inside the frame
                    if right_coor[0] == 0:
                        # does not require a top block
                        is_right_appended = True
                        dispatchable_list.append(dict(
                            left=item,
                            top=None,
                            coor=(right_coor[0] // params_i, right_coor[1] // params_i)
                        ))
                    else:
                        # require a top block
                        top_block_coor = (right_coor[0] - params_i, right_coor[1])
                        top_block_index = (top_block_coor[0] // params_i, top_block_coor[1] // params_i)
                        top_block_status = dispatched_array[top_block_index]
                        if top_block_status:
                            # top block is dispatched
                            is_right_appended = True
                            dispatchable_list.append(dict(
                                left=item,
                                top=top_block_index,
                                coor=(right_coor[0] // params_i, right_coor[1] // params_i)
                            ))
            else:
                # outside the frame
                is_right_appended = True
            if bottom_coor[0] < shape[0]:
                # inside the frame
                if bottom_coor[1] == 0:
                    # does not require a left block
                    is_bottom_appended = True
                    dispatchable_list.append(dict(
                        left=None,
                        top=item,
                        coor=(bottom_coor[0] // params_i, bottom_coor[1] // params_i)
                    ))
                else:
                    # require a left block
                    left_block_coor = (bottom_coor[0], bottom_coor[1] - params_i)
                    left_block_index = (left_block_coor[0] // params_i, left_block_coor[1] // params_i)
                    left_block_status = dispatched_array[left_block_index ]
                    if left_block_status:
                        # left block is dispatched
                        is_bottom_appended = True
                        dispatchable_list.append(dict(
                            left=left_block_index,
                            top=item,
                            coor=(bottom_coor[0] // params_i, bottom_coor[1] // params_i)
                        ))
            else:
                # outside the frame
                is_bottom_appended = True
            if is_right_appended and is_bottom_appended:
                invalidate_list.append(item)
    dispatchable_list = [dict(t) for t in set(tuple(d.items()) for d in dispatchable_list)]
    return dispatchable_list, invalidate_list

def processing(frame: Frame, params: Params, q_matrix: np.ndarray, reconstructed_path: Path, pool: Pool, pass_num=1, per_block_row_bit_count=[]) -> tuple:
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
    print("Processing", frame.index)
    if (frame.is_intraframe is False and frame.prev is None) or ((frame.prev is not None) and (frame.prev.index + 1 != frame.index)):
        raise Exception('Frame index mismatch. Current: {}, Previous: {}'.format(frame.index, frame.prev.index))
    current_reconstructed_frame = None
    if frame.is_intraframe:
        if params.ParallelMode == 2:
            row_block_no = frame.shape[0] // frame.params_i
            col_block_no = frame.shape[1] // frame.params_i
            total_blocks = row_block_no * col_block_no
            finished_blocks = 0
            dispatched_array = np.zeros((row_block_no, col_block_no)).astype(bool)
            reconstructed_block_dump = [[None] * col_block_no for _ in range(row_block_no)]
            qtc_block_dump = QTCFrame(shape=(row_block_no, col_block_no), vbs_enable=params.VBSEnable)
            mv_dump = MotionVectorFrame(shape=(row_block_no, col_block_no), vbs_enable=params.VBSEnable, is_intraframe=frame.is_intraframe)
            split_counter = 0
            qp_rc = None
            qp_rc_initial = None
            bitcount_per_iter = 0
            perframeBR_remain = 0
            remain_blocks = 0
            bitcount_per_frame = 0

            if params.RCflag != 0:
                initial_perframeBR = params.perframeBR
                bitbudgetPerRow = params.bitbudgetPerRow
                bitbudgetPerBlock = bitbudgetPerRow / col_block_no
                if frame.height == 288 and frame.width == 352:
                    table = CIF_bitcount_perRow_i
                elif frame.height == 144 and frame.width == 176:
                    table = QCIF_bitcount_perRow_i
                for index, value in table.items():
                    value_perblock = value/col_block_no
                    if value_perblock <= bitbudgetPerBlock:
                        qp_rc_initial = index
                        break
            
            while finished_blocks < total_blocks:
                jobs = []
                dispatchable_list, invalidate_list = get_next_dispatchable_block(dispatched_array, frame.shape, frame.params_i)
                if params.RCflag != 0:
                    if len(dispatchable_list) == 1:
                        perframeBR_remain = initial_perframeBR
                        qp_rc = qp_rc_initial
                        q_matrix = quantization_matrix(params.i, qp_rc)
                    else:
                        remain_blocks = total_blocks - finished_blocks
                        bitbudgetPerBlock = perframeBR_remain/remain_blocks
                        bitbudgetPerRow = bitbudgetPerBlock * len(dispatchable_list)
                        for index, value in table.items():
                            value_perblock = value/col_block_no
                            value_perlist = value_perblock * len(dispatchable_list)
                            if value_perlist <= bitbudgetPerRow:
                                qp_rc = index
                                break
                        q_matrix = quantization_matrix(params.i, qp_rc)
                for item in dispatchable_list:
                    left_coor_index = item['left']
                    top_coor_index = item['top']
                    coor_index = item['coor']
                    coor = (coor_index[0] * frame.params_i, coor_index[1] * frame.params_i)
                    left_block = reconstructed_block_dump[left_coor_index[0]][left_coor_index[1]] if left_coor_index is not None else None
                    top_block = reconstructed_block_dump[top_coor_index[0]][top_coor_index[1]] if top_coor_index is not None else None
                    current_block = frame.raw[coor[0]:coor[0] + frame.params_i, coor[1]:coor[1] + frame.params_i]
                    if params.VBSEnable:
                        prev_predictor = mv_dump.raw[left_coor_index[0]][left_coor_index[1]] if left_coor_index is not None else None
                        if prev_predictor is not None:
                            if prev_predictor['vbs'] is VBSMarker.SPLIT:
                                prev_predictor = prev_predictor['predictor'][-1]
                            else:
                                prev_predictor = prev_predictor['predictor']
                    else:
                        prev_predictor = mv_dump.raw[left_coor_index[0]][left_coor_index[1]] if left_coor_index is not None else None
                    job = pool.apply_async(func=intraframe_prediction, args=(
                        frame.index,
                        dict(
                            left=left_coor_index,
                            top=top_coor_index,
                            current=coor_index
                        ),
                        dict(
                            left=left_block,
                            top=top_block,
                            current=current_block
                        ),
                        q_matrix,
                        params,
                        prev_predictor,
                        qp_rc
                    ))
                    jobs.append(job)
                for item in invalidate_list:
                    dispatched_array[item] = False
                for job in jobs: 
                    result = job.get()
                    current_coor = result[0]
                    dispatched_array[current_coor] = True
                    reconstructed_block_dump[current_coor[0]][current_coor[1]] = result[3]
                    qtc_block_dump.set(current_coor, result[1])
                    mv_dump.set(current_coor, result[2])
                    split_counter += result[4]
                    finished_blocks += 1
                    bitcount_per_iter += result[5]
                if params.RCflag != 0:
                    perframeBR_remain -= bitcount_per_iter
                bitcount_per_frame += bitcount_per_iter
            bitcount_per_row = bitcount_per_frame/row_block_no

        elif params.ParallelMode == 0:
            qtc_block_dump, mv_dump, current_reconstructed_frame, split_counter, row_number, bitcount_per_frame, per_block_row_bit_count = intraframe_prediction_mode0(frame, q_matrix, params, None, pass_num, per_block_row_bit_count)
            bitcount_per_row = bitcount_per_frame/row_number

    elif params.ParallelMode == 2 or params.ParallelMode == 0:
        jobs = []
        results = []
        counter = 0
        bitcount_per_frame = 0
        table = None
        qp_rc = None
        if params.RCflag != 0:
            bitbudgetPerRow = params.bitbudgetPerRow
            if frame.height == 288 and frame.width == 352:
                table = CIF_bitcount_perRow_p
            elif frame.height == 144 and frame.width == 176:
                table = QCIF_bitcount_perRow_p
            for index, value in table.items():
                if value <= bitbudgetPerRow:
                    qp_rc = index
                    break
            q_matrix = quantization_matrix(params.i, qp_rc)
        for y in range(0, frame.shape[0], frame.params_i):
            job = pool.apply_async(func=interframe_prediction, args=(
                counter,
                frame, 
                params, 
                q_matrix,
                y,
                qp_rc
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
            bitcount_per_frame += result[5]
        bitcount_per_row = bitcount_per_frame/counter
    elif params.ParallelMode == 1:
        jobs = []
        results = []
        split_counter = 0
        bitcount_per_frame = 0
        table = None
        qp_rc = None
        row_block_no = frame.height // frame.params_i
        col_block_no = frame.width // frame.params_i
        if params.RCflag != 0:
            bitbudgetPerRow = params.bitbudgetPerRow
            bitbudgetPerBlock = bitbudgetPerRow / col_block_no
            if frame.height == 288 and frame.width == 352:
                table = CIF_bitcount_perRow_p
            elif frame.height == 144 and frame.width == 176:
                table = QCIF_bitcount_perRow_p
            for index, value in table.items():
                value_perblock = value/col_block_no
                if  value_perblock <= bitbudgetPerBlock:
                    qp_rc = index
                    break
            q_matrix = quantization_matrix(params.i, qp_rc)
        reconstructed_block_dump = [[None] * col_block_no for _ in range(row_block_no)]
        qtc_block_dump = QTCFrame(shape=(row_block_no, col_block_no), vbs_enable=params.VBSEnable)
        mv_dump = MotionVectorFrame(shape=(row_block_no, col_block_no), vbs_enable=params.VBSEnable, fme_enable=params.FMEEnable)  
        for y in range(0, frame.height, frame.params_i):
            for x in range(0, frame.width, frame.params_i):
                job = pool.apply_async(func=interframe_block_prediction, args=(
                    (y, x),
                    frame,
                    params,
                    q_matrix,
                    None,
                    None,
                    qp_rc
                ))
                jobs.append(job)
        
        for job in jobs:
            result = job.get()
            current_coor = (result[0][0] // frame.params_i, result[0][1] // frame.params_i)
            qtc_block_dump.set(current_coor, result[1])
            mv_dump.set(current_coor, result[2])
            reconstructed_block_dump[current_coor[0]][current_coor[1]] = result[3]
            split_counter += result[4]
            bitcount_per_frame += result[5]
        
        bitcount_per_row = bitcount_per_frame / row_block_no
    
    if current_reconstructed_frame is None:
        current_reconstructed_frame = Frame(frame=frame)
        current_reconstructed_frame.block_to_pixel(reconstructed_block_dump)

    current_reconstructed_frame.convert_within_range()
    current_reconstructed_frame.dump(reconstructed_path.joinpath('{}'.format(frame.index)))

    print(f"frame index: {frame.index}, bit count: {bitcount_per_frame}")
    
    return current_reconstructed_frame, mv_dump, qtc_block_dump, split_counter, bitcount_per_row, bitcount_per_frame, per_block_row_bit_count 

def processing_mode3(frame: Frame, config: Config, q_matrix: np.ndarray, reconstructed_path: Path, prev_data_queue: Queue, next_data_queue: Queue, write_data_func: callable):
    """
        Calculate the motion vector for a block from the search window in parallel mode 3.

        Parameters:
            frame (Frame): The current frame.
            params_r (int): The search window size.
            q_matrix (np.ndarray): The quantization matrix.
            reconstructed_path (Path): The path to write the reconstructed frame to.
            prev_data_queue (Queue): The queue of previous data.
            current_data_queue (Queue): The queue of current data.

        Returns:
            frame (Frame): The current reconstructed frame.
            mv_dump (MotionVectorFrame): The motion vectors.
            qtc_block_dump (QTCFrame): The quantized transformed coefficients.
    """
    print("Processing", frame.index)
    params = config.params
    current_reconstructed_frame = None
    bitcount_per_frame = 0
    y_counter = 0
    if frame.is_intraframe:
        qtc_block_dump, mv_dump, current_reconstructed_frame, split_counter, _, _ = intraframe_prediction_mode0(frame, q_matrix, params, next_data_queue)
    else:
        empty_frame = Frame(-1, frame.height, frame.width, params_i=params.i, data=np.full(frame.height*frame.width, 256).reshape(frame.height, frame.width).astype(np.uint16))
        frame.prev = empty_frame.copy()
        row_block_no = frame.height // frame.params_i
        col_block_no = frame.width // frame.params_i
        reconstructed_block_dump = [[None] * col_block_no for _ in range(row_block_no)]
        qtc_block_dump = QTCFrame(shape=(row_block_no, col_block_no), vbs_enable=params.VBSEnable)
        mv_dump = MotionVectorFrame(shape=(row_block_no, col_block_no), vbs_enable=params.VBSEnable, fme_enable=params.FMEEnable)
        split_counter = 0
        total_rows = frame.height/params.i
        table = None
        if params.RCflag != 0:
            initial_perframeBR = params.perframeBR
            initial_bitbudgetPerRow = params.bitbudgetPerRow
            if frame.height == 288 and frame.width == 352:
                table = CIF_bitcount_perRow_p
            elif frame.height == 144 and frame.width == 176:
                table = QCIF_bitcount_perRow_p
        for y in range(0, frame.height, frame.params_i):
            # clear prev_motion_vector when a row is finished
            prev_motion_vector = None
            if params.RCflag != 0:
                if y == 0:
                    bitbudgetPerRow = initial_bitbudgetPerRow
                    perframeBR_remain = initial_perframeBR
                else:
                    perframeBR_remain = perframeBR_remain - bitcount_per_frame 
                    rows_remain = total_rows - y_counter 
                    bitbudgetPerRow = perframeBR_remain / rows_remain
                for index, value in table.items():
                    if value <= bitbudgetPerRow:
                        qp_rc = index
                        break
                q_matrix = quantization_matrix(params.i, qp_rc) 
            for x in range(0, frame.width, frame.params_i):
                current_coor = (y, x)
                search_window_top_left, search_window_bottom_right = extend_block(current_coor, frame.params_i, (params.r, params.r, params.r, params.r), frame.shape)
                current_search_window = frame.prev.raw[search_window_top_left[0]:search_window_bottom_right[0], search_window_top_left[1]:search_window_bottom_right[1]]
                while np.isin(256, current_search_window):
                    prev_reconstructed_frame_data = prev_data_queue.get()
                    reconstructed_blocks = prev_reconstructed_frame_data[1]
                    current_frame = frame
                    for reconstructed_block_index in range(len(reconstructed_blocks)):
                        reconstructed_block = reconstructed_blocks[reconstructed_block_index]
                        if current_frame.prev is None:
                            current_frame.prev = empty_frame.copy()
                        current_frame.prev.set(prev_reconstructed_frame_data[0], reconstructed_block)
                        current_frame = current_frame.prev
                        if reconstructed_block_index + 1 == params.nRefFrames:
                            break
                # block is processable as the search window is filled
                if params.RCflag != 0:
                    current_coor, qtc_block, min_motion_vector, reconstructed_block, current_split_counter, bitcount_per_block = interframe_block_prediction(current_coor, frame, params, q_matrix, prev_motion_vector, next_data_queue, qp_rc)
                else:
                    current_coor, qtc_block, min_motion_vector, reconstructed_block, current_split_counter, bitcount_per_block = interframe_block_prediction(current_coor, frame, params, q_matrix, prev_motion_vector, next_data_queue, params.qp)
                if current_split_counter > 0:
                    split_counter += current_split_counter
                if params.VBSEnable:
                    if min_motion_vector['vbs'] == VBSMarker.SPLIT:
                        prev_motion_vector = min_motion_vector['predictor'][-1]
                    else:
                        prev_motion_vector = min_motion_vector['predictor']
                else:
                    prev_motion_vector = min_motion_vector
                current_coor_index = (current_coor[0] // frame.params_i, current_coor[1] // frame.params_i)
                reconstructed_block_dump[current_coor_index[0]][current_coor_index[1]] = reconstructed_block
                qtc_block_dump.set(current_coor_index, qtc_block)
                mv_dump.set(current_coor_index, min_motion_vector)
                bitcount_per_frame += bitcount_per_block
                # print("done", current_coor, frame.index)
            y_counter += 1
            
    
    if current_reconstructed_frame is None:
        current_reconstructed_frame = Frame(frame=frame)
        current_reconstructed_frame.block_to_pixel(reconstructed_block_dump)

    current_reconstructed_frame.convert_within_range()
    current_reconstructed_frame.dump(reconstructed_path.joinpath('{}'.format(frame.index)))
    
    print("Frame {} reconstructed".format(frame.index))
    write_data_func((frame.index, mv_dump, qtc_block_dump), config)
