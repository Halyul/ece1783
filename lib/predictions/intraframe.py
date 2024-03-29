import numpy as np
from lib.config.config import Params
from lib.enums import Intraframe
from lib.components.frame import Frame, extend_block, convert_within_range
from lib.components.qtc import QTCBlock, QTCFrame, quantization_matrix
from lib.components.mv import MotionVector, MotionVectorFrame
from lib.enums import VBSMarker
from lib.predictions.misc import rdo
from multiprocessing import Queue
from qp_bitcount import CIF_bitcount_perRow_i, QCIF_bitcount_perRow_i

def intraframe_vbs(reconstructed_block: np.ndarray, block_dict, qtc_block: QTCBlock, diff_predictor: int, params: Params, qp_rc_vbs = None):
    original_block = block_dict['current']

    if params.RCflag != 0:
        block_rdo_cost = rdo(original_block, reconstructed_block, qtc_block, diff_predictor, qp_rc_vbs, is_intraframe=True)
    else:
        block_rdo_cost = rdo(original_block, reconstructed_block, qtc_block, diff_predictor, params.qp, is_intraframe=True)
    subblock_params_i = params.i // 2
    if params.RCflag != 0:
        qp = qp_rc_vbs - 1 if qp_rc_vbs > 0 else 0
        q_matrix = quantization_matrix(subblock_params_i, qp)
    else:
        qp = params.qp - 1 if params.qp > 0 else 0
        q_matrix = quantization_matrix(subblock_params_i, qp)
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
        
        # select vertical edge
        if centered_top_left_index == 0 or centered_top_left_index == 2:
            # top left or bottom left
            left_block = block_dict['left']
            if left_block is None:
                hor_block = np.full((subblock_params_i, 1), 128)
            else:
                if centered_top_left_index == 0:
                    hor_block = left_block[:, -1][:subblock_params_i].reshape(subblock_params_i, 1)
                else:
                    hor_block = left_block[:, -1][subblock_params_i:].reshape(subblock_params_i, 1)
        else:
            # top right or bottom right
            hor_top_left, _ = extend_block(centered_top_left, subblock_params_i, (0, 0, 0, 1), original_block.shape)
            hor_block = reconstructed_subblocks[hor_top_left[0]:hor_top_left[0] + subblock_params_i, hor_top_left[1]:hor_top_left[1] + 1]
        hor_block = hor_block.repeat(subblock_params_i, Intraframe.HORIZONTAL.value)
        hor_mae = np.abs(hor_block - current_block).mean()

        # select horizontal edge
        if centered_top_left_index == 0 or centered_top_left_index == 1:
            # top left or top right
            top_block = block_dict['top']
            if top_block is None:
                ver_block = np.full((1, subblock_params_i), 128)
            else:
                if centered_top_left_index == 0:
                    ver_block = top_block[-1, :][:subblock_params_i].reshape(1, subblock_params_i)
                else:
                    ver_block = top_block[-1, :][subblock_params_i:].reshape(1, subblock_params_i)
        else:
            # bottom left or bottom right
            ver_top_left, _ = extend_block(centered_top_left, subblock_params_i, (1, 0, 0, 0), original_block.shape)
            ver_block = reconstructed_subblocks[ver_top_left[0]:ver_top_left[0] + 1, ver_top_left[1]:ver_top_left[1] + subblock_params_i]
        ver_block = ver_block.repeat(subblock_params_i, Intraframe.VERTICAL.value)
        ver_mae = np.abs(ver_block - current_block).mean()
        predictor_block = None
        if ver_mae < hor_mae:
            current_predictor = MotionVector(Intraframe.VERTICAL.value, -1, mae=ver_mae)
            predictor_block = ver_block
        else:
            current_predictor = MotionVector(Intraframe.HORIZONTAL.value, -1, mae=hor_mae)
            predictor_block = hor_block
        qtc_subblock = QTCBlock(block=current_block - predictor_block, q_matrix=q_matrix, qp=qp)
        qtc_subblock.block_to_qtc()
        residual_subblocks.append(qtc_subblock.block)
        reconstructed_subblock = qtc_subblock.block + predictor_block
        diff_subpredictor = current_predictor - prev_predictor if prev_predictor is not None else current_predictor
        prev_predictor = current_predictor
        subblock_rdo_cost += rdo(current_block, reconstructed_subblock, qtc_subblock, diff_subpredictor, qp, is_intraframe=True)
        qtc_subblocks.append(qtc_subblock.qtc_block)
        reconstructed_subblocks[centered_top_left[0]:centered_top_left[0] + subblock_params_i, centered_top_left[1]:centered_top_left[1] + subblock_params_i] = reconstructed_subblock
        subpredictor_dump.append(current_predictor)

    if subblock_rdo_cost < block_rdo_cost:
        qtc_stack = np.concatenate((np.concatenate((qtc_subblocks[0], qtc_subblocks[1]), axis=1), np.concatenate((qtc_subblocks[2], qtc_subblocks[3]), axis=1)), axis=0)
        temp_stack = np.concatenate((np.concatenate((residual_subblocks[0], residual_subblocks[1]), axis=1), np.concatenate((residual_subblocks[2], residual_subblocks[3]), axis=1)), axis=0)
        qtc_block = QTCBlock(qtc_block=qtc_stack, block=temp_stack, qp = qp)
        return qtc_block, reconstructed_subblocks, subpredictor_dump
    else:
        return qtc_block, reconstructed_block, None

def intraframe_prediction(index, coor_dict, block_dict, q_matrix: np.ndarray, params: Params, prev_predictor,qp = None) -> tuple:
    current_coor = coor_dict['current']
    current_block = block_dict['current']
    left_coor = coor_dict['left']
    top_coor = coor_dict['top']
    split_counter = 0
    mv_integrate = ''
    # select vertical edge
    if left_coor is None:
        hor_block = np.full((params.i, 1), 128)
    else:
        hor_block = block_dict['left'][:, -1].reshape(params.i, 1)
    hor_block = hor_block.repeat(params.i, Intraframe.HORIZONTAL.value)
    hor_mae = np.abs(hor_block - current_block).mean()

    # select horizontal edge
    if top_coor is None:
        ver_block = np.full((1, params.i), 128)
    else:
        ver_block = block_dict['top'][-1, :].reshape(1, params.i)
    ver_block = ver_block.repeat(params.i, Intraframe.VERTICAL.value)
    ver_mae = np.abs(ver_block - current_block).mean()

    predictor_block = None
    if ver_mae < hor_mae:
        current_predictor = MotionVector(Intraframe.VERTICAL.value, -1, mae=ver_mae)
        predictor_block = ver_block
    else:
        current_predictor = MotionVector(Intraframe.HORIZONTAL.value, -1, mae=hor_mae)
        predictor_block = hor_block

    if params.RCflag !=0:
        qtc_block = QTCBlock(block=current_block - predictor_block, q_matrix=q_matrix, qp = qp)
    else:
        qtc_block = QTCBlock(block=current_block - predictor_block, q_matrix=q_matrix, qp = params.qp)

    qtc_block.block_to_qtc()
    reconstructed_block = qtc_block.block + predictor_block
    diff_predictor = current_predictor - prev_predictor if prev_predictor is not None else current_predictor

    bitcount_per_block = len(qtc_block.to_str()) + len(current_predictor.to_str(is_intraframe=True))

    if params.VBSEnable:
        if params.RCflag != 0:
            vbs_qtc_block, vbs_reconstructed_block, vbs_predictor = intraframe_vbs(reconstructed_block, block_dict, qtc_block, diff_predictor, params, qp_rc_vbs=qp)
        else:  
            vbs_qtc_block, vbs_reconstructed_block, vbs_predictor = intraframe_vbs(reconstructed_block, block_dict, qtc_block, diff_predictor, params, qp_rc_vbs= params.qp)
        reconstructed_block = vbs_reconstructed_block
        if vbs_predictor is not None:
            for mv in vbs_predictor:
                mv_integrate += mv.to_str(is_intraframe=True)
                vbs_mv_length = len(mv_integrate)
            bitcount_per_block = len(vbs_qtc_block.to_str()) + vbs_mv_length
            qtc_block = dict(
                vbs=VBSMarker.SPLIT,
                qtc_block=vbs_qtc_block,
            )
            current_predictor = dict(
                vbs=VBSMarker.SPLIT,
                predictor=vbs_predictor,
            )
            split_counter += 1
            print('vbs used in Frame', index, (current_coor[0] * params.i, current_coor[1] * params.i))
        else:
            bitcount_per_block = len(qtc_block.to_str()) + len(current_predictor.to_str(is_intraframe=True))
            qtc_block = dict(
                vbs=VBSMarker.UNSPLIT,
                qtc_block=qtc_block,
            )
            current_predictor = dict(
                vbs=VBSMarker.UNSPLIT,
                predictor=current_predictor,
            )
    return coor_dict['current'], qtc_block, current_predictor, reconstructed_block, split_counter, bitcount_per_block

def intraframe_prediction_mode0(frame: Frame, q_matrix: np.ndarray, params: Params, buffer, data_queue: Queue = None, pass_num=1, per_block_row_bit_count=[], scale_factor=1) -> tuple:
    """
        Calculate intra-frame prediction.
        No parallisim due to block dependency.

        Parameters:
            frame (Frame): The current frame.
            q_matrix (np.ndarray): The quantization matrix.
            params (Params): The parameters.
            data_queue (Queue): The queue to store the data.

        Returns:
            qtc_block_dump (QTCFrame): The quantized transformed coefficients.
            predictor_dump (MotionVectorFrame): The predictor blocks.
            reconstructed_block_dump (Frame): The reconstructed blocks.
    """
    height, width = frame.shape
    block_frame = frame.pixel_to_block().astype(int)
    reconstructed_block_dump = Frame(frame.index, height, width, params_i=frame.params_i, data=np.empty(frame.shape, dtype=int))
    predictor_dump = MotionVectorFrame(is_intraframe=True, vbs_enable=params.VBSEnable)
    qtc_block_dump = QTCFrame(params_i=frame.params_i, vbs_enable=params.VBSEnable)
    y_counter = 0
    x_counter = 0
    mv_integrate = ''
    prev_predictor = None
    split_counter = 0
    bitcount_per_frame = 0
    total_rows = height/params.i
    table = None
    qp_rc = None
    if params.RCflag != 0:
        initial_perframeBR = params.perframeBR
        initial_bitbudgetPerRow = params.bitbudgetPerRow
        if height == 288 and width == 352:
            table = CIF_bitcount_perRow_i
        elif height == 144 and width == 176:
            table = QCIF_bitcount_perRow_i

    for y in range(0, height, frame.params_i):
        predictor_dump.new_row()
        qtc_block_dump.new_row()
        bitcount_per_row = 0
        if params.RCflag != 0:
            if y == 0:
                bitbudgetPerRow = initial_bitbudgetPerRow
                perframeBR_remain = initial_perframeBR
            else:
                perframeBR_remain = perframeBR_remain - bitcount_per_frame
                rows_remain = total_rows - y_counter 
                bitbudgetPerRow = perframeBR_remain / rows_remain
            if pass_num == 2:
                bit_budget_ratio = per_block_row_bit_count[y_counter]
                bitbudgetPerRow = params.perframeBR * bit_budget_ratio
            for index, value in table.items():
                if value * scale_factor <= bitbudgetPerRow:
                    qp_rc = index
                    break
            if qp_rc == None:
                qp_rc = 11

        for x in range(0, width, frame.params_i):
            current_coor = (y, x)
            current_block = block_frame[y_counter, x_counter]
            hor_top_left, _ = extend_block(current_coor, frame.params_i, (0, 0, 0, 1), (height, width))
            ver_top_left, _ = extend_block(current_coor, frame.params_i, (1, 0, 0, 0), (height, width))

            if params.RCflag != 0:
                q_matrix = quantization_matrix(params.i, qp_rc)
            
            # select vertical edge
            if hor_top_left[Intraframe.HORIZONTAL.value] == current_coor[Intraframe.HORIZONTAL.value]:
                hor_block = np.full((frame.params_i, 1), 128)
            else:
                hor_block = reconstructed_block_dump.raw[hor_top_left[0]:hor_top_left[0] + frame.params_i, hor_top_left[1]:hor_top_left[1] + 1]
            hor_block = hor_block.repeat(frame.params_i, Intraframe.HORIZONTAL.value)
            hor_mae = np.abs(hor_block - current_block).mean()

            # select horizontal edge
            if ver_top_left[Intraframe.VERTICAL.value] == current_coor[Intraframe.VERTICAL.value]:
                ver_block = np.full((1, frame.params_i), 128)
            else:
                ver_block = reconstructed_block_dump.raw[ver_top_left[0]:ver_top_left[0] + 1, ver_top_left[1]:ver_top_left[1] + frame.params_i]
            ver_block = ver_block.repeat(frame.params_i, Intraframe.VERTICAL.value)
            ver_mae = np.abs(ver_block - current_block).mean()

            predictor_block = None
            if ver_mae < hor_mae:
                current_predictor = MotionVector(Intraframe.VERTICAL.value, -1, mae=ver_mae)
                predictor_block = ver_block
            else:
                current_predictor = MotionVector(Intraframe.HORIZONTAL.value, -1, mae=hor_mae)
                predictor_block = hor_block

            if params.RCflag != 0:
                qtc_block = QTCBlock(block=current_block - predictor_block, q_matrix=q_matrix, qp= qp_rc)
            else:    
                qtc_block = QTCBlock(block=current_block - predictor_block, q_matrix=q_matrix, qp= params.qp)
            
            qtc_block.block_to_qtc()
            reconstructed_block = qtc_block.block + predictor_block
            diff_predictor = current_predictor - prev_predictor if prev_predictor is not None else current_predictor
            prev_predictor = current_predictor
            bitcount_per_block = len(qtc_block.to_str()) + len(current_predictor.to_str(is_intraframe=True))

            vbsed = False
            buffer_state = False
            if buffer.state('intraframe_prediction_mode0_vbs', current_coor):
                # buffer state exists
                buffer_state = True
                vbsed = buffer.get('intraframe_prediction_mode0_vbs', current_coor)

            if not buffer_state:
                vbsed = True

            if params.VBSEnable:
                if vbsed:
                    if params.RCflag != 0:
                        vbs_qtc_block, vbs_reconstructed_block, vbs_predictor = intraframe_vbs(reconstructed_block, dict(
                        current=current_block,
                        left=hor_block,
                        top=ver_block,
                        ), qtc_block, diff_predictor, params, qp_rc)
                    else:  
                        vbs_qtc_block, vbs_reconstructed_block, vbs_predictor = intraframe_vbs(reconstructed_block, dict(
                        current=current_block,
                        left=hor_block,
                        top=ver_block,
                        ), qtc_block, diff_predictor, params)
                    reconstructed_block = vbs_reconstructed_block
                    if vbs_predictor is not None:
                        for mv in vbs_predictor:
                            mv_integrate += mv.to_str(is_intraframe=True)
                        vbs_mv_length = len(mv_integrate)
                        bitcount_per_block = len(vbs_qtc_block.to_str()) + vbs_mv_length
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
                        if not buffer_state:
                            buffer.add('intraframe_prediction_mode0_vbs', current_coor, True)
                    else:
                        bitcount_per_block = len(qtc_block.to_str()) + len(current_predictor.to_str(is_intraframe=True))
                        qtc_block = dict(
                            vbs=VBSMarker.UNSPLIT,
                            qtc_block=qtc_block,
                        )
                        current_predictor = dict(
                            vbs=VBSMarker.UNSPLIT,
                            predictor=current_predictor,
                        )
                        if not buffer_state:
                            buffer.add('intraframe_prediction_mode0_vbs', current_coor, False)
                else:
                    bitcount_per_block = len(qtc_block.to_str()) + len(current_predictor.to_str(is_intraframe=True))
                    qtc_block = dict(
                        vbs=VBSMarker.UNSPLIT,
                        qtc_block=qtc_block,
                    )
                    current_predictor = dict(
                        vbs=VBSMarker.UNSPLIT,
                        predictor=current_predictor,
                    )
            bitcount_per_row += bitcount_per_block
            predictor_dump.append(current_predictor)
            qtc_block_dump.append(qtc_block)
            reconstructed_block_dump.raw[y_counter * frame.params_i:y_counter * frame.params_i + frame.params_i, x_counter * frame.params_i:x_counter * frame.params_i + frame.params_i] = reconstructed_block
            if data_queue is not None:
                data_queue.put((current_coor, [convert_within_range(reconstructed_block)]))
            x_counter += 1
        y_counter += 1
        per_block_row_bit_count.append(bitcount_per_row)
        bitcount_per_frame += bitcount_per_row
        x_counter = 0
    per_block_row_bit_count = [item / bitcount_per_frame for item in per_block_row_bit_count]
    return (qtc_block_dump, predictor_dump, reconstructed_block_dump, split_counter, y_counter, bitcount_per_frame, per_block_row_bit_count)