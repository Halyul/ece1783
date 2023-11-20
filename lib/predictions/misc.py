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