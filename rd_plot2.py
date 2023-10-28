from lib.yuv_processor import YUVProcessor
import time
import numpy as np
from lib.config.config import Config
from lib.signal_processing import psnr
import matplotlib.pyplot as plt
import math

if __name__ == '__main__':
    config = Config('config.yaml')
    mv_path = config.output_path.mv_folder
    residual_path = config.output_path.residual_folder
    original_path = config.output_path.original_folder
    reconstructed_path = config.output_path.reconstructed_folder
    video_name = config.input.split('/')[-1]
    video_path = config.statistics.path.joinpath(video_name)
    params_i_list = [8, 16]
    params_qp_list = [3, 4]
    params_i_period_list = [1, 4, 10]
    params_r = 2
    stop_at = 10
    results = []

    for params_i_index in range(len(params_i_list)):
        results.append([])
        params_i = params_i_list[params_i_index]
        params_qp = params_qp_list[params_i_index]
        for params_i_period in params_i_period_list:
            print("current run: i={}, qp={}, i_period={}".format(params_i, params_qp, params_i_period))
            start = time.time()
            reader = YUVProcessor('config.yaml', 
                                config_override=dict(
                                    params=dict(
                                        i=params_i,
                                        r=params_r,
                                        i_period=params_i_period,
                                        qp=params_qp,
                                        stop_at=stop_at,
                                    )
                                )
                                )
            end = time.time()
            size_array = []
            for frame_index in range(stop_at):
                mv_file = mv_path.joinpath('{}'.format(frame_index))
                qtc_file = residual_path.joinpath('{}'.format(frame_index))
                size = mv_file.stat().st_size * 8 + qtc_file.stat().st_size * 8
                size_array.append([frame_index, size])
            size_array = np.array(size_array)
            data = {
                'i': params_i,
                'i_period': params_i_period,
                'qp': params_qp,
                'size_array': size_array,
            }
            results[-1].append(data)

    for results_index in range(len(results)):
        curves = results[results_index]
        params_i = params_i_list[results_index]
        params_qp = params_qp_list[results_index]
        for curve in curves:
            plt.plot(curve['size_array'][:, 0], curve['size_array'][:, 1], label = "i_period={}".format(curve['i_period']), marker='o') 
        plt.xlabel('frame index')
        plt.ylabel('bit-count in bits')
        plt.title('i={}, qp={}'.format(params_i, params_qp))
        plt.ticklabel_format(useOffset=False, style='plain')
        plt.legend()
        plt.savefig(config.statistics.path.joinpath('rd_plot2_i_{}.png'.format(params_i)))
        plt.clf()
