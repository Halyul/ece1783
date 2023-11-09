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
    params_i_list = [16]
    params_i_period_list = [8]
    params_r = 4
    stop_at = 10
    results = []

    for i in range(len(params_i_list)):
        results.append([])
        params_i = params_i_list[i]
        params_qp_list = range(0, int(math.log2(params_i) + 8))
        for params_i_period in params_i_period_list:
            results[-1].append([])
            for params_qp in params_qp_list:
                print("current run: i={}, i_period={}, qp={}".format(params_i, params_i_period, params_qp))
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
                total_size = 0
                total_psnr = 0
                for j in range(stop_at):
                    mv_file = mv_path.joinpath('{}'.format(j))
                    qtc_file = residual_path.joinpath('{}'.format(j))
                    total_size += mv_file.stat().st_size * 8 + qtc_file.stat().st_size * 8
                    original_file = original_path.joinpath(str(j)).read_bytes()
                    reconstructed_file = reconstructed_path.joinpath(str(j)).read_bytes()

                    original_frame = np.frombuffer(original_file, dtype=np.uint8).reshape(reader.info['paded_height'], reader.info['paded_width'])
                    reconstructed_frame = np.frombuffer(reconstructed_file, dtype=np.uint8).reshape(reader.info['paded_height'], reader.info['paded_width'])
                    total_psnr += psnr(original_frame, reconstructed_frame)
                average_psnr = total_psnr / stop_at
                exec_time = end - start
                print('Time: {}s'.format(exec_time))
                data = {
                    'i': params_i,
                    'i_period': params_i_period,
                    'qp': params_qp,
                    'size': total_size,
                    'psnr': average_psnr,
                    'time': exec_time
                }
                results[-1][-1].append(data)
                print(data)
    
    for i in range(len(results)):
        curves = results[i]
        params_i = params_i_list[i]
        for curve in curves:
            x_axis = [data['size'] for data in curve]
            y_axis = [data['psnr'] for data in curve]
            combined_array = np.column_stack((x_axis, y_axis))
            plt.plot(combined_array[:, 0], combined_array[:, 1], label = "i_period={}".format(curve[0]['i_period']), marker='o') 
        plt.xlabel('bitrate in bits')
        plt.ylabel('psnr in dB')
        plt.title('R-D Plot for i={}'.format(params_i))
        plt.ticklabel_format(useOffset=False, style='plain')
        plt.legend()
        plt.savefig(config.statistics.path.joinpath('rd_plot_i_{}.png'.format(params_i_list[i])))
        plt.clf()

    for i in range(len(results)):
        curves = results[i]
        params_i = params_i_list[i]
        for curve in curves:
            x_axis = [data['size'] for data in curve]
            y_axis = [data['time'] for data in curve]
            combined_array = np.column_stack((x_axis, y_axis))
            plt.plot(combined_array[:, 0], combined_array[:, 1], label = "i_period={}".format(curve[0]['i_period']), marker='o') 
        plt.xlabel('bitrate in bits')
        plt.ylabel('time in seconds')
        plt.title('Execution Time for i={}'.format(params_i))
        plt.ticklabel_format(useOffset=False, style='plain')
        plt.legend()
        plt.savefig(config.statistics.path.joinpath('execution_time_i_{}.png'.format(params_i_list[i])))
        plt.clf()