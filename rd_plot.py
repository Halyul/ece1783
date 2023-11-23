from lib.yuv_processor import YUVProcessor
import numpy as np
from lib.config.config import Config
from lib.signal_processing import psnr
import matplotlib.pyplot as plt
import time
import math

if __name__ == '__main__':
    config = Config('config.yaml')
    mv_path = config.output_path.mv_folder
    residual_path = config.output_path.residual_folder
    original_path = config.output_path.original_folder
    reconstructed_path = config.output_path.reconstructed_folder
    split_counter_path = config.output_path.split_counter_file
    video_name = config.input.split('/')[-1]
    video_path = config.statistics.path.joinpath(video_name)
    meta_file = config.decoder.input_path.meta_file
    params_i = 16
    params_i_period = 10
    params_r = 16
    stop_at = -1
    nRefFrames=1
    VBSEnable=True
    FMEEnable=True
    FastME=True
    RCflag = 0
    targetBR = '1 mbps'
    params_overrides = [
        dict(
            ParallelMode=0,
        ),
        dict(
            ParallelMode=1,
        )
    ]
    results = []

    for i in range(len(params_overrides)):
        results.append([])
        feature_set = params_overrides[i]
        params_qp_list = range(0, int(math.log2(params_i) + 8))
        for params_qp in params_qp_list:
            print("current run: set={}".format(str(feature_set)))
            start = time.time()
            reader = YUVProcessor('config.yaml', 
                                config_override=dict(
                                    params=dict(
                                        i=params_i,
                                        r=params_r,
                                        i_period=params_i_period,
                                        stop_at=stop_at,
                                        qp=params_qp,
                                        nRefFrames=nRefFrames,
                                        VBSEnable=VBSEnable,
                                        FMEEnable=FMEEnable,
                                        FastME=FastME,
                                        RCflag=RCflag,
                                        targetBR=targetBR,
                                        **feature_set
                                    )
                                )
                                )
            end = time.time()
            total_size = 0
            total_psnr = 0
            l = meta_file.read_text().split(',')
            total_frames = stop_at = int(l[0])
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
            split_percentage = split_counter_path.read_text().split('\n').pop(0).split(' ')[1]
            data = {
                'qp': params_qp,
                'size': total_size,
                'psnr': average_psnr,
                'split_percentage': float(split_percentage[:5]),
                'time': exec_time
            }
            results[-1].append(data)
            print(data)
    
    # R-D Plot
    for i in range(len(results)):
        curve = results[i]
        x_axis = [data['size'] for data in curve]
        y_axis = [data['psnr'] for data in curve]
        combined_array = np.column_stack((x_axis, y_axis))
        plt.plot(combined_array[:, 0], combined_array[:, 1],
                 label="ParallelMode={}".format(i), marker='o')
    plt.xlabel('bitrate in bits')
    plt.ylabel('psnr in dB')
    plt.title('R-D Plot')
    plt.legend()
    plt.savefig(config.statistics.path.joinpath('rd_plot.png'))
    plt.clf()
