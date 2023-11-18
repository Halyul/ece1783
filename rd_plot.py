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
    params_i = 16
    params_i_period = 8
    params_r = 4
    stop_at = 10
    params_overrides = [
        dict(
            i=params_i,
            r=params_r,
            nRefFrames=1,
            VBSEnable=False,
            FMEEnable=False,
            FastME=False,
        ),
        dict(
            i=params_i,
            r=params_r,
            nRefFrames=4,
            VBSEnable=False,
            FMEEnable=False,
            FastME=False,
        ),
        dict(
            i=params_i,
            r=params_r,
            nRefFrames=1,
            VBSEnable=True,
            FMEEnable=False,
            FastME=False,
        ),
        dict(
            i=params_i,
            r=params_r,
            nRefFrames=1,
            VBSEnable=False,
            FMEEnable=True,
            FastME=False,
        ),
        dict(
            i=params_i,
            r=params_r,
            nRefFrames=1,
            VBSEnable=False,
            FMEEnable=False,
            FastME=True,
        ),
        dict(
            i=params_i,
            r=params_r,
            nRefFrames=4,
            VBSEnable=True,
            FMEEnable=True,
            FastME=True,
        )
    ]
    results = []

    for i in range(len(params_overrides)):
        results.append([])
        feature_set = params_overrides[i]
        params_qp_list = range(0, int(math.log2(feature_set['i']) + 8))
        for params_qp in params_qp_list:
            print("current run: set={}".format(str(feature_set)))
            start = time.time()
            reader = YUVProcessor('config.yaml', 
                                config_override=dict(
                                    params=dict(
                                        i_period=params_i_period,
                                        stop_at=stop_at,
                                        qp=params_qp,
                                        **feature_set
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
                 label="feature_set={}".format(i), marker='o')
    plt.xlabel('bitrate in bits')
    plt.ylabel('psnr in dB')
    plt.title('R-D Plot')
    plt.legend()
    plt.savefig(config.statistics.path.joinpath('rd_plot.png'))
    plt.clf()

    # execution time
    for i in range(len(results)):
        curve = results[i]
        x_axis = [data['size'] for data in curve]
        y_axis = [data['time'] for data in curve]
        combined_array = np.column_stack((x_axis, y_axis))
        plt.plot(combined_array[:, 0], combined_array[:, 1],
                 label="feature_set={}".format(i), marker='o')
    plt.xlabel('bitrate in bits')
    plt.ylabel('time in seconds')
    plt.title('Execution Time')
    plt.legend()
    plt.savefig(config.statistics.path.joinpath('execution_time.png'))
    plt.clf()

    # VBS percentage, x = qp
    points = results[2]
    x_axis = [data['qp'] for data in points]
    y_axis = [data['split_percentage'] for data in points]
    combined_array = np.column_stack((x_axis, y_axis))
    plt.plot(combined_array[:, 0], combined_array[:, 1], marker='o') 
    plt.xlabel('tested qp values')
    plt.ylabel('percentage in %')
    plt.title('VBS percentage, x = qp')
    plt.ticklabel_format(useOffset=False, style='plain')
    plt.savefig(config.statistics.path.joinpath('vbs_percentage_qp.png'))
    plt.clf()

    # VBS percentage, x = bitrate
    points = results[2]
    data = [dict(size=data['size'], split_percentage=data['split_percentage']) for data in points]
    data.sort(key=lambda x: x['size'])
    x_axis = [data['size'] for data in data]
    y_axis = [data['split_percentage'] for data in data]
    combined_array = np.column_stack((x_axis, y_axis))
    plt.plot(combined_array[:, 0], combined_array[:, 1], marker='o')
    plt.xlabel('bitrate in bits')
    plt.ylabel('percentage in %')
    plt.title('VBS percentage, x = bitrate')
    plt.savefig(config.statistics.path.joinpath('vbs_percentage_size.png'))
    plt.clf()
