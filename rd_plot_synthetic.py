from lib.yuv_processor import YUVProcessor
import numpy as np
from lib.config.config import Config
from lib.signal_processing import psnr
import matplotlib.pyplot as plt

if __name__ == '__main__':
    config = Config('config.yaml')
    mv_path = config.output_path.mv_folder
    residual_path = config.output_path.residual_folder
    original_path = config.output_path.original_folder
    reconstructed_path = config.output_path.reconstructed_folder
    split_counter_path = config.output_path.split_counter_file
    video_name = config.input.split('/')[-1]
    video_path = config.statistics.path.joinpath(video_name)
    input_video = 'videos/synthetic.yuv'
    video_params = dict(
        height=288,
        width=352,
        color_space=420,
    )
    params_i = 16
    params_i_period = 8
    params_r = 4
    stop_at = 10
    params_qp = 4
    params_overrides = [
        dict(
            nRefFrames=1,
            VBSEnable=False,
            FMEEnable=False,
            FastME=False,
        ),
        dict(
            nRefFrames=2,
            VBSEnable=False,
            FMEEnable=False,
            FastME=False,
        ),
        dict(
            nRefFrames=3,
            VBSEnable=False,
            FMEEnable=False,
            FastME=False,
        ),
        dict(
            nRefFrames=4,
            VBSEnable=False,
            FMEEnable=False,
            FastME=False,
        ),
    ]
    results = []

    for i in range(len(params_overrides)):
        feature_set = params_overrides[i]
        print("current run: set={}".format(str(feature_set)))
        reader = YUVProcessor('config.yaml', 
                            config_override=dict(
                                input=input_video,
                                video_params=video_params,
                                params=dict(
                                    qp=params_qp,
                                    **feature_set
                                )
                            )
        )
        data = []
        for j in range(stop_at):
            mv_file = mv_path.joinpath('{}'.format(j))
            qtc_file = residual_path.joinpath('{}'.format(j))
            bitsize = mv_file.stat().st_size * 8 + qtc_file.stat().st_size * 8
            original_file = original_path.joinpath(str(j)).read_bytes()
            reconstructed_file = reconstructed_path.joinpath(str(j)).read_bytes()

            original_frame = np.frombuffer(original_file, dtype=np.uint8).reshape(reader.info['paded_height'], reader.info['paded_width'])
            reconstructed_frame = np.frombuffer(reconstructed_file, dtype=np.uint8).reshape(reader.info['paded_height'], reader.info['paded_width'])
            frame_psnr = psnr(original_frame, reconstructed_frame)
            data.append(dict(
                frame_index = j,
                size = bitsize,
                psnr = frame_psnr,
            ))
        results.append(data)
        print(data)
    
    for i in range(len(results)):
        curve = results[i]
        data = [dict(size=data['size'], psnr=data['psnr']) for data in curve]
        data.sort(key=lambda x: x['size'])
        x_axis = [data['size'] for data in data]
        y_axis = [data['psnr'] for data in data]
        combined_array = np.column_stack((x_axis, y_axis))
        plt.plot(combined_array[:, 0], combined_array[:, 1],
                 label="nRefFrames={}".format(params_overrides[i]['nRefFrames']), marker='o')
        plt.ticklabel_format(useOffset=False, style='plain')
        plt.xlabel('bitrate in bits')
        plt.ylabel('psnr in dB')
        plt.title('R-D Plot for nRefFrames = {}'.format(params_overrides[i]['nRefFrames']))
        plt.legend()
        plt.savefig(config.statistics.path.joinpath('rd_plot_synthetic_{}.png'.format(params_overrides[i]['nRefFrames'])))
        plt.clf()
