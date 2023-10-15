from lib.utils.config import Config
from lib.signal_processing import psnr, ssim
import matplotlib.pyplot as plt
import pathlib
from PIL import Image
from PIL import ImageDraw
import numpy as np

config_class = Config('config.yaml')
config = config_class.config

output_path = pathlib.Path.cwd().joinpath(config['statistics']['path'])
if not output_path.exists():
    output_path.mkdir()

video_name = config['input'].split('/')[-1]
params_i = config['params']['i']
params_r = config['params']['r']
params_n = config['params']['n']

video_path = output_path.joinpath(video_name)
video_path.mkdir(exist_ok=True)
params_i_path = video_path.joinpath('i_{}'.format(params_i))
params_i_path.mkdir(exist_ok=True)
params_r_path = params_i_path.joinpath('r_{}'.format(params_r))
params_r_path.mkdir(exist_ok=True)
params_n_path = params_r_path.joinpath('n_{}'.format(params_n))
params_n_path.mkdir(exist_ok=True)
pngs_path = params_n_path.joinpath('pngs')
pngs_path.mkdir(exist_ok=True)
statistics_file = params_n_path.joinpath('statistics.csv')

data_path = pathlib.Path.cwd().joinpath(config['output_path']['main_folder'])
mv_path = data_path.joinpath(config['output_path']['mv_folder'])
original_path = data_path.joinpath(config['output_path']['original_folder'])
reconstructed_path = data_path.joinpath(config['output_path']['reconstructed_folder'])
meta_file = data_path.joinpath(config['output_path']['meta_file'])
mae_file = data_path.joinpath(config['output_path']['mae_file'])

l = meta_file.read_text().split(',')
total_frames = int(l[0])
height, width = int(l[1]), int(l[2])

mae_list = mae_file.read_text().split('\n')
mae_dict = {}
for item in mae_list:
    if item == '':
        continue
    frame_index, mae = item.split(' ')
    mae_dict[int(frame_index)] = float(mae)

array = []
for i in range(total_frames):
    original_file = original_path.joinpath(str(i)).read_bytes()
    reconstructed_file = reconstructed_path.joinpath(str(i)).read_bytes()

    original_frame = np.frombuffer(original_file, dtype=np.uint8).reshape(height, width)
    reconstructed_frame = np.frombuffer(reconstructed_file, dtype=np.uint8).reshape(height, width)

    psnr_value = psnr(original_frame, reconstructed_frame)
    ssim_value = ssim(original_frame, reconstructed_frame).mean()
    mae_value = mae_dict[i]

    array.append([i, mae_value, psnr_value, ssim_value])
    print("frame index: {}, mae: {}, psnr: {}, ssim: {}".format(i, mae_value, psnr_value, ssim_value))

    # combine two image
    original_frame = Image.fromarray(original_frame)
    reconstructed_frame = Image.fromarray(reconstructed_frame)
    combined = Image.new('L', (width * 2, height))
    combined.paste(original_frame, (0, 0))
    combined.paste(reconstructed_frame, (width, 0))
    # add text to left bottom to image to indicate mae, psnr, ssim
    draw = ImageDraw.Draw(combined)
    draw.text((5, height - 55), 'mae: {}'.format(mae_value), fill=255)
    draw.text((5, height - 45), 'psnr: {}'.format(psnr_value), fill=255)
    draw.text((5, height - 35), 'ssim: {}'.format(ssim_value), fill=255)
    draw.text((5, height - 25), 'left: original', fill=255)
    draw.text((5, height - 15), 'right: reconstructed', fill=255)
    combined.save(pngs_path.joinpath('{}.png'.format(i)))

array = np.array(array)
np.savetxt(statistics_file, array, delimiter=',', header='frame_index,mae,psnr,ssim', comments='')

plt.plot(array[:, 0], array[:, 1])
plt.xlabel('frame index')
plt.ylabel('mae')
plt.title('mae for {}\ni={}, r={}, n={}h={}, w={}'.format(video_name, params_i, params_r, params_n, height, width))
plt.savefig(params_n_path.joinpath('mae.png'))
plt.clf()

plt.plot(array[:, 0], array[:, 2])
plt.xlabel('frame index')
plt.ylabel('psnr')
plt.title('psnr for {}\ni={}, r={}, n={}h={}, w={}'.format(video_name, params_i, params_r, params_n, height, width))
plt.savefig(params_n_path.joinpath('psnr.png'))
plt.clf()

plt.plot(array[:, 0], array[:, 3])
plt.xlabel('frame index')
plt.ylabel('ssim')
plt.title('ssim for {}\ni={}, r={}, n={}\nh={}, w={}'.format(video_name, params_i, params_r, params_n, height, width))
plt.savefig(params_n_path.joinpath('ssim.png'))
plt.clf()