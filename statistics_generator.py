from lib.utils.config import Config
from lib.signal_processing import psnr, ssim
import matplotlib.pyplot as plt
import pathlib
from PIL import Image
from PIL import ImageDraw
import numpy as np
import multiprocessing as mp

def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result, new_width, new_height

def frame_parallel_helper(original_path, reconstructed_path, i, height, width, mae_dict, pngs_path):
    original_file = original_path.joinpath(str(i)).read_bytes()
    reconstructed_file = reconstructed_path.joinpath(str(i)).read_bytes()

    original_frame = np.frombuffer(original_file, dtype=np.uint8).reshape(height, width)
    reconstructed_frame = np.frombuffer(reconstructed_file, dtype=np.uint8).reshape(height, width)

    psnr_value = psnr(original_frame, reconstructed_frame)
    ssim_value = ssim(original_frame, reconstructed_frame).mean()
    mae_value = mae_dict[i]

    print("frame index: {}, mae: {}, psnr: {}, ssim: {}".format(i, mae_value, psnr_value, ssim_value))

    # combine two image
    original_frame = Image.fromarray(original_frame)
    reconstructed_frame = Image.fromarray(reconstructed_frame)
    original_frame, width, height = add_margin(original_frame, 10, 5, 0, 10, 255)
    reconstructed_frame, _, _ = add_margin(reconstructed_frame, 10, 10, 0, 5, 255)
    combined = Image.new('L', (width * 2, height))
    combined.paste(original_frame, (0, 0))
    combined.paste(reconstructed_frame, (width, 0))
    # add text to left bottom to image to indicate mae, psnr, ssim
    combined, _, new_height = add_margin(combined, 0, 0, 50, 0, 255)
    draw = ImageDraw.Draw(combined)
    draw.text((5, new_height - 35), 'mae: {}'.format(mae_value), fill=0)
    draw.text((5, new_height - 25), 'psnr: {}'.format(psnr_value), fill=0)
    draw.text((5, new_height - 15), 'ssim: {}'.format(ssim_value), fill=0)
    draw.text((width // 2, new_height - 45), 'original', fill=0)
    draw.text((width // 2 * 3, new_height - 45), 'reconstructed', fill=0)
    combined.save(pngs_path.joinpath('{}.png'.format(i)))

    return (i, mae_value, psnr_value, ssim_value)

def residual_parallel_helper(original_path, residual_path, i, height, width, residual_pngs_path):
    current_file = original_path.joinpath(str(i)).read_bytes()
    prev_file = original_path.joinpath(str(i - 1)).read_bytes()
    generated_residual_file = residual_path.joinpath(str(i)).read_bytes()

    current_frame = np.frombuffer(current_file, dtype=np.uint8).reshape(height, width).astype(np.int16)
    prev_frame = np.frombuffer(prev_file, dtype=np.uint8).reshape(height, width).astype(np.int16)
    generated_residual_frame = np.frombuffer(generated_residual_file, dtype=np.int16).reshape(height, width)

    residual_frame = current_frame - prev_frame
    residual_frame = np.abs(residual_frame)

    # combine two image
    residual_frame = Image.fromarray(residual_frame)
    generated_residual_frame = Image.fromarray(generated_residual_frame)
    residual_frame, width, height = add_margin(residual_frame, 10, 5, 0, 10, 255)
    generated_residual_frame, _, _ = add_margin(generated_residual_frame, 10, 10, 0, 5, 255)
    combined = Image.new('L', (width * 2, height))
    combined.paste(residual_frame, (0, 0))
    combined.paste(generated_residual_frame, (width, 0))
    # add text
    combined, _, new_height = add_margin(combined, 0, 0, 20, 0, 255)
    draw = ImageDraw.Draw(combined)
    draw.text((width // 4 , new_height - 15), 'Abs Diff w/o Motion Compensation', fill=0)
    draw.text((width // 4 * 5, new_height - 15), 'Abs Diff w/ Motion Compensation', fill=0)
    combined.save(residual_pngs_path.joinpath('{}.png'.format(i)))

if __name__ == '__main__':
    config_class = Config('config.yaml')
    config = config_class.config
    pool = mp.Pool(mp.cpu_count())

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
    residual_pngs_path = params_n_path.joinpath('residual_pngs')
    residual_pngs_path.mkdir(exist_ok=True)
    statistics_file = params_n_path.joinpath('statistics.csv')

    data_path = pathlib.Path.cwd().joinpath(config['output_path']['main_folder'])
    mv_path = data_path.joinpath(config['output_path']['mv_folder'])
    original_path = data_path.joinpath(config['output_path']['original_folder'])
    reconstructed_path = data_path.joinpath(config['output_path']['reconstructed_folder'])
    residual_path = data_path.joinpath(config['output_path']['residual_folder'])
    meta_file = data_path.joinpath(config['output_path']['meta_file'])
    mae_file = data_path.joinpath(config['output_path']['mae_file'])

    l = meta_file.read_text().split(',')
    total_frames = int(l[0])
    height, width = int(l[1]), int(l[2])

    residual_jobs = []
    for i in range(1, total_frames):
        job = pool.apply_async(func=residual_parallel_helper, args=(
            original_path,
            residual_path,
            i,
            height,
            width,
            residual_pngs_path,
        ))
        residual_jobs.append(job)

    mae_list = mae_file.read_text().split('\n')
    mae_dict = {}
    for item in mae_list:
        if item == '':
            continue
        frame_index, mae = item.split(' ')
        mae_dict[int(frame_index)] = float(mae)

    jobs = []
    results = []
    for i in range(total_frames):
        job = pool.apply_async(func=frame_parallel_helper, args=(
            original_path,
            reconstructed_path,
            i,
            height,
            width,
            mae_dict,
            pngs_path,
        ))
        jobs.append(job)
    
    for job in jobs:
        results.append(job.get())

    results.sort(key=lambda x: x[0])
    array = np.array(results)
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

    for job in residual_jobs:
        job.get()

    pool.close()
    pool.join()