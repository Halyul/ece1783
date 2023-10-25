#!/usr/bin/env python3
from lib.config.config import Config
from lib.signal_processing import psnr, ssim
from lib.utils.misc import convert_within_range, construct_predicted_frame
import matplotlib.pyplot as plt
import pathlib
from PIL import Image
from PIL import ImageDraw
import numpy as np
import multiprocessing as mp

"""
    Add margin to image

    Parameters:
        pil_image (Image): image to be added margin
        top (int): top margin
        right (int): right margin
        bottom (int): bottom margin
        left (int): left margin
        color (int): color of margin

    Returns:
        result (Image): image with margin
        new_width (int): new width of image
        new_height (int): new height of image
"""
def add_margin(pil_img: Image, top: int, right: int, bottom: int, left: int, color: int) -> tuple:
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result, new_width, new_height

"""
    Generate statistics for each frame in parallel

    Parameters:
        original_path (Path): path to original frames
        reconstructed_path (Path): path to reconstructed frames
        i (int): frame index
        height (int): height of frame
        width (int): width of frame
        mae_dict (dict): dictionary of mae values
        pngs_path (Path): path to save pngs

    Returns:
        i (int): frame index
        mae_value (float): mae value
        psnr_value (float): psnr value
        ssim_value (float): ssim value
"""
def frame_parallel_helper(original_path: pathlib.Path, reconstructed_path: pathlib.Path, i: int, height: int, width: int, mae_dict: dict, pngs_path: pathlib.Path) -> tuple:
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
    draw.text((width // 2 - 20, new_height - 45), 'original', fill=0)
    draw.text((width // 2 * 3 - 25, new_height - 45), 'reconstructed', fill=0)
    combined.save(pngs_path.joinpath('{}.png'.format(i)))

    return (i, mae_value, psnr_value, ssim_value)

"""
    Generate residual image comparsion for each frame in parallel

    Parameters:
        original_path (Path): path to original frames
        residual_path (Path): path to residual frames
        i (int): frame index
        height (int): height of frame
        width (int): width of frame
        residual_pngs_path (Path): path to save pngs
"""
def residual_parallel_helper(original_path: pathlib.Path, residual_path: pathlib.Path, i: int, height: int, width: int, residual_pngs_path: pathlib.Path) -> None:
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

"""
    Generate predicted frame in parallel

    Parameters:
        total_frames (int): total number of frames
        mv_path (Path): path to motion vectors
        reconstructed_path (Path): path to reconstructed frames
        params_i (int): i parameter
        output_path (Path): path to save pngs
        height (int): height of frame
        width (int): width of frame
"""
def predicted_frame_parallel_helper(total_frames: int, mv_path: pathlib.Path, reconstructed_path: pathlib.Path, params_i: int, output_path: pathlib.Path, height: int, width: int) -> None:
    for i in range(total_frames):
        prev_index = i - 1
        if prev_index == -1:
            prev_frame = np.full(height*width, 128).reshape(height, width)
        else:
            prev_file = reconstructed_path.joinpath('{}'.format(prev_index))
            prev_file_bytes = prev_file.read_bytes()
            prev_frame_uint8 = np.frombuffer(prev_file_bytes, dtype=np.uint8).reshape(height, width)
            prev_frame = np.array(prev_frame_uint8, dtype=np.int16)
        
        mv_file = mv_path.joinpath('{}'.format(i))
        mv_file_lines = mv_file.read_text().split('\n')
        mv_dump = []
        mv_counter = 0
        for line in mv_file_lines:
            if line == '':
                continue
            min_motion_vector_y, min_motion_vector_x = line.split(' ')
            min_motion_vector = (int(min_motion_vector_y), int(min_motion_vector_x))
            if mv_counter == 0:
                mv_dump.append([])
            mv_dump[-1].append(min_motion_vector)
            mv_counter += 1
            if mv_counter == width // params_i:
                mv_counter = 0
    
        current_reconstructed_frame = construct_predicted_frame(mv_dump, prev_frame, params_i)
        current_reconstructed_frame = convert_within_range(current_reconstructed_frame)

        reconstructed_frame = Image.fromarray(current_reconstructed_frame)
        reconstructed_frame.save(output_path.joinpath('{}.png'.format(i)))
        print("predicted frame {} written".format(i))

if __name__ == '__main__':
    config = Config('config.yaml')
    pool = mp.Pool(mp.cpu_count())

    output_path = pathlib.Path.cwd().joinpath(config.statistics.path)

    video_name = config.input.split('/')[-1]
    params_i = config.params.i
    params_r = config.params.r
    params_n = config.params.n

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
    predicted_pngs_path = params_n_path.joinpath('predicted_pngs')
    predicted_pngs_path.mkdir(exist_ok=True)
    statistics_file = params_n_path.joinpath('statistics.csv')

    original_path = config.output_path.original_folder
    reconstructed_path = config.output_path.reconstructed_folder
    residual_path = config.output_path.residual_folder
    meta_file = config.output_path.meta_file
    mae_file = config.output_path.mae_file
    mv_path = config.output_path.mv_folder

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

    predicted_job = pool.apply_async(func=predicted_frame_parallel_helper, args=(
        total_frames,
        mv_path,
        reconstructed_path,
        params_i,
        predicted_pngs_path,
        height,
        width,
    ))

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
    plt.title('mae for {}\ni={}, r={}, n={}\nh={}, w={}'.format(video_name, params_i, params_r, params_n, height, width))
    plt.savefig(params_n_path.joinpath('mae.png'))
    plt.clf()

    plt.plot(array[:, 0], array[:, 2])
    plt.xlabel('frame index')
    plt.ylabel('psnr')
    plt.title('psnr for {}\ni={}, r={}, n={}\nh={}, w={}'.format(video_name, params_i, params_r, params_n, height, width))
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

    predicted_job.get()

    pool.close()
    pool.join()