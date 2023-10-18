#!/usr/bin/env python3
from lib.utils.config import Config
from lib.signal_processing import psnr, ssim
from lib.utils.misc import block_create
from lib.utils.quantization import quantization_matrix, frame_qtc_to_tc, residual_coefficients_to_residual_frame
from lib.utils.entropy import exp_golomb_decoding, reording_decoding, rle_decoding
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
        params_i (int): block size parameter
        q_matrix (np.ndarray): quantization matrix
"""
def residual_parallel_helper(original_path: pathlib.Path, residual_path: pathlib.Path, i: int, height: int, width: int, residual_pngs_path: pathlib.Path, params_i: int, q_matrix: np.ndarray) -> None:
    current_file = original_path.joinpath(str(i)).read_bytes()
    prev_file = original_path.joinpath(str(i - 1)).read_bytes()
    qtc_file_lines = residual_path.joinpath(str(i)).read_text().split('\n')

    current_frame = np.frombuffer(current_file, dtype=np.uint8).reshape(height, width).astype(np.int16)
    prev_frame = np.frombuffer(prev_file, dtype=np.uint8).reshape(height, width).astype(np.int16)

    qtc_dump = []
    qtc_counter = 0
    for line in qtc_file_lines:
        if line == '':
            continue
        qtc = reording_decoding(rle_decoding(([exp_golomb_decoding(x) for x in line.split(' ')]), q_matrix.shape), q_matrix.shape)
        if qtc_counter == 0:
            qtc_dump.append([])
        qtc_dump[-1].append(qtc)
        qtc_counter += 1
        if qtc_counter == width // params_i:
            qtc_counter = 0

    residual_frame_qtc = np.array(qtc_dump, dtype=np.int16)
    residual_frame_coefficients = frame_qtc_to_tc(residual_frame_qtc, q_matrix)
    generated_residual_frame = residual_coefficients_to_residual_frame(residual_frame_coefficients, params_i, (height, width)).astype(np.int16)

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
        residual_path (Path): path to residual frames
        reconstructed_path (Path): path to reconstructed frames
        output_path (Path): path to save pngs
        height (int): height of frame
        width (int): width of frame
        params_i (int): block size parameter
        q_matrix (np.ndarray): quantization matrix
"""
def predicted_frame_parallel_helper(total_frames: int, residual_path: pathlib.Path, reconstructed_path: pathlib.Path, output_path: pathlib.Path, height: int, width: int, params_i: int, q_matrix: np.ndarray) -> None:
    for i in range(total_frames):
        qtc_file_lines = residual_path.joinpath(str(i)).read_text().split('\n')
        qtc_dump = []
        qtc_counter = 0
        for line in qtc_file_lines:
            if line == '':
                continue
            qtc = reording_decoding(rle_decoding(([exp_golomb_decoding(x) for x in line.split(' ')]), q_matrix.shape), q_matrix.shape)
            if qtc_counter == 0:
                qtc_dump.append([])
            qtc_dump[-1].append(qtc)
            qtc_counter += 1
            if qtc_counter == width // params_i:
                qtc_counter = 0

        residual_frame_qtc = np.array(qtc_dump, dtype=np.int16)
        residual_frame_coefficients = frame_qtc_to_tc(residual_frame_qtc, q_matrix)
        residual_frame = residual_coefficients_to_residual_frame(residual_frame_coefficients, params_i, (height, width)).astype(np.int16)

        reconstructed_file = reconstructed_path.joinpath(str(i)).read_bytes()
        reconstructed_frame = np.frombuffer(reconstructed_file, dtype=np.uint8).reshape(height, width).astype(np.int16)

        predicted_frame = reconstructed_frame - residual_frame
        predicted_frame = predicted_frame.astype(np.uint8)
        predicted_frame = Image.fromarray(predicted_frame)
        predicted_frame.save(output_path.joinpath('{}.png'.format(i)))
        print("predicted frame {} written".format(i))

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
    params_qp = config['params']['qp']
    params_i_period = config['params']['i_period']

    video_path = output_path.joinpath(video_name)
    video_path.mkdir(exist_ok=True)
    params_i_path = video_path.joinpath('i_{}'.format(params_i))
    params_i_path.mkdir(exist_ok=True)
    params_r_path = params_i_path.joinpath('r_{}'.format(params_r))
    params_r_path.mkdir(exist_ok=True)
    params_qp_path = params_r_path.joinpath('qp_{}'.format(params_qp))
    params_qp_path.mkdir(exist_ok=True)
    params_i_period_path = params_qp_path.joinpath('i_period_{}'.format(params_i_period))
    params_i_period_path.mkdir(exist_ok=True)
    pngs_path = params_i_period_path.joinpath('pngs')
    pngs_path.mkdir(exist_ok=True)
    residual_pngs_path = params_i_period_path.joinpath('residual_pngs')
    residual_pngs_path.mkdir(exist_ok=True)
    predicted_pngs_path = params_i_period_path.joinpath('predicted_pngs')
    predicted_pngs_path.mkdir(exist_ok=True)
    statistics_file = params_i_period_path.joinpath('statistics.csv')

    data_path = pathlib.Path.cwd().joinpath(config['output_path']['main_folder'])
    original_path = data_path.joinpath(config['output_path']['original_folder'])
    reconstructed_path = data_path.joinpath(config['output_path']['reconstructed_folder'])
    residual_path = data_path.joinpath(config['output_path']['residual_folder'])
    meta_file = data_path.joinpath(config['output_path']['meta_file'])
    mae_file = data_path.joinpath(config['output_path']['mae_file'])

    l = meta_file.read_text().split(',')
    total_frames = int(l[0])
    height, width = int(l[1]), int(l[2])
    params_i = int(l[3])
    params_qp = int(l[4])
    q_matrix = quantization_matrix(params_i, params_qp)

    residual_jobs = []
    for i in range(1, total_frames):
        job = pool.apply_async(func=residual_parallel_helper, args=(
            original_path,
            residual_path,
            i,
            height,
            width,
            residual_pngs_path,
            params_i,
            q_matrix,
        ))
        residual_jobs.append(job)

    predicted_job = pool.apply_async(func=predicted_frame_parallel_helper, args=(
        total_frames,
        residual_path,
        reconstructed_path,
        predicted_pngs_path,
        height,
        width,
        params_i,
        q_matrix,
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
    plt.title('i={}, r={}, qp={}, i_period={}\n{}, h={}, w={}'.format(params_i, params_r, params_qp, params_i_period, video_name, height, width))
    plt.savefig(params_i_period_path.joinpath('mae.png'))
    plt.clf()

    plt.plot(array[:, 0], array[:, 2])
    plt.xlabel('frame index')
    plt.ylabel('psnr')
    plt.title('i={}, r={}, qp={}, i_period={}\n{}, h={}, w={}'.format(params_i, params_r, params_qp, params_i_period, video_name, height, width))
    plt.savefig(params_i_period_path.joinpath('psnr.png'))
    plt.clf()

    plt.plot(array[:, 0], array[:, 3])
    plt.xlabel('frame index')
    plt.ylabel('ssim')
    plt.title('i={}, r={}, qp={}, i_period={}\n{}, h={}, w={}'.format(params_i, params_r, params_qp, params_i_period, video_name, height, width))
    plt.savefig(params_i_period_path.joinpath('ssim.png'))
    plt.clf()

    for job in residual_jobs:
        job.get()

    predicted_job.get()

    pool.close()
    pool.join()