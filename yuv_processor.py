import pathlib
import shutil
import math
import multiprocessing as mp
from enum import Enum
import numpy as np
from PIL import Image

from lib.config import Config

class YUVFormat(Enum):
    YUV444 = 444
    YUV422 = 422
    YUV420 = 420

class Identifier(Enum):
    FRAME = b'FRAME\n'
    END = 0x0A.to_bytes()
    SPACER = 0x20.to_bytes()

class YUVProcessor:

    HEADER_IDENTIFIERS = {
        'FORMAT': bytes('YUV4MPEG2', 'ascii'),
        'WIDTH': bytes('W', 'ascii'),
        'HEIGHT': bytes('H', 'ascii'),
        'FRAMERATE': bytes('F', 'ascii'),
        'INTERLACE': bytes('I', 'ascii'),
        'ASPECT_RATIO': bytes('A', 'ascii'),
        'COLOR_SPACE': bytes('C', 'ascii'),
        'COMMENT': bytes('X', 'ascii'),
        '__order': ['FORMAT', 'WIDTH', 'HEIGHT', 'FRAMERATE', 'INTERLACE', 'ASPECT_RATIO', 'COLOR_SPACE', 'COMMENT']
    }

    FRAMERATE_PREDEFINED = {
        '30:1': '30 FPS',
        '25:1': '25 FPS',
        '24:1': '24 FPS',
        '30000:1001': '29.97 FPS',
        '24000:1001': '23.98 FPS',
    }

    INTERLACE_PREDEFINED = {
        'p': 'Progressive',
        't': 'Top Field First',
        'b': 'Bottom Field First',
        'm': 'Mixed',
    }

    ASPECT_RATIO_PREDEFINED = {
        '0:0': 'Unknown',
        '1:1': 'Square',
        '4:3': '4:3, 480x480',
        '4:5': '4:3, 720x480',
        '32:27': '16:9, 720x480',
    }

    COLOR_SPACE_PREDEFINED = {
        '420jpeg': '420', # YUV 4:2:0 JPEG
        '420paldv': '420', # YUV 4:2:0 PAL-DV
        '420': '420', # YUV 4:2:0
        '422': '422', # YUV 4:2:2
        '444': '444', # YUV 4:4:4
        'mono': 'YCbCr plane only',
    }
    
    def __init__(self, config_path):
        self.__config = Config(config_path)
        self.config = self.__config.config
        self.file_path = self.config['input']
        self.file = pathlib.Path.cwd().joinpath(self.file_path)
        if not self.file.exists():
            raise FileNotFoundError('File not found.')
        self.__mp = MultiProcessFW(self.__config)
        self.info = {
            'format': '',
            'width': '',
            'height': '',
            'framerate': '',
            'interlace': '',
            'aspect_ratio': '',
            'color_space': '',
            'comment': '',
            'frame_count': '',
        }
        self.__file_stream = open(self.file, 'rb')
        self.__byte = None
        self.__format = None
        self.__frame_index = 0

        self.upscale = self.config['upscale']
        if self.upscale:
            if self.upscale == 420:
                self.upscale = YUVFormat.YUV420
            elif self.upscale == 422:
                self.upscale = YUVFormat.YUV422
            else:
                self.upscale = YUVFormat.YUV444
        self.__offsets = None
        self.__frame_processor = FrameProcessing(self.__config, self.upscale)
        self.__debug = self.config['debug'] if 'debug' in self.config else False
        self.__deconstruct()
        return
    
    """
        Deconstruct the YUV file into its components.
    """
    def __deconstruct(self):
        self.__read_header()
        self.__read_frames()
        self.info['frame_count'] = self.__frame_index
        self.__mp.done()
        return
    
    """
        Read the header of the YUV file.
    """
    def __read_header(self) -> None:
        raw_header = bytearray()
        while self.__read_byte() != Identifier.END.value:
            raw_header.extend(self.__byte)
            if self.__byte == Identifier.SPACER.value:
                self.HEADER_IDENTIFIERS['__order'].pop(0)
                continue

            order = self.HEADER_IDENTIFIERS['__order'][0]
            current_decoded_byte = self.__decode_current_byte()
            if order == 'FORMAT':
                self.info['format'] += current_decoded_byte
            elif order == 'WIDTH':
                if self.__byte == self.HEADER_IDENTIFIERS['WIDTH']:
                    continue
                self.info['width'] += current_decoded_byte
            elif order == 'HEIGHT':
                if self.__byte == self.HEADER_IDENTIFIERS['HEIGHT']:
                    continue
                self.info['height'] += current_decoded_byte
            elif order == 'FRAMERATE':
                if self.__byte == self.HEADER_IDENTIFIERS['FRAMERATE']:
                    continue
                self.info['framerate'] += current_decoded_byte
            elif order == 'INTERLACE':
                if self.__byte == self.HEADER_IDENTIFIERS['INTERLACE']:
                    continue
                self.info['interlace'] += current_decoded_byte
            elif order == 'ASPECT_RATIO':
                if self.__byte == self.HEADER_IDENTIFIERS['ASPECT_RATIO']:
                    continue
                self.info['aspect_ratio'] += current_decoded_byte
            elif order == 'COLOR_SPACE':
                if self.__byte == self.HEADER_IDENTIFIERS['COLOR_SPACE']:
                    continue
                self.info['color_space'] += current_decoded_byte
            elif order == 'COMMENT':
                if self.__byte == self.HEADER_IDENTIFIERS['COMMENT']:
                    continue
                self.info['comment'] += current_decoded_byte
            else:
                raise Exception('Invalid header identifier.')
        
        self.info['width'] = int(self.info['width'])
        self.info['height'] = int(self.info['height'])
        if self.info['framerate'] in self.FRAMERATE_PREDEFINED:
            self.info['framerate'] = self.FRAMERATE_PREDEFINED[self.info['framerate']]
        if self.info['interlace'] in self.INTERLACE_PREDEFINED:
            self.info['interlace'] = self.INTERLACE_PREDEFINED[self.info['interlace']]
        if self.info['aspect_ratio'] in self.ASPECT_RATIO_PREDEFINED:
            self.info['aspect_ratio'] = self.ASPECT_RATIO_PREDEFINED[self.info['aspect_ratio']]
        if self.info['color_space'] in self.COLOR_SPACE_PREDEFINED:
            self.info['color_space'] = self.COLOR_SPACE_PREDEFINED[self.info['color_space']]
        else:
            self.info['color_space'] = str(YUVFormat.YUV420.value)
        
        if self.info['color_space'] == str(YUVFormat.YUV422.value):
            self.__format = YUVFormat.YUV422
        elif self.info['color_space'] == str(YUVFormat.YUV444.value):
            self.__format = YUVFormat.YUV444
        elif self.info['color_space'] == str(YUVFormat.YUV420.value):
            self.__format = YUVFormat.YUV420
        else:
            raise Exception('Unknown color space {}.'.format(self.info['color_space']))
        self.__offsets = self.__get_offsets()

        raw_header.extend(Identifier.SPACER.value) # add colorspace
        raw_header.extend(self.HEADER_IDENTIFIERS['COLOR_SPACE'])
        raw_header.extend(bytes(str(self.upscale.value), 'ascii')) # add upscale
        raw_header.extend(self.__byte) # add END_IDENTIFIER
        self.__mp.add_to_video_q(raw_header)
        return
    
    """
        Read the frames of the YUV file, and then upscale to YUV444.
    """
    def __read_frames(self) -> None:
        self.__read_byte() # skil END_IDENTIFIER, after this line, self.__byte == b'F'
        while self.__byte != Identifier.END.value:
            # skip first "FRAME"
            # when exit, self.__byte == self.END_IDENTIFIER
            self.__read_byte()

        mode = None
        while not self.__eof():
            
            result = bytearray()
            while not ((len(result) > 6 and result[-len(Identifier.FRAME.value):] == Identifier.FRAME.value) or self.__eof()):
                # read yuv components + "FRAME"
                # when exit, self.__byte == self.END_IDENTIFIER
                result.extend(self.__read_byte())

            if self.__eof():
                # end of file
                yuv_components = result
            else:
                # has next "FRAME"
                yuv_components = result[:-len(Identifier.FRAME.value)]

            # process each frame
            self.__mp.dispatch(self.__frame_processor.upscale, (self.info['width'], self.__frame_index, self.__offsets, yuv_components, self.__format), self.__debug)
            print(self.__frame_index)
            self.__frame_index += 1
        
        return
    
    """
        Return the offsets of the YUV file.

        Returns:
            offsets (tuple): The offsets of the YUV file.
    """
    def __get_offsets(self) -> tuple:
        y_length = int(self.info['width']) * int(self.info['height'])
        if self.__format == YUVFormat.YUV420:
            u_length = y_length // 4
            v_length = y_length // 4
        elif self.__format == YUVFormat.YUV422:
            u_length = y_length // 2
            v_length = y_length // 2
        else:
            u_length = y_length
            v_length = y_length
        return (y_length, u_length, v_length)

    """
        Check if the file stream is at the end of file.

        Returns:
            eof (bool): True if the file stream is at the end of file.
    """
    def __eof(self) -> bool:
        return self.__byte == b''
    
    """
        Read one byte from the file stream.

        Returns:
            byte (bytes): The current byte.
    """
    def __read_byte(self) -> bytes:
        self.__byte = self.__file_stream.read(1)
        return self.__byte

    """
        Decode the current byte to ASCII.

        Returns:
            byte (str): The current byte in ASCII.
    """
    def __decode_current_byte(self) -> str:
        return self.__byte.decode('ascii')

    """
        Append value to header info.

        Parameters:
            key (str): The key of the header info.
            value (str): The value to be appended.
    """
    def append_to_header_info(self, key, value):
        self.info[key] += value
        return

class MultiProcessFW:

    def __init__(self, config) -> None:
        self.__config = config
        self.manager = mp.Manager()
        self.pool = mp.Pool(mp.cpu_count() + 2)
        self.jobs = []

        self.__video_path = self.__config.get_output_path('video')
        self.__png_path = self.__config.get_output_path('pngs')
        self.__y_only_path = self.__config.get_output_path('y_only')
        self.video_q = None
        if self.__video_path is not None:
            self.video_q = self.manager.Queue()
            self.output_watcher = self.pool.apply_async(self.write_raw_bytes_to_file, (self.__video_path, self.video_q,))
            self.__clear_output_path()

        self.png_q = None
        if self.__png_path is not None:
            self.png_q = self.manager.Queue()
            self.png_watcher = self.pool.apply_async(self.write_to_png, (self.__png_path, self.png_q,))
            self.__clear_png_path()
            pathlib.Path.cwd().joinpath(self.__png_path).mkdir(parents=True, exist_ok=True)

        self.y_only_q = None
        if self.__y_only_path is not None:
            self.y_only_q = self.manager.Queue()
            self.y_only_watcher = self.pool.apply_async(self.write_to_y_only_file, (self.__y_only_path, self.y_only_q,))
            self.__clear_y_only_path()
            pathlib.Path.cwd().joinpath(self.__y_only_path).mkdir(parents=True, exist_ok=True)

    """
        Clear the output video path.
    """
    def __clear_output_path(self):
        if self.__video_path is not None:
            pathlib.Path.cwd().joinpath(self.__video_path.parent).mkdir(parents=True, exist_ok=True)
            if self.__video_path.exists():
                self.__video_path.unlink()
                self.__video_path.write_bytes(b'')
        return
    
    """
        Clear the png path.
    """
    def __clear_png_path(self):
        if self.__png_path is not None:
            if self.__png_path.exists():
                shutil.rmtree(self.__png_path)
        return

    """
        Clear the y_only path.
    """
    def __clear_y_only_path(self):
        if self.__y_only_path is not None:
            if self.__y_only_path.exists():
                shutil.rmtree(self.__y_only_path)
        return

    """
        Run a function in parallel.

        Parameters:
            func (function): The function to be run in parallel.
            data (tuple): The data to be passed to the function.
            debug (bool): True if debug mode is on.
    """
    def dispatch(self, func, data, debug=False):
        if debug:
            func(data, (
                self.video_q, 
                self.png_q,
                self.y_only_q
            ))
        else:
            job = self.pool.apply_async(func=func, args=(
                data, 
                (
                    self.video_q, 
                    self.png_q,
                    self.y_only_q
                ),
            ))
            self.jobs.append(job)
        return
    
    """
        Add data to output video queue.

        Parameters:
            data (bytearray): The data to be added to the queue.
    """
    def add_to_video_q(self, data):
        if self.__video_path is not None:
            self.video_q.put((-1, data))
        return
    
    """
        Write Y-only frame to file.
        
        Parameters:
            path (pathlib.Path): The path of the png file.
            q (mp.Queue): The queue to get data from.
    """
    @staticmethod
    def write_to_y_only_file(path, q):
        while True:
            data = q.get()
            if data == 'kill':
                break
            (y, frame_index) = data
            path.joinpath('{}'.format(frame_index)).write_bytes(y)
            print('done write ', frame_index)
        return

    """
        Write RGB to png file.
        
        Parameters:
            path (pathlib.Path): The path of the png file.
            q (mp.Queue): The queue to get data from.
    """
    @staticmethod
    def write_to_png(path, q):
        while True:
            data = q.get()
            if data == 'kill':
                break
            (rgb, frame_index) = data
            img = Image.fromarray(rgb)
            img.save(path.joinpath('{}.png'.format(frame_index)))
            print('done write png', frame_index)
        return
    
    """
        Write raw bytes to a video file.

        Parameters:
            file (pathlib.Path): The path of the video file.
            q (mp.Queue): The queue to get data from.
    """
    @staticmethod
    def write_raw_bytes_to_file(file, q):
        next_expected_frame = 0
        pending_frames = {}
        with file.open("wb") as f:
            while True:
                data = q.get()
                if data == 'kill':
                    break
                frame_index, frame = data
                if frame_index == -1:
                    f.write(frame)
                    f.flush()
                    continue
                if frame_index == next_expected_frame:
                    f.write(frame)
                    f.flush()
                    next_expected_frame += 1
                    while next_expected_frame in pending_frames:
                        f.write(pending_frames.pop(next_expected_frame))
                        print('done write pending frame', next_expected_frame)
                        f.flush()
                        next_expected_frame += 1
                else:
                    pending_frames[frame_index] = frame
                    print('added to pending frames', frame_index)
        return
    
    """
        Wait all processes to be done.
    """
    def done(self):
        for job in self.jobs: 
            job.get()
        
        if self.__video_path is not None:
            self.video_q.put('kill')
        if self.__png_path is not None:
            self.png_q.put('kill')
        if self.__y_only_path is not None:
            self.y_only_q.put('kill')
        self.pool.close()
        self.pool.join()

        return
    
class FrameProcessing:

    def __init__(self, config, upscale) -> None:
        self.__config = config
        self.config = self.__config.config
        self.__upscale = upscale
        self.__noise = self.config['noise'] if 'noise' in self.config else None

    """
        Upscale the frame to YUV444.

        Parameters:
            data (tuple): The data to be processed.
            queues (tuple): The queues to put data to.
    """
    def upscale(self, data, queues):
        (width, frame_index, format_tuple, yuv_components, mode) = data
        (video_q, png_q, y_only_q) = queues

        pixel_list = []

        for i in range(format_tuple[0]):
            y_component = yuv_components[i]
            current_row = i // width
            current_col = i % width
            if mode == YUVFormat.YUV420:
                # YUV420
                u_component = yuv_components[format_tuple[0] + current_row // 2 * width // 2 + current_col // 2]
                v_component = yuv_components[format_tuple[0] + format_tuple[1] + current_row // 2 * width // 2 + current_col // 2]
            elif mode == YUVFormat.YUV422:
                # YUV422
                u_component = yuv_components[format_tuple[0] + i // 2]
                v_component = yuv_components[format_tuple[0] + format_tuple[1] + i // 2]
            else:
                # YUV444
                u_component = yuv_components[format_tuple[0] + i]
                v_component = yuv_components[format_tuple[0] + format_tuple[1] + i]
                break
            if i % width == 0:
                pixel_list.append([])
            pixel_list[current_row].append([y_component, u_component, v_component])

        np_array = np.array(pixel_list)

        # output png
        if png_q is not None:
            self.__output_to_pngs((np_array, frame_index), png_q)

        # output video
        if video_q is not None:
            self.__output_to_video((np_array, frame_index), video_q)

        # output y_only
        if y_only_q is not None:
            self.__output_to_y_only_files((np_array, frame_index), y_only_q)

        return
    
    def __get_padding(self, width, height):
        i = self.config['params']['i']
        return (math.ceil(width / i) * i - width, math.ceil(height / i) * i - height)
    
    def __output_to_pngs(self, data, q):
        np_array, frame_index = data
        noise = None
        
        y = np_array[:, :, 0]
        u = np_array[:, :, 1]
        v = np_array[:, :, 2]

        if self.__noise is not None:
            noise = np.random.normal(0, 50, y.shape)
            if self.__noise == 'y':
                y = y + noise
            elif self.__noise == 'u':
                u = u + noise
            elif self.__noise == 'v':
                v = v + noise

        r = 1.164 * (y - 16) + 1.596 * (v - 128)
        g = 1.164 * (y - 16) - 0.813 * (v - 128) - 0.392 * (u - 128)
        b = 1.164 * (y - 16) + 2.017 * (u - 128)

        if self.__noise is not None:
            if self.__noise == 'r':
                r = r + noise
            elif self.__noise == 'g':
                g = g + noise
            elif self.__noise == 'b':
                b = b + noise

        r = np.clip(r, 0, 255).astype(np.uint8)
        g = np.clip(g, 0, 255).astype(np.uint8)
        b = np.clip(b, 0, 255).astype(np.uint8)
        rgb = np.stack((r, g, b), axis=-1)
        q.put((rgb, frame_index))
    
    def __output_to_video(self, data, q):
        np_array, frame_index = data

        np_array_uint8 = np.array(np_array, dtype=np.uint8)
        yuv_frame = bytearray()
        yuv_frame.extend(Identifier.FRAME.value)

        yuv_frame.extend(np_array_uint8[:, :, 0].tobytes())
        if self.__upscale == YUVFormat.YUV420:
            yuv_frame.extend(np_array_uint8[::2, ::2, 1].tobytes())
            yuv_frame.extend(np_array_uint8[::2, ::2, 2].tobytes())
        elif self.__upscale == YUVFormat.YUV422:
            yuv_frame.extend(np_array_uint8[:, ::2, 1].tobytes())
            yuv_frame.extend(np_array_uint8[:, ::2, 2].tobytes())
        else:
            yuv_frame.extend(np_array_uint8[:, :, 1].tobytes())
            yuv_frame.extend(np_array_uint8[:, :, 2].tobytes())
        q.put((frame_index, yuv_frame))
    
    def __output_to_y_only_files(self, data, q):
        np_array, frame_index = data
        # padding
        np_array_uint8 = np.array(np_array, dtype=np.uint8)
        q.put((np_array_uint8[:, :, 0], "{}.y-only".format(frame_index)))
        y_only_array = np_array[:, :, 0]
        
        pad_width, pad_height = self.__get_padding(y_only_array.shape[1], y_only_array.shape[0])
        y_only_padded = np.pad(y_only_array, ((0, pad_height), (0, pad_width)), 'constant', constant_values=128)
        
        params_i = self.config['params']['i']
        offset = y_only_padded.shape[1] // params_i
        block_size = params_i ** 2
        # combine i rows into 1 row
        # group into i x 1, with <x> channels
        a = y_only_padded.reshape(params_i, -1, params_i, 1) # (i, -1, i, <x>)
        b = []
        # for loop width // i
        # select every i-th column
        # group into one array, size = i**2 * 3
        # = one block in a row, raster order
        for i in range(offset):
            b.append(a[:, i::offset].reshape(-1, block_size * 1)) # [all rows, start::step = width // i] (-1, i**2 * <x>))

        # combine into 1 array
        # group into i x i, with <x> channels
        # each row has width // i blocks
        c = np.block(b).reshape(-1, offset, block_size) # (-1,  width // i, i**2)

        # average block
        average_block = c.mean(2).round().astype(int).reshape(-1, offset, 1).repeat(block_size, 2)

        # reshape to original form
        d = average_block.reshape(-1, offset, params_i, params_i)
        e = []
        for i in range(offset):
            e.append(d[:, i].reshape(-1, params_i))
        y_only_averaged_array = np.block(e)
        if not y_only_padded.shape == y_only_averaged_array.shape:
            raise Exception('Shape mismatch.')
        
        y_only_averaged_array_uint8 = np.array(y_only_averaged_array, dtype=np.uint8)
        q.put((y_only_averaged_array_uint8, "{}.y-only-averaged".format(frame_index)))

        # difference
        if 'diff_factor' in self.config['params']:
            y_only_diff = (y_only_padded - y_only_averaged_array) * self.config['params']['diff_factor']
            y_only_diff_uint8 = np.array(y_only_diff, dtype=np.uint8)
            q.put((y_only_diff_uint8, "{}.y-only-diff".format(frame_index)))