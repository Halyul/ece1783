import pathlib
import shutil
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
        '420jpeg': 'YUV 4:2:0 JPEG',
        '420paldv': 'YUV 4:2:0 PAL-DV',
        '420': 'YUV 4:2:0',
        '422': 'YUV 4:2:2',
        '444': 'YUV 4:4:4',
        'mono': 'YCbCr plane only',
    }
    
    def __init__(self, config_path):
        self.config = Config(config_path).config
        self.file_path = self.config['input']
        self.file = pathlib.Path.cwd().joinpath(self.file_path)
        if not self.file.exists():
            raise FileNotFoundError('File not found.')
        self.__output_path = pathlib.Path.cwd().joinpath(self.config['output_video']) if 'output_video' in self.config else None
        self.__png_path = pathlib.Path.cwd().joinpath(self.config['output_pngs']) if 'output_pngs' in self.config is not None else None
        self.__mp = MultiProcessFW(self.__output_path, self.__png_path)
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
        self.__frame_processor = FrameProcessing(self.upscale, self.config['noise'] if 'noise' in self.config else None)
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
            self.__offsets = (self.info['width'] * self.info['height'], self.info['width'] * self.info['height'] // 2)
        elif self.info['color_space'] == str(YUVFormat.YUV444.value):
            self.__format = YUVFormat.YUV444
        else:
            self.__format = YUVFormat.YUV420
        self.__offsets = self.__get_offsets()

        raw_header.extend(Identifier.SPACER.value) # add colorspace
        raw_header.extend(self.HEADER_IDENTIFIERS['COLOR_SPACE'])
        raw_header.extend(bytes(str(self.upscale.value), 'ascii')) # add upscale
        raw_header.extend(self.__byte) # add END_IDENTIFIER
        self.__mp.add_to_output_q(raw_header)
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

class MultiProcessFW:

    def __init__(self, output_path, png_path) -> None:
        self.manager = mp.Manager()
        self.pool = mp.Pool(mp.cpu_count() + 2)
        self.jobs = []
        self.__output_path = output_path
        self.__png_path = png_path

        self.output_q = None
        if self.__output_path is not None:
            self.output_q = self.manager.Queue()
            self.output_watcher = self.pool.apply_async(self.write_raw_bytes_to_file, (self.__output_path, self.output_q,))
            self.__clear_output_path()

        self.png_q = None
        if self.__png_path is not None:
            self.png_q = self.manager.Queue()
            self.png_watcher = self.pool.apply_async(self.write_to_png, (self.__png_path, self.png_q,))
            self.__clear_png_path()
            pathlib.Path.cwd().joinpath(self.__png_path).mkdir(parents=True, exist_ok=True)

    """
        Clear the output video path.
    """
    def __clear_output_path(self):
        if self.__output_path is not None:
            pathlib.Path.cwd().joinpath(self.__output_path.parent).mkdir(parents=True, exist_ok=True)
            if self.__output_path.exists():
                self.__output_path.unlink()
                self.__output_path.write_bytes(b'')
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
        Run a function in parallel.

        Parameters:
            func (function): The function to be run in parallel.
            data (tuple): The data to be passed to the function.
            debug (bool): True if debug mode is on.
    """
    def dispatch(self, func, data, debug=False):
        if debug:
            func(data, self.output_q, self.png_q)
        else:
            job = self.pool.apply_async(func=func, args=(data, self.output_q, self.png_q,))
            self.jobs.append(job)
        return
    
    """
        Add data to output video queue.

        Parameters:
            data (bytearray): The data to be added to the queue.
    """
    def add_to_output_q(self, data):
        if self.__output_path is not None:
            self.output_q.put((-1, data))
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
        
        if self.__output_path is not None:
            self.output_q.put('kill')
        if self.__png_path is not None:
            self.png_q.put('kill')
        self.pool.close()
        self.pool.join()

        return
    
class FrameProcessing:

    def __init__(self, upscale, noise=None) -> None:
        self.__upscale = upscale
        self.__noise = noise

    """
        Upscale the frame to YUV444.

        Parameters:
            data (tuple): The data to be processed.
            output_q (mp.Queue or None): The queue to put the output video data if provided.
            png_q (mp.Queue or None): The queue to put the output png.
    """
    def upscale(self, data, output_q, png_q):
        (width, frame_index, format_tuple, yuv_components, mode) = data

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
            png_q.put((rgb, frame_index))

        # output video
        if output_q is not None:
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
            output_q.put((frame_index, yuv_frame))

        return