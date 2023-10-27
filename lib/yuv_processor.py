import pathlib
import math
from lib.config.config import Config
from lib.utils.enums import YUVFormat, Identifier
from lib.utils.misc import get_padding

from lib.frame_processing import upscale
from lib.multi_processing import MultiProcessingNew as MP

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
    
    def __init__(self, config_path: str):
        self.__config = Config(config_path, True)
        self.file_path = self.__config.input
        self.file = pathlib.Path.cwd().joinpath(self.file_path)
        if not self.file.exists():
            raise FileNotFoundError('File not found.')
        self.__mp = MP(self.__config)
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
        self.__stop_at = self.__config.params.stop_at
        self.upscale = self.__config.upscale
        if self.upscale:
            if self.upscale == 420:
                self.upscale = YUVFormat.YUV420
            elif self.upscale == 422:
                self.upscale = YUVFormat.YUV422
            else:
                self.upscale = YUVFormat.YUV444
        self.__offsets = None
        self.__deconstruct()
        return
    
    """
        Deconstruct the YUV file into its components.
    """
    def __deconstruct(self) -> None:
        self.__read_header()
        self.__read_frames()
        self.info['frame_count'] = self.__frame_index
        paded_width, paded_height = get_padding(self.info['width'], self.info['height'], self.__config.params.i)
        self.__config.output_path.meta_file.write_text(str("{},{},{},{},{}".format(self.__frame_index, paded_height, paded_width, self.__config.params.i, self.__config.params.qp)))

        self.__mp.done()
        return
    
    """
        Read the header of the YUV file.
    """
    def __read_header(self) -> None:
        # raw_header = bytearray()
        # while self.__read_byte() != Identifier.END.value:
        #     raw_header.extend(self.__byte)
        #     if self.__byte == Identifier.SPACER.value:
        #         self.HEADER_IDENTIFIERS['__order'].pop(0)
        #         continue

        #     order = self.HEADER_IDENTIFIERS['__order'][0]
        #     current_decoded_byte = self.__decode_current_byte()
        #     if order == 'FORMAT':
        #         self.info['format'] += current_decoded_byte
        #     elif order == 'WIDTH':
        #         if self.__byte == self.HEADER_IDENTIFIERS['WIDTH']:
        #             continue
        #         self.info['width'] += current_decoded_byte
        #     elif order == 'HEIGHT':
        #         if self.__byte == self.HEADER_IDENTIFIERS['HEIGHT']:
        #             continue
        #         self.info['height'] += current_decoded_byte
        #     elif order == 'FRAMERATE':
        #         if self.__byte == self.HEADER_IDENTIFIERS['FRAMERATE']:
        #             continue
        #         self.info['framerate'] += current_decoded_byte
        #     elif order == 'INTERLACE':
        #         if self.__byte == self.HEADER_IDENTIFIERS['INTERLACE']:
        #             continue
        #         self.info['interlace'] += current_decoded_byte
        #     elif order == 'ASPECT_RATIO':
        #         if self.__byte == self.HEADER_IDENTIFIERS['ASPECT_RATIO']:
        #             continue
        #         self.info['aspect_ratio'] += current_decoded_byte
        #     elif order == 'COLOR_SPACE':
        #         if self.__byte == self.HEADER_IDENTIFIERS['COLOR_SPACE']:
        #             continue
        #         self.info['color_space'] += current_decoded_byte
        #     elif order == 'COMMENT':
        #         if self.__byte == self.HEADER_IDENTIFIERS['COMMENT']:
        #             continue
        #         self.info['comment'] += current_decoded_byte
        #     else:
        #         raise Exception('Invalid header identifier.')
        
        self.info['width'] = int(self.__config.video_params.width)
        self.info['height'] = int(self.__config.video_params.height)
        # if self.info['framerate'] in self.FRAMERATE_PREDEFINED:
        #     self.info['framerate'] = self.FRAMERATE_PREDEFINED[self.info['framerate']]
        # if self.info['interlace'] in self.INTERLACE_PREDEFINED:
        #     self.info['interlace'] = self.INTERLACE_PREDEFINED[self.info['interlace']]
        # if self.info['aspect_ratio'] in self.ASPECT_RATIO_PREDEFINED:
        #     self.info['aspect_ratio'] = self.ASPECT_RATIO_PREDEFINED[self.info['aspect_ratio']]
        # if self.info['color_space'] in self.COLOR_SPACE_PREDEFINED:
        #     self.info['color_space'] = self.COLOR_SPACE_PREDEFINED[self.info['color_space']]
        # else:
            # self.info['color_space'] = str(YUVFormat.YUV420.value)

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

        # raw_header.extend(Identifier.SPACER.value) # add colorspace
        # raw_header.extend(self.HEADER_IDENTIFIERS['COLOR_SPACE'])
        # raw_header.extend(bytes(str(self.upscale.value), 'ascii')) # add upscale
        # raw_header.extend(self.__byte) # add END_IDENTIFIER
        self.__mp.signal_q.put((self.info['height'], self.info['width']))
        return
    
    """
        Read the frames of the YUV file, and then upscale to YUV444.
    """
    def __read_frames(self) -> None:
        self.__read_byte() # skil END_IDENTIFIER, after this line, self.__byte == b'F'
        # while self.__byte != Identifier.END.value:
        #     # skip first "FRAME"
        #     # when exit, self.__byte == self.END_IDENTIFIER
        #     self.__read_byte()
        index = 0
        while not self.__eof():
            
            if self.__frame_index == self.__stop_at:
                break

            result = bytearray()
            # while not ((len(result) > 6 and result[-len(Identifier.FRAME.value):] == Identifier.FRAME.value) or self.__eof()):
            while index != self.__offsets[0] + self.__offsets[1] + self.__offsets[2]:
                # read yuv components + "FRAME"
                # when exit, self.__byte == self.END_IDENTIFIER
                result.extend(self.__read_byte())
                index += 1
            index = 0

            # if self.__eof():
            #     # end of file
            #     yuv_components = result
            # else:
            #     # has next "FRAME"
            #     yuv_components = result[:-len(Identifier.FRAME.value)]
            yuv_components = result

            # process each frame
            self.__mp.dispatch(
                upscale, 
                (
                    (self.info['width'], self.__frame_index, self.__offsets, yuv_components, self.__format),
                    self.__config,
                )
            )
            
            print(self.__frame_index)
            self.__frame_index += 1

        return
    
    """
        Return the offsets of the YUV file.

        Returns:
            (y, u, v) (tuple): The offsets of the YUV file.
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
    def append_to_header_info(self, key: str, value: str) -> None:
        self.info[key] += value
        return
