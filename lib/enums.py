from enum import Enum

class YUVFormat(Enum):
    YUV444 = 444
    YUV422 = 422
    YUV420 = 420

class Identifier(Enum):
    FRAME = b'FRAME\n'
    END = 0x0A.to_bytes()
    SPACER = 0x20.to_bytes()

class Intraframe(Enum):
    HORIZONTAL = 1
    VERTICAL = 0

class TypeMarker(Enum):
    I_FRAME = 1
    P_FRAME = 0

class VBSMarker(Enum):
    SPLIT = 1
    UNSPLIT = 0