import numpy as np
from lib.block_processing import *

width = 12
height = 12
i = 4
block_size = i ** 2
q = 1 # rgb or index
offset = width // i

a = np.arange(width*height*1).reshape(width, height, 1).repeat(3, 2)
"""
a = [
    [[  0   0   0] [  1   1   1] [  2   2   2] [  3   3   3] [  4   4   4] [  5   5   5] [  6   6   6] [  7   7   7] [  8   8   8] [  9   9   9] [ 10  10  10] [ 11  11  11]]
    [[ 12  12  12] [ 13  13  13] [ 14  14  14] [ 15  15  15] [ 16  16  16] [ 17  17  17] [ 18  18  18] [ 19  19  19] [ 20  20  20] [ 21  21  21] [ 22  22  22] [ 23  23  23]]
    [[ 24  24  24] [ 25  25  25] [ 26  26  26] [ 27  27  27] [ 28  28  28] [ 29  29  29] [ 30  30  30] [ 31  31  31] [ 32  32  32] [ 33  33  33] [ 34  34  34] [ 35  35  35]]
    [[ 36  36  36] [ 37  37  37] [ 38  38  38] [ 39  39  39] [ 40  40  40] [ 41  41  41] [ 42  42  42] [ 43  43  43] [ 44  44  44] [ 45  45  45] [ 46  46  46] [ 47  47  47]]
    [[ 48  48  48] [ 49  49  49] [ 50  50  50] [ 51  51  51] [ 52  52  52] [ 53  53  53] [ 54  54  54] [ 55  55  55] [ 56  56  56] [ 57  57  57] [ 58  58  58] [ 59  59  59]]
    [[ 60  60  60] [ 61  61  61] [ 62  62  62] [ 63  63  63] [ 64  64  64] [ 65  65  65] [ 66  66  66] [ 67  67  67] [ 68  68  68] [ 69  69  69] [ 70  70  70] [ 71  71  71]]
    [[ 72  72  72] [ 73  73  73] [ 74  74  74] [ 75  75  75] [ 76  76  76] [ 77  77  77] [ 78  78  78] [ 79  79  79] [ 80  80  80] [ 81  81  81] [ 82  82  82] [ 83  83  83]]
    [[ 84  84  84] [ 85  85  85] [ 86  86  86] [ 87  87  87] [ 88  88  88] [ 89  89  89] [ 90  90  90] [ 91  91  91] [ 92  92  92] [ 93  93  93] [ 94  94  94] [ 95  95  95]]
    [[ 96  96  96] [ 97  97  97] [ 98  98  98] [ 99  99  99] [100 100 100] [101 101 101] [102 102 102] [103 103 103] [104 104 104] [105 105 105] [106 106 106] [107 107 107]]
    [[108 108 108] [109 109 109] [110 110 110] [111 111 111] [112 112 112] [113 113 113] [114 114 114] [115 115 115] [116 116 116] [117 117 117] [118 118 118] [119 119 119]]
    [[120 120 120] [121 121 121] [122 122 122] [123 123 123] [124 124 124] [125 125 125] [126 126 126] [127 127 127] [128 128 128] [129 129 129] [130 130 130] [131 131 131]]
    [[132 132 132] [133 133 133] [134 134 134] [135 135 135] [136 136 136] [137 137 137] [138 138 138] [139 139 139] [140 140 140] [141 141 141] [142 142 142] [143 143 143]]
]

a[:, :, 0] = [
    [  0   1   2   3   4   5   6   7   8   9  10  11]
    [ 12  13  14  15  16  17  18  19  20  21  22  23]
    [ 24  25  26  27  28  29  30  31  32  33  34  35]
    [ 36  37  38  39  40  41  42  43  44  45  46  47]
    [ 48  49  50  51  52  53  54  55  56  57  58  59]
    [ 60  61  62  63  64  65  66  67  68  69  70  71]
    [ 72  73  74  75  76  77  78  79  80  81  82  83]
    [ 84  85  86  87  88  89  90  91  92  93  94  95]
    [ 96  97  98  99 100 101 102 103 104 105 106 107]
    [108 109 110 111 112 113 114 115 116 117 118 119]
    [120 121 122 123 124 125 126 127 128 129 130 131]
    [132 133 134 135 136 137 138 139 140 141 142 143]
]
"""
# Main code:
# combine i rows into 1 row
# group into i x 1, with <x> channels
# for Y only
b = a[:, :, 0].reshape(i, -1, i, 1) # (i, -1, i, x)
"""
b = [
    [
        [[  0]  [  1]  [  2]  [  3]]
        [[  4]  [  5]  [  6]  [  7]]
        [[  8]  [  9]  [ 10]  [ 11]]
        [[ 12]  [ 13]  [ 14]  [ 15]]
        [[ 16]  [ 17]  [ 18]  [ 19]]
        [[ 20]  [ 21]  [ 22]  [ 23]]
        [[ 24]  [ 25]  [ 26]  [ 27]]
        [[ 28]  [ 29]  [ 30]  [ 31]]
        [[ 32]  [ 33]  [ 34]  [ 35]]
    ]
    [
        [[ 36]  [ 37]  [ 38]  [ 39]]
        [[ 40]  [ 41]  [ 42]  [ 43]]
        [[ 44]  [ 45]  [ 46]  [ 47]]
        [[ 48]  [ 49]  [ 50]  [ 51]]
        [[ 52]  [ 53]  [ 54]  [ 55]]
        [[ 56]  [ 57]  [ 58]  [ 59]]
        [[ 60]  [ 61]  [ 62]  [ 63]]
        [[ 64]  [ 65]  [ 66]  [ 67]]
        [[ 68]  [ 69]  [ 70]  [ 71]]
    ]
    [
        [[ 72]  [ 73]  [ 74]  [ 75]]
        [[ 76]  [ 77]  [ 78]  [ 79]]
        [[ 80]  [ 81]  [ 82]  [ 83]]
        [[ 84]  [ 85]  [ 86]  [ 87]]
        [[ 88]  [ 89]  [ 90]  [ 91]]
        [[ 92]  [ 93]  [ 94]  [ 95]]
        [[ 96]  [ 97]  [ 98]  [ 99]]
        [[100]  [101]  [102]  [103]]
        [[104]  [105]  [106]  [107]]
    ]
    [
        [[108]  [109]  [110]  [111]]
        [[112]  [113]  [114]  [115]]
        [[116]  [117]  [118]  [119]]
        [[120]  [121]  [122]  [123]]
        [[124]  [125]  [126]  [127]]
        [[128]  [129]  [130]  [131]]
        [[132]  [133]  [134]  [135]]
        [[136]  [137]  [138]  [139]]
        [[140]  [141]  [142]  [143]]
    ]
]
"""
c = []

# for loop width // i
# select every i-th column
# group into one array, size = i**2 * 3
# = one block in a row, raster order
for j in range(offset):
    c.append(b[:, j::offset].reshape(-1, block_size * 1))
    # separate into 3 channels if needed
    # e = d.reshape(-1, block_size, q) # (-1, i**2, 3)
    #average d.mean(1).round().astype(np.uint8)

# combine f, g, h into 1 array
# group into i x i, with 3 channels
# each row has width // i blocks
f = np.block(c).reshape(-1, offset, block_size) # (-1,  width // i, i**2, 3)
"""
f = [
    [
        [  0   1   2   3  12  13  14  15  24  25  26  27  36  37  38  39]
        [  4   5   6   7  16  17  18  19  28  29  30  31  40  41  42  43]
        [  8   9  10  11  20  21  22  23  32  33  34  35  44  45  46  47]
    ]
    [
        [ 48  49  50  51  60  61  62  63  72  73  74  75  84  85  86  87]
        [ 52  53  54  55  64  65  66  67  76  77  78  79  88  89  90  91]
        [ 56  57  58  59  68  69  70  71  80  81  82  83  92  93  94  95]
    ]
    [
        [ 96  97  98  99 108 109 110 111 120 121 122 123 132 133 134 135]
        [100 101 102 103 112 113 114 115 124 125 126 127 136 137 138 139]
        [104 105 106 107 116 117 118 119 128 129 130 131 140 141 142 143]
    ]
]
"""
# print(a)
# print(b)
# print(f)
g = f.mean(2).round().astype(int).reshape(-1, offset, 1).repeat(block_size, 2)
# print(g)
"""
g = [
    [
        [ 20  20  20  20  20  20  20  20  20  20  20  20  20  20  20  20]
        [ 24  24  24  24  24  24  24  24  24  24  24  24  24  24  24  24]
        [ 28  28  28  28  28  28  28  28  28  28  28  28  28  28  28  28]
    ]
    [
        [ 68  68  68  68  68  68  68  68  68  68  68  68  68  68  68  68]
        [ 72  72  72  72  72  72  72  72  72  72  72  72  72  72  72  72]
        [ 76  76  76  76  76  76  76  76  76  76  76  76  76  76  76  76]
    ]
    [
        [116 116 116 116 116 116 116 116 116 116 116 116 116 116 116 116]
        [120 120 120 120 120 120 120 120 120 120 120 120 120 120 120 120]
        [124 124 124 124 124 124 124 124 124 124 124 124 124 124 124 124]
    ]
]
"""

h = g.reshape(-1, offset, i, i)
# print(h)
j = []
for k in range(offset):
    j.append(h[:,k].reshape(-1, i))
l = np.block(j)
# print(l)
"""
l = [
    [ 20  20  20  20  24  24  24  24  28  28  28  28]
    [ 20  20  20  20  24  24  24  24  28  28  28  28]
    [ 20  20  20  20  24  24  24  24  28  28  28  28]
    [ 20  20  20  20  24  24  24  24  28  28  28  28]
    [ 68  68  68  68  72  72  72  72  76  76  76  76]
    [ 68  68  68  68  72  72  72  72  76  76  76  76]
    [ 68  68  68  68  72  72  72  72  76  76  76  76]
    [ 68  68  68  68  72  72  72  72  76  76  76  76]
    [116 116 116 116 120 120 120 120 124 124 124 124]
    [116 116 116 116 120 120 120 120 124 124 124 124]
    [116 116 116 116 120 120 120 120 124 124 124 124]
    [116 116 116 116 120 120 120 120 124 124 124 124]
]
"""

# print(np.array_equal(f, block_create(a[:,:,0], i)[0]))
# print(np.array_equal(l, pixel_create(h, (height, width), i)))


# params_r = 4
# params_i = 2
# f1 = a
# f1_reconstructed = calc_motion_vector_helper(f1, 0, np.full(height*width, 128).reshape(height, width), params_i, params_r, None)
# print(f1_reconstructed)
# f2 = np.roll(a, 1, axis=1)
# print(f2[:,:,0])
# f2_reconstructed = calc_motion_vector_helper(f2, 1, f1_reconstructed, params_i, params_r, None)
# print(f2_reconstructed)

"""
array([210, 211, 205, 210])
array([[196, 172, 194, 212, 208, 212, 204, 176],
       [190, 188, 188, 204, 208, 212, 208, 176],
       [154, 200, 200, 200, 204, 212, 208, 176],
       [104, 164, 200, 198, 202, 212, 208, 172],
       [104, 120, 176, 200, 200, 208, 204, 172],
       [116, 104, 124, 180, 204, 208, 204, 172]])
"""
b = a[:,:,0]
c = a[:,:,1]
d = a[:,:,2]
e = np.stack((c, d), axis=2)
