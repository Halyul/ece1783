import numpy as np
import math

width = 12
height = 12
i = 4
block_size = i ** 2
q = 1 # rgb or index
offset = width // i

a = np.arange(width*height*1).reshape(width, height, 1).repeat(3, 2)
# print(a[:, :, 0])
# Main code:
# combine i rows into 1 row
# group into i x 1, with <x> channels
# for Y only
b = a[:, :, 0].reshape(i, -1, i, 1) # (i, -1, i, x)
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
# print(a)
# print(b)
# print(f)
g = f.mean(2).round().astype(int).reshape(-1, offset, 1).repeat(block_size, 2)
# print(g)

h = g.reshape(-1, offset, i, i)
print(h)
j = []
for k in range(offset):
    j.append(h[:,k].reshape(-1, i))
l = np.block(j)
print(l)




x = [
    [ 20,  20,  20,  20,  24,  24,  24,  24,  28,  28,  28,  28],
    [ 20,  20,  20,  20,  24,  24,  24,  24,  28,  28,  28,  28],
    [ 20,  20,  20,  20,  24,  24,  24,  24,  28,  28,  28,  28],
    [ 20,  20,  20,  20,  24,  24,  24,  24,  28,  28,  28,  28],
    [ 68,  68,  68,  68,  72,  72,  72,  72,  76,  76,  76,  76],
    [ 68,  68,  68,  68,  72,  72,  72,  72,  76,  76,  76,  76],
    [ 68,  68,  68,  68,  72,  72,  72,  72,  76,  76,  76,  76],
    [ 68,  68,  68,  68,  72,  72,  72,  72,  76,  76,  76,  76],
    [116, 116, 116, 116, 120, 120, 120, 120, 124, 124, 124, 124],
    [116, 116, 116, 116, 120, 120, 120, 120, 124, 124, 124, 124],
    [116, 116, 116, 116, 120, 120, 120, 120, 124, 124, 124, 124],
    [116, 116, 116, 116, 120, 120, 120, 120, 124, 124, 124, 124]
]




# for array a[b][c][d]
# numpy array axis=0 a[<axis>][c][d]
# numpy array axis=1 a[b][<axis>][d]
# numpy array axis=2 a[b][c][<axis>]
# print("padding")
# x = 11
# print("{x}x{x}, i = 2, pad=".format(x=x), math.ceil(x / 2) * 2 - x)
# print("{x}x{x}, i = 4, pad=".format(x=x), math.ceil(x / 4) * 4 - x)
# x = 10
# print("{x}x{x}, i = 2, pad=".format(x=x), math.ceil(x / 2) * 2 - x)
# print("{x}x{x}, i = 4, pad=".format(x=x), math.ceil(x / 4) * 4 - x)
# x = 64
# print("{x}x{x}, i = 2, pad=".format(x=x), math.ceil(x / 2) * 2 - x)
# print("{x}x{x}, i = 4, pad=".format(x=x), math.ceil(x / 4) * 4 - x)
# print("{x}x{x}, i = 64, pad=".format(x=x), math.ceil(x / 64) * 64 - x)
# x = 100
# print("{x}x{x}, i = 2, pad=".format(x=x), math.ceil(x / 2) * 2 - x)
# print("{x}x{x}, i = 4, pad=".format(x=x), math.ceil(x / 4) * 4 - x)
# print("{x}x{x}, i = 64, pad=".format(x=x), math.ceil(x / 64) * 64 - x)

# y = 64
# print("352x288, i = {y}, 352, pad=".format(y=y), math.ceil(352 / y) * y - 352)
# print("352x288, i = {y}, 288, pad=".format(y=y), math.ceil(288 / y) * y - 288)

# y = 4
# print("354x209, i = {y}, 354, pad=".format(y=y), math.ceil(354 / y) * y - 354)
# print("354x209, i = {y}, 209, pad=".format(y=y), math.ceil(209 / y) * y - 209)

# y = 2
# print("351x208, i = {y}, 351, pad=".format(y=y), math.ceil(351 / y) * y - 351)
# print("351x208, i = {y}, 208, pad=".format(y=y), math.ceil(208 / y) * y - 208)