from PIL import Image
import numpy as np

# im = Image.open('ffmpeg/output_0001.png') # Can be many different formats.
# pix = im.load()
# x, y = im.size # Get the width and hight of the image for iterating over
# pixel_values = list(im.getdata())
# i = 0
# for j in range(2):
#     for k in range(x):
#         r, g, b = pixel_values[i]
#         y_1 = 0.299 * r + 0.587 * g + 0.114 * b + 16
#         u = -0.169 * r - 0.331 * g + 0.5 * b + 128
#         v = 0.5 * r - 0.419 * g + 0.081 * b + 128
#         y_1 = np.clip(y_1, 0, 255).astype(np.uint8)
#         u = np.clip(u, 0, 255).astype(np.uint8)
#         v = np.clip(v, 0, 255).astype(np.uint8)
#         print(i, [y_1, u, v], pixel_values[i])
#         i += 1

c = np.array([
    [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10,11,12], [10,110,120], [10,11,12]],
    [[13, 14, 15], [16,17, 18], [19,20,21], [22,23,24], [22,230,240], [22,23,24]],
    [[1, 9, 3], [4, 8, 6], [7, 7, 9], [10,6,12], [10,5,120], [10,4,12]]
    ])
print(c[:,:,0])
print(c[::2,::2,1])
print(c[:,::2,2])