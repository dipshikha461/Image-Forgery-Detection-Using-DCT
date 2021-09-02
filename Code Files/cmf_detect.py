"""Importing necessary libraries"""

import numpy as np
from scipy import fft
import cv2
from operator import itemgetter
from quant_matrix import QuantizationMatrix

"""Defining necessary parameters"""

BLOCK_SIZE = 8                           # Block size (8x8)
QF = 0.75                                # Quality Factor
shift_thresh = 10                        # Threshold for shift vector count
stride = 1                               # Sliding window stride count / overlap
Q_8x8 = QuantizationMatrix().get_qm(QF)  # 8x8 quantization matrix based on QF

"""Reading input image"""

image = "forged3.png"  # Use an image of your choice
original_image = cv2.imread(image, cv2.IMREAD_COLOR)
image = cv2.imread(image, 0)

overlay = original_image.copy()
cv2.imshow("Original Image", original_image)

img = np.array(image)
height, width = img.shape

""" 
    a) Create sliding windows
    b) Apply dct transform to each block
    c) Quantize all dct coefficients
"""
quant_row_matrices = []  # to store quantized blocks as rows

for i in range(0, height - BLOCK_SIZE, stride):
    for j in range(0, width - BLOCK_SIZE, stride):
        block = img[i: i + BLOCK_SIZE, j: j + BLOCK_SIZE]
        dct_matrix = fft.dct(block)                           # dct
        quant_block = np.round(np.divide(dct_matrix, Q_8x8))  # quantization of dct co-effs
        block_row = list(quant_block.flatten())               # adding as rows
        quant_row_matrices.append([(i, j), block_row])        # left-corner pixel co-ordinates and block

"""Lexicographic sort"""
sorted_blocks = sorted(quant_row_matrices, key=itemgetter(1))

"""
    a)Finding matched blocks
    b)Euclidean operations for calculating shift vectors 
"""

matched_blocks = []     # FORMAT: [[block1], [block2], (pos1), (pos2), shift vector]
shift_vec_count = {}     # to keep track of sf count

for i in range(len(sorted_blocks) - 1):
    if sorted_blocks[i][1] == sorted_blocks[i + 1][1]:
        point1 = sorted_blocks[i][0]
        point2 = sorted_blocks[i + 1][0]
        s = np.linalg.norm(np.array(point1) - np.array(point2))  # shift vector
        shift_vec_count[s] = shift_vec_count.get(s, 0) + 1  # increment count for s
        matched_blocks.append([sorted_blocks[i][1], sorted_blocks[i + 1][1],
                               point1, point2, s])


"""Applying the shift vector threshold"""
matched_pixels_start = []
for sf in shift_vec_count:
    if shift_vec_count[sf] > shift_thresh:
        for row in matched_blocks:
            if sf == row[4]:
                matched_pixels_start.append([row[2], row[3]])


"""Plotting results"""
alpha = 0.5

for starting_points in matched_pixels_start:
    p1 = starting_points[0]
    p2 = starting_points[1]

    overlay[p1[0]: p1[0] + BLOCK_SIZE, p1[1]: p1[1] + BLOCK_SIZE] = (0, 0, 255)
    overlay[p2[0]: p2[0] + BLOCK_SIZE, p2[1]: p2[1] + BLOCK_SIZE] = (0, 255, 0)

cv2.addWeighted(overlay, alpha, original_image, 1, 0, original_image)
cv2.imshow("Detected Forged/Duplicated Regions", original_image)

cv2.waitKey(0)
cv2.destroyAllWindows()