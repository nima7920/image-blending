import numpy as np
import cv2
import poisson_blend
import multiresolution_blend

''' applying poisson blending '''
# reading images and scaling them to [0,1]
img1 = cv2.imread("inputs/cards.jpg")
img2 = cv2.imread("inputs/desk.jpg")

img1 = (img1 / 255).astype('float32')
img2 = (img2 / 255).astype('float32')

source = img1[195:485, 355:715, :].astype('float32')
target = img2[195:485, 355:715, :].astype('float32')

mask = np.zeros((290, 360), dtype='float32')
mask[5:-5, 5:-5] = 1

max_matrix, min_matrix = np.ones(source.shape, dtype='float32') * 255, np.zeros(source.shape, dtype='float32')
result = poisson_blend.blend_images(source, target, mask) * 255
result = np.minimum(result, max_matrix)
result = np.maximum(result, min_matrix)
result = result.astype('uint8')
final_result = (img2.copy() * 255).astype('uint8')
final_result[195:485, 355:715, :] = result
cv2.imwrite("outputs/cards-on-desk.jpg", final_result)

''' applying multiresolution blending '''
img1 = cv2.imread("inputs/tree_spring.jpg")
img2 = cv2.imread("inputs/tree_fall.jpg")

img_float1 = (img1 / 255).astype('float32')
img_float2 = (img2 / 255).astype('float32')

kernel_size, sigma, n = 151, 30, 10
img_list = multiresolution_blend.generate_gaussian_stack(img1, kernel_size, sigma, n)
lap_list = multiresolution_blend.generate_laplacian_stack(img1, kernel_size, sigma, n)

''' creating masks '''
h, w = img1.shape[0], img1.shape[1]
mask = np.zeros(img1.shape, dtype='float32')
mask[:, 0:w//2] = 1
masks = multiresolution_blend.generate_masks(mask, 151, 30, n)

result = multiresolution_blend.blend_images(img_float1, img_float2, masks, kernel_size, sigma, n)
result = multiresolution_blend.convert_from_float32_to_uint8(result)
cv2.imwrite("outputs/tree-spring-fall.jpg", result)
