import numpy as np
import cv2

def blur_img(img, kernel_size, sigma):
    result = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
    return result


def generate_gaussian_stack(img, kernel_size, sigma, stack_size):
    result = []
    result.append(img)
    temp_img = img.copy()
    for i in range(1, stack_size):
        temp_img = blur_img(temp_img, kernel_size, sigma)
        result.append(temp_img)
    return result

def generate_laplacian_stack(img, kernel_size, sigma, stack_size):
    gaussian_pyramid = generate_gaussian_stack(img, kernel_size, sigma, stack_size)
    result = []
    for i in range(0, stack_size - 1):
        temp = gaussian_pyramid[i].copy() - gaussian_pyramid[i + 1].copy()
        result.append(temp)
    result.append(gaussian_pyramid[-1].copy())
    return result


def generate_masks(mask, kernel_size, sigma, size):
    result = []
    new_mask = blur_img(mask, kernel_size, sigma)
    result.append(new_mask)
    for i in range(size):
        new_mask = blur_img(new_mask, kernel_size, sigma)
        result.append(new_mask)

    return result


def generate_blended_stack(img1, img2, masks, kernel_size, sigma, stack_size):
    result = []
    laplacian_stack1 = generate_laplacian_stack(img1, kernel_size, sigma, stack_size)
    laplacian_stack2 = generate_laplacian_stack(img2, kernel_size, sigma, stack_size)
    for i in range(stack_size):
        blended_img = laplacian_stack1[i] * masks[i] + laplacian_stack2[i] * (1 - masks[i])
        result.append(blended_img)
    return result


def blend_images(img1, img2, masks, kernel_size, sigma, stack_size):
    blended_stack = generate_blended_stack(img1, img2, masks, kernel_size, sigma, stack_size)
    result = blended_stack[0].copy()
    for i in range(1, stack_size):
        result = result + blended_stack[i]
    return result


def convert_from_float32_to_uint8(img):
    max_index, min_index = np.max(img), np.min(img)
    a = 255 / (max_index - min_index)
    b = 255 - a * max_index
    result = (a * img + b).astype(np.uint8)
    return result
