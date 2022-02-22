import numpy as np
import cv2
import scipy.sparse.linalg as sparse_la
from scipy import sparse


def get_laplacian(img):
    kernel = np.asarray([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype='float32')
    result = cv2.filter2D(img, ddepth=-1, kernel=kernel)
    return result


def generate_matrix_b(source, target, mask):
    source_laplacian_flatten = get_laplacian(source).flatten('C')
    target_flatten = target.flatten('C')
    mask_flatten = mask.flatten('C')
    b = (mask_flatten) * source_laplacian_flatten + (1 - mask_flatten) * target_flatten
    return b


def generate_matrix_A(mask):
    data, cols, rows = [], [], []
    h, w = mask.shape[0], mask.shape[1]
    mask_flatten = mask.flatten('C')
    zeros = np.where(mask_flatten == 0)
    ones = np.where(mask_flatten == 1)
    # adding ones to data
    n = zeros[0].size
    data.extend(np.ones(n, dtype='float32').tolist())
    rows.extend(zeros[0].tolist())
    cols.extend(zeros[0].tolist())

    # adding 4s to data
    m = ones[0].size
    data.extend((np.ones(m, dtype='float32') * (4)).tolist())
    rows.extend(ones[0].tolist())
    cols.extend(ones[0].tolist())

    # adding -1s
    data.extend((np.ones(m, dtype='float32') * (-1)).tolist())
    rows.extend(ones[0].tolist())
    cols.extend((ones[0] - 1).tolist())

    data.extend((np.ones(m, dtype='float32') * (-1)).tolist())
    rows.extend(ones[0].tolist())
    cols.extend((ones[0] + 1).tolist())

    data.extend((np.ones(m, dtype='float32') * (-1)).tolist())
    rows.extend(ones[0].tolist())
    cols.extend((ones[0] - w).tolist())

    data.extend((np.ones(m, dtype='float32') * (-1)).tolist())
    rows.extend(ones[0].tolist())
    cols.extend((ones[0] + w).tolist())
    return data, cols, rows


def solve_sparse_linear_equation(data, cols, rows, b, h, w):
    sparse_matrix = sparse.csr_matrix((data, (rows, cols)), shape=(h * w, h * w), dtype='float32')
    f = sparse_la.spsolve(sparse_matrix, b)
    f = np.reshape(f, (h, w)).astype('float32')
    return f


def blend_images(source, target, mask):
    h, w = source.shape[0], source.shape[1]
    source_b, source_g, source_r = cv2.split(source)
    target_b, target_g, target_r = cv2.split(target)
    data, cols, rows = generate_matrix_A(mask)
    b_b = generate_matrix_b(source_b, target_b, mask)
    b_g = generate_matrix_b(source_g, target_g, mask)
    b_r = generate_matrix_b(source_r, target_r, mask)
    blended_b = solve_sparse_linear_equation(data, cols, rows, b_b, h, w)
    blended_g = solve_sparse_linear_equation(data, cols, rows, b_g, h, w)
    blended_r = solve_sparse_linear_equation(data, cols, rows, b_r, h, w)
    result = cv2.merge((blended_b, blended_g, blended_r))
    return result
