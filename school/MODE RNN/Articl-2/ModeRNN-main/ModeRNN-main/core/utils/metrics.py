__author__ = 'yunbo'

import numpy as np
from scipy.signal import convolve2d


def log10(t):
    """
    Calculates the base-10 log of each element in t.
    @param t: The tensor from which to calculate the base-10 log.
    @return: A tensor with the base-10 log of each element in t.
    """

    numerator = np.log(t)
    denominator = np.log(10)
    return numerator / denominator


# def sharp_diff_error(gen_frames, gt_frames):
#     """
#     Computes the Sharpness Difference error between the generated images and the ground truth
#     images.
#     @param gen_frames: A tensor of shape [batch_size, height, width, 3]. The frames generated by the
#                        generator model.
#     @param gt_frames: A tensor of shape [batch_size, height, width, 3]. The ground-truth frames for
#                       each frame in gen_frames.
#     @return: A scalar tensor. The Sharpness Difference error over each frame in the batch.
#     """
#     shape = gen_frames.shape
#     if shape[3] > 1:
#         return 0
#     gen_frames = gen_frames[:, :, :, 0]
#     gt_frames = gt_frames[:, :, :, 0]
#     num_pixels = shape[1] * shape[2]
#
#     # gradient difference
#     # create filters [-1, 1] and [[1],[-1]] for diffing to the left and down respectively.
#     # TODO: Could this be simplified with one filter [[-1, 2], [0, -1]]?
#     pos = np.identity(3)
#     neg = -1 * pos
#     filter_x = np.expand_dims(np.stack([neg, pos], axis=0), 0)  # [-1, 1]
#     shape_tmp = filter_x.shape
#     filter_x = np.reshape(filter_x, [shape_tmp[0]*shape_tmp[1]*shape_tmp[2], shape_tmp[3]])
#     filter_y = np.stack([np.expand_dims(pos, 0), np.expand_dims(neg, 0)]) # [[1],[-1]]
#     shape_tmp = filter_y.shape
#     filter_y = np.reshape(filter_y, [shape_tmp[0] * shape_tmp[1] * shape_tmp[2], shape_tmp[3]])
#     padding = 'same'
#     grad_diff = []
#
#     for i in range(shape[0]):
#         gen_dx = np.abs(convolve2d(gen_frames[i], filter_x, mode=padding))
#         gen_dy = np.abs(convolve2d(gen_frames[i], filter_y, mode=padding))
#         gt_dx = np.abs(convolve2d(gt_frames[i], filter_x, mode=padding))
#         gt_dy = np.abs(convolve2d(gt_frames[i], filter_y, mode=padding))
#         grad_diff.append(np.sum(np.abs(gen_dx + gen_dy - gt_dx - gt_dy)))
#
#     grad_diff = np.stack(grad_diff)
#
#     batch_errors = 10 * log10(1 / ((1 / num_pixels) * grad_diff))
#     return np.mean(batch_errors)


def batch_mae_frame_float(gen_frames, gt_frames):
    # [batch, width, height] or [batch, width, height, channel]
    if gen_frames.ndim == 3:
        axis = (1, 2)
    elif gen_frames.ndim == 4:
        axis = (1, 2, 3)
    x = np.float32(gen_frames)
    y = np.float32(gt_frames)
    mae = np.sum(np.absolute(x - y), axis=axis, dtype=np.float32)
    return np.mean(mae)


def batch_psnr(gen_frames, gt_frames):
    # [batch, width, height] or [batch, width, height, channel]
    if gen_frames.ndim == 3:
        axis = (1, 2)
    elif gen_frames.ndim == 4:
        axis = (1, 2, 3)
    x = np.int32(gen_frames)
    y = np.int32(gt_frames)
    num_pixels = float(np.size(gen_frames[0]))
    mse = np.sum((x - y) ** 2, axis=axis, dtype=np.float32) / num_pixels
    psnr = 20 * np.log10(255) - 10 * np.log10(mse)
    return np.mean(psnr)


def batch_diff_sharp(gen_frames, gt_frames):
    # [batch, width, height] or [batch, width, height, channel]
    num_pixels = float(np.size(gen_frames[0]))
    mse = []
    for b in range(gen_frames.shape[0]):
        delta_gt = np.abs(gt_frames[b, 1:] - gt_frames[b, :-1])[:, 1:] + \
                   np.abs(gt_frames[b, :, 1:] - gt_frames[b, :, :-1])[1:]
        delta_gen = np.abs(gen_frames[b, 1:] - gen_frames[b, :-1])[:, 1:] + \
                    np.abs(gen_frames[b, :, 1:] - gen_frames[b, :, :-1])[1:]
        sum = np.sum(np.abs(delta_gt - delta_gen)) / num_pixels
        mse.append(sum)
    mse = np.stack(mse)
    psnr = 20 * np.log10(255) - 10 * np.log10(mse)
    return np.mean(psnr)


def batch_diff_sharp_yunbo(gen_frames, gt_frames):
    # [batch, width, height] or [batch, width, height, channel]
    out = []
    shape = gen_frames.shape
    for b in range(gen_frames.shape[0]):
        t1 = np.abs(gt_frames[b, 1:, :-1] - gt_frames[b, :-1, :-1])
        t2 = np.abs(gt_frames[b, :-1, 1:] - gt_frames[b, :-1, :-1])
        t3 = np.abs(gen_frames[b, 1:, :-1] - gen_frames[b, :-1, :-1])
        t4 = np.abs(gen_frames[b, :-1, 1:] - gen_frames[b, :-1, :-1])
        N = (shape[-3] - 1) * (shape[-2] - 1) * shape[-1]
        out_tmp = np.sum(np.abs((t1 + t2) - (t3 + t4))) / N
        out.append(10 * np.log10(255 * 255 / out_tmp))
    out = np.stack(out)
    out = np.mean(out)
    return out


def batch_gdl(gen_frames, gt_frames):
    # [batch, width, height] or [batch, width, height, channel]
    t1 = np.power(np.absolute(gt_frames[:, 1:, :, :] - gt_frames[:, :-1, :, :]) -
                  np.absolute(gen_frames[:, 1:, :, :] - gen_frames[:, :-1, :, :]), 2)
    t2 = np.power(np.absolute(gt_frames[:, :, :-1, :] - gt_frames[:, :, 1:, :]) -
                  np.absolute(gen_frames[:, :, :-1, :] - gen_frames[:, :, 1:, :]), 2)
    return np.mean(t1[:, :, 1:] + t2[:, 1:])


def cal_csi(pd, gt, level):
    # [w,h]

    pdf = pd.astype(np.float32)
    gtf = gt.astype(np.float32)
    # print (np.mean(pdf))
    # print (np.mean(gtf))
    pd_ = np.zeros(pd.shape)
    gt_ = np.zeros(gt.shape)
    pd_[(pdf + 30) / 2 >= level] = 1
    gt_[(gtf + 30) / 2 >= level] = 1
    csi_ = pd_ + gt_
    if (csi_ >= 1).sum() == 0:
        return 0.0
    return float((csi_ == 2).sum()) / float((csi_ >= 1).sum())