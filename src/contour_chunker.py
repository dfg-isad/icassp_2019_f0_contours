import numpy as np


def buffer_signal(x, blocksize, hopsize):
    num_samples = len(x)
    num_frames = int(np.fix((num_samples-(blocksize-hopsize))/hopsize))

    buffer_idx = np.tile(np.arange(blocksize)[:, np.newaxis], (1, num_frames)) + \
                 np.tile(np.arange(num_frames)*hopsize, (blocksize, 1))

    return x[buffer_idx]


def chunk_contours_to_unit_length(contours,
                                  targets,
                                  block_size,
                                  hop_size):
    """ Cut existing contours with moving window into unit-length contours and stack those
        as feature matrix. Contour targets are transferred accordingly.
    Args:
        contours (list of np.ndarray): List of variable length contous
        target (np.ndarray): Target values per contour
        block_size (int): Window size in samples
        hop_size (int): Hop size in samples
    Returns:
        feat_mat (2d np.ndarray): Feature matrix (rows corresponds to chunks)
        target (np.ndarray): Chunk-wise target values
        contour_id (np.ndarray): Chunk-wise contour ids
    """
    assert len(targets) == len(contours)
    X = []
    y = []
    contour_id = []
    for c in range(len(contours)):
        curr_feat_mat = buffer_signal(contours[c], block_size, hop_size).T
        curr_target = np.ones(curr_feat_mat.shape[0], dtype=int)*targets[c]
        curr_contour_id = np.ones(curr_feat_mat.shape[0], dtype=int)*c
        X.append(curr_feat_mat)
        y.append(curr_target)
        contour_id.append(curr_contour_id)
    X = np.vstack(X)
    y = np.concatenate(y)
    contour_id = np.concatenate(contour_id)
    return X, y, contour_id
