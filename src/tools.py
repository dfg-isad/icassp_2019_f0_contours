import numpy as np

from dataset import import_datasets


def assert_increasing_class_ids(class_ids):
    """" If unique class ids are not a strictly increasing sequence, perform mapping.
    Args:
        class_ids (np.ndarray): Class IDs
    Returns:
        class_ids (np.ndarray): Class IDs (with strictly increasing unique values)
        mapping (dict): Mapping dictionary, keys are new class ids, values are old class IDs
    Example:
        Given a vector class_id = [0,0,1,1,1,3,3,3,4,4]
        -> unique values = (0, 1, 3, 4)
        -> mapped to (0, 1, 2, 3),
        -> returned class_id -> [0,0,1,1,1,2,2,2,3,3]
        -> mapping = {0:0, 1:1, 2:3, 3:4}
    """
    unique_vals = np.sort(np.unique(class_ids))
    num_uv = len(unique_vals)
    mapping = {i: unique_vals[i] for i in range(num_uv)}
    for i in range(num_uv):
        class_ids[class_ids == unique_vals[i]] = i
    return class_ids, mapping


def import_and_preprocess_datasets(dir_data, datasets_to_import, min_contour_len, max_contour_len):
    """ Import and pre-process datasets
    Args:
        dir_data (string): Directory where datasets are stored
        datasets_to_import (list of strings): Labels of datasets to be imported
        min_contour_len (int): Minimum contour len
        max_contour_len (int): Maximum contour len
    Returns:
        datataset (dict): Dictionary with all datasets, dataset labels are keys, Dataset() instances are values
    """

    datasets = import_datasets(dir_data, datasets_to_import)

    # special treatment for Weimar Jazz Database -> remove non-annotated contours from dataset
    if 'wjd' in datasets_to_import:
        datasets['wjd'].remove_class('nan')

    # special treatment for guitar dataset -> remove HA (harmonics) and DN (dead-note) classes
    if 'idmt_smt_guitar' in datasets_to_import:
        datasets['idmt_smt_guitar'].remove_class('HA')
        datasets['idmt_smt_guitar'].remove_class('DN')

    # special treatment for GENRE dataset -> subsample by factor 2 to ensure unified hopsize value of 5.8 ms
    if 'melodia_music_genre' in datasets_to_import:
        datasets['melodia_music_genre'].subsample(2)

    # remove contours that are too short
    for label in datasets.keys():
        datasets[label].keep_contours_in_length_range(min_contour_len, max_contour_len)

    return datasets
