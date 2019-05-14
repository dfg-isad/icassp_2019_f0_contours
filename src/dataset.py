import numpy as np
import pickle
import pandas as pd
import os


class Dataset:
    """ Class to wrap set of f0 contours with corresponding metadata table """
    def __init__(self, fn_contours, fn_metadata):
        """ Initialize new dataset
        Args:
            fn_contours (string): Pickle file name with raw contours (see Contour class)
            fn_metadata (string): CSV file name with contour metadata stored as pandas DataFrame
        """
        self.contours, self.metadata = Dataset.load_contour_dataset(fn_contours, fn_metadata)
        self.class_label_to_id = None
        self.class_id_to_label = None
        self.file_label_to_id = None
        self.file_id_to_label = None
        self.id_mapping = None

        self.assert_string_class_labels()
        self.generate_class_ids()
        self.generate_file_ids()

    def __len__(self):
        return len(self.contours)

    def assert_string_class_labels(self):
        self.metadata['class_label'] = [str(_) for _ in self.metadata['class_label']]

    def generate_class_ids(self):
        """ Derive class IDs from contour-wise class label annotations """
        self.metadata['class_id'], \
        self.class_id_to_label, \
        self.class_label_to_id = Dataset.get_id_mapping_from_labels(self.metadata['class_label'])

    def generate_file_ids(self):
        """ Derive file IDs from contour-wise WAV file name annotations """
        self.metadata['file_id'], \
        self.file_id_to_label, \
        self.file_label_to_id = Dataset.get_id_mapping_from_labels(self.metadata['fn_wav'])

    def select(self, idx):
        self.metadata = self.metadata.loc[idx, :]
        self.contours = [self.contours[_] for _ in range(len(self.contours)) if idx[_]]
        assert self.metadata.shape[0] == len(self.contours)

    def keep_contours_in_length_range(self, min_contour_len, max_contour_len):
        """ Remove all contours below a minimum length
        Args:
            min_contour_len (int): Minimum contour length
        """
        contour_len = np.array([len(contour) for contour in self.contours])
        idx = np.logical_and(contour_len >= min_contour_len,
                             contour_len <= max_contour_len)
        self.select(idx)
        print("{}/{} contours selected with length within [{}, {}]".format(np.sum(idx),
                                                                           len(idx),
                                                                           min_contour_len,
                                                                           max_contour_len))

    def subsample(self, factor):
        """ Subsampling of contours
        Args:
            factor (int): Subsampling factor
        """
        if not isinstance(factor, int) or factor <= 0:
            raise ValueError('Subsampling factor must be positive integer!')
        for i in range(len(self.contours)):
            self.contours[i].f0_hz = self.contours[i].f0_hz[::factor]
            self.contours[i].t_sec = self.contours[i].t_sec[::factor]
            self.contours[i].f0_cent_rel = self.contours[i].f0_cent_rel[::factor]

    def remove_class(self, class_label):
        idx = np.logical_not(self.metadata['class_label'].as_matrix() == class_label)
        self.select(idx)
        print("{}/{} contours selected which are not class {}".format(np.sum(idx), len(idx), class_label))

    @staticmethod
    def get_id_mapping_from_labels(label_list):
        """ General function to derive unique IDs from given list of annotations
        Args:
            label_list (list of strings): Annotated labels
        Returns
            id_list (list of int): Corresponding IDs
            id_to_label (dict): Mapping from IDs to original labels (IDs are keys, labels are values)
            label_to_id (dict): Mapping from original labels to IDs (Labels are keys, IDs are values)
        """
        sorted_unique_class_labels = sorted(np.unique(label_list))
        num_unique_class_labels = len(sorted_unique_class_labels)
        id_to_label = {_: sorted_unique_class_labels[_] for _ in range(num_unique_class_labels)}
        label_to_id = {id_to_label[_]: _ for _ in range(num_unique_class_labels)}
        id_list = [label_to_id[_.strip()] for _ in label_list]
        return id_list, id_to_label, label_to_id

    @staticmethod
    def load_contour_dataset(fn_contours, fn_metadata):
        """ Load dataset
        Args:
            fn_contours (string): Pickle file name with raw contours (see Contour class)
            fn_metadata (string): CSV file name with contour metadata stored as pandas DataFrame
        """
        with open(fn_contours, 'rb') as f:
            contours = pickle.load(f)
        metadata = pd.read_csv(fn_metadata, sep=',')
        # sanity check
        assert len(contours) == metadata.shape[0]
        return contours, metadata


def import_datasets(dir_data, dataset_labels):
    """ Import existing contour datasets
    Args:
        dir_data (string): Directory where datasets are stored
        dataset_labels (list of strings): Dataset labels
    Returns:
        datasets (dict of Dataset): Datasets, keys are dataset labels
    """
    datasets = {}
    for dataset_label in dataset_labels:
        print('Load from dataset "{}"'.format(dataset_label), end='')
        datasets[dataset_label] = Dataset(fn_contours=os.path.join(dir_data, "db_{}_contours.pickle".format(dataset_label)),
                                          fn_metadata=os.path.join(dir_data, "db_{}_metadata.csv".format(dataset_label)))
        print(' {} contours'.format(len(datasets[dataset_label])))
    return datasets
