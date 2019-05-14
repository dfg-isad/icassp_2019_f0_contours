import numpy as np
import time
from keras.models import Model, Sequential
from keras.layers import LSTM, Dense, Flatten, Input, AtrousConvolution1D, Conv1D, MaxPool1D, Activation, Dropout
from keras.optimizers import Adadelta, Adam
from keras.layers.normalization import BatchNormalization
from keras.models import load_model, save_model, clone_model
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical

# we don't need the external packages here
class BitteliFeatures:
    pass
class ContourFeatures:
    pass

from contour_chunker import chunk_contours_to_unit_length


class ContourFeatureExtractorAll:
    """ Wrapper to compute all variants of contour features """

    def __init__(self):
        self.extractors = {'Pymus': ContourFeatureExtractorPymus(),
                           'Motif': ContourFeatureExtractorBitteli()}
        # todo add more extractors when implemented

    def extract_features_from_contours(self, contours):
        """ Extract features for given set of contours using all contour feature extractors
        Args:
            contours (list of Contour): Contours
        Returns:
            features (dict of 2d np.ndarray): Dictionary with feature matrices, keys are extractor labels
            feature_labels (dict of list of string): Dictionary with feature dimension labels, keys are extractor labels
        """
        extractor_labels = self.extractors.keys()
        features = {_: None for _ in extractor_labels}
        feature_labels = {_: None for _ in extractor_labels}

        for extractor in extractor_labels:
            features[extractor], feature_labels[extractor] = \
                self.extractors[extractor].extract_features_from_contours(contours=contours)

        return features, feature_labels


class ContourFeatureExtractorBase:
    """ Bass class for contour feature extractors """

    is_trainable = False

    def extract_features_from_contours(self,
                                       contours,
                                       do_split_into_sub_contours=False,
                                       blocksize=34,
                                       hopsize=17):
        """ Extract features from multiple contours
        Args:
            contours (list of Contour): Frequency contours to be processed (num_contours)
            do_split_into_sub_contours (bool): Switch to cut (variable-length) contours to fixed length contours first
            blocksize (int): Blocksize to be used if do_split_into_sub_contours == True
            hopsize (int): Hopsize to be used if do_split_into_sub_contours == True
        Returns:
            features (2d np.ndarray): Feature matrix (num_contours x num_features)
            feature_labels (list of strings): Feature dimension labels
            contour_id (np.ndarray): Contour ID (num_contours) -> groups feature vectors to original contours,
                                     if do_split_into_sub_contours == True, all feature vectors from subcontours of the
                                     first original contour will have contour_id values of 0, and so on
        """
        features = []
        feature_labels = []
        num_orig_contours = len(contours)
        if do_split_into_sub_contours:
            # split contour into subcontours first
            feature_matrix, _, contour_id = chunk_contours_to_unit_length([_.f0_cent_rel for _ in contours],
                                                                           np.ones(num_orig_contours),
                                                                           block_size=blocksize,
                                                                           hop_size=hopsize)
            # todo convert feature matrix in to contours before feature extractor is called
            raise NotImplementedError
        for i in range(len(contours)):
            contour_id = np.arange(len(contours))
            curr_features, curr_feature_labels = self.extract_features_from_contour(contours[i])
            features.append(curr_features)
            if i == 0:
                feature_labels = curr_feature_labels
        features = np.vstack(features)
        return features, feature_labels, contour_id

    def extract_features_from_contour(self, contour):
        """ Extract features from contour
        Args:
            contour (Contour): Frequency contour
        Returns:
            features (np.ndarray): Feature matrix (num_features)
            feature_labels (list of strings): Feature dimension labels
        """
        raise NotImplementedError('Implement this in derived class')

    @staticmethod
    def contours_to_feature_matrix(contours,
                                   targets,
                                   blocksize,
                                   hopsize):
        """ Convert set of frequency contours into feature matrix to be fed into a neural network by chunking them into
            contour parts of equal lengh (basically by windowing them with a fixed-size window and an overlap)
        Args:
            contours (list of Contour): Frequency contours (num_contours)
            targets (np.ndarray): Contour class IDs (num_contours)
            blocksize (int): Blocksize for contour chunking
            hopsize (int): Hopsize for contour chunking
        Returns:
            feature_matrix (2D np.ndarray): Feature matrix (num_sub_contours x blocksize)
            targets (np.ndarray): Sub-contour class IDs (num_sub_contours)
            contour_id (np.ndarray): Contour IDs where sub-contours originate from
        TODO
            add optional log-time resampling here
        """
        # chunk contours into sub-contours of equal length
        feature_matrix, _, contour_id = chunk_contours_to_unit_length([_.f0_cent_rel for _ in contours],
                                                                   targets,
                                                                    block_size=blocksize,
                                                                    hop_size=hopsize)

        targets = targets[contour_id]

        # center contours
        feature_matrix -= np.mean(feature_matrix, axis=1, keepdims=True)
        # reshape to 3D numpy array for conv 1D / RNN layers: (samples, time_steps, features)
        feature_matrix = feature_matrix.reshape((feature_matrix.shape[0], feature_matrix.shape[1], 1))

        return feature_matrix, targets, contour_id


class ContourFeatureExtractorPymus(ContourFeatureExtractorBase):
    """ Wrapper to f0 contour features from pymus python project (Jazzomat Research Project) """

    def __init__(self):
        self.extractor = ContourFeatures()
        self.is_trainable = False

    def extract_features_from_contour(self, contour):
        features, feature_labels = self.extractor.process(contour.t_sec,
                                                          contour.f0_hz,
                                                          contour.f0_cent_rel)
        return features, feature_labels


class ContourFeatureExtractorBitteli(ContourFeatureExtractorBase):

    def __init__(self):
        self.extractor = BitteliFeatures()
        self.is_trainable = False

    def extract_features_from_contour(self, contour):
        # https://github.com/rabitt/motif
        # https://github.com/rabitt/icassp-2017-world-music > calls bitteli feature extractor from motif
        features = self.extractor.get_feature_vector(contour.t_sec,
                                                     contour.f0_hz,
                                                     np.ones_like(contour.f0_hz),
                                                     1000/5.8)
        feature_labels = self.extractor.feature_names
        return features, feature_labels


class ContourFeatureExtractorNeuralNetwork(ContourFeatureExtractorBase):

    def __init__(self,
                 type='RNN',
                 num_feature_blocks=2,
                 feature_dim=25,
                 max_abs_interval_size_semitones=2,
                 step_size_cents=5,
                 batch_size=50,
                 num_epochs=2,
                 optimizer=Adadelta(clipnorm=1.25, lr=.5),
                 use_dictionary_mapping=False,
                 dropout_ratio=0.2,
                 dilation_rate=1):
        super(ContourFeatureExtractorBase, self).__init__()
        self.is_trainable = True

        self.model_create_methods = {'RNN': self.create_rnn_model,
                                     'CNN': self.create_cnn_model}
        self.model_create_method = self.model_create_methods[type]

        assert num_feature_blocks >= 1, "Parameter 'num_feature_blocks' must >= 1!"
        self.num_feature_blocks = num_feature_blocks

        self.feature_dim = feature_dim
        self.max_abs_interval_size_semitones = max_abs_interval_size_semitones
        self.step_size_cents = step_size_cents
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.min_contour_len = None
        self.max_contour_len = None
        self.model = None
        self.optimizer = optimizer
        self.use_dictionary_mapping = use_dictionary_mapping
        self.dropout_ratio = dropout_ratio
        self.dilation_rate = dilation_rate

    def create_rnn_model(self,
                         contour_len,
                         num_classes):
        """ Create RNN model for contour classification
        Args:
            contour_len (int): Input sequence length
            num_classes (int): Number of classes
        """
        self.model = Sequential()
        self.model.add(LSTM(self.feature_dim,
                       return_sequences=True if self.num_feature_blocks > 1 else False,
                       name='lstm_1',
                       input_shape=(contour_len, 1),
                       dropout=self.dropout_ratio))
        self.model.add(BatchNormalization())
        self.model.add(Activation(activation="relu"))
        self.model.add(Dropout(self.dropout_ratio))
        # additional feature block(s)
        for i in range(1, self.num_feature_blocks):
            self.model.add(LSTM(self.feature_dim,
                                return_sequences=False,
                                name='lstm_{}'.format(i+1),
                                dropout=self.dropout_ratio))
            self.model.add(BatchNormalization())
            self.model.add(Activation(activation="relu"))
            self.model.add(Dropout(self.dropout_ratio))
        self.model.add(Dense(self.feature_dim, name='features'))
        self.model.add(Dense(num_classes, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=self.optimizer,
                           metrics=['accuracy'])

    def create_cnn_model(self,
                         contour_len,
                         num_classes,
                         kernel_size=5,
                         num_filters=20):
        """ Create 1D-CNN model for contour classification
        Args:
            contour_len (int): Input sequence length
            num_classes (int): Number of classes
        """
        self.model = Sequential()
        # self.model.add(Input(shape=))

        self.model.add(Conv1D(input_shape=(contour_len, 1),
                              filters=num_filters,
                              kernel_size=kernel_size,
                              padding="same",
                              name='conv1',
                              dilation_rate=self.dilation_rate))
        self.model.add(BatchNormalization())
        self.model.add(Activation(activation="relu"))
        self.model.add(Dropout(self.dropout_ratio))
        self.model.add(MaxPool1D(pool_size=2))
        # additional feature block(s)
        for i in range(1, self.num_feature_blocks):
            self.model.add(Conv1D(filters=num_filters,
                                  kernel_size=kernel_size,
                                  padding="same",
                                  name="conv{}".format(i+1),
                                  dilation_rate=self.dilation_rate))
            self.model.add(BatchNormalization())
            self.model.add(Activation(activation="relu"))
            self.model.add(Dropout(self.dropout_ratio))
        self.model.add(Flatten())
        self.model.add(Dense(self.feature_dim, name='features'))
        self.model.add(Dense(num_classes, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=self.optimizer,
                           metrics=['accuracy'])

    def train(self,
              contours,
              targets,
              min_contour_len,
              max_contour_len,
              num_classes):
        """ Train model
        Args:
            contours (list of Contour): Variable-length contours
            targets (np.ndarray): Contour-wise target values
            min_contour_len (int): Minimum contour length to consider
            max_contour_len (int): Maximum contour length to consider
            num_classes (int): Number of classes
        """
        self.min_contour_len = min_contour_len
        self.max_contour_len = max_contour_len

        # create model
        self.model_create_method(min_contour_len,
                                 num_classes)

        # convert variable-length contours to unit-length feature matrix for model training
        X, targets, contour_id = ContourFeatureExtractorBase.contours_to_feature_matrix(contours,
                                                                                        targets,
                                                                                        self.min_contour_len,
                                                                                        self.min_contour_len//2)

        # one-hot encoding of target values
        y = to_categorical(targets, num_classes)

        # train model with early stopping
        callbacks = list()
        callbacks.append(EarlyStopping(monitor='loss',
                                       patience=25))
        self.model.fit(X,
                       y,
                       batch_size=self.batch_size,
                       epochs=self.num_epochs,
                       shuffle=True,
                       callbacks=callbacks,
                       validation_split=.2,
                       verbose=2)

        # remove final softmax layer to get feature extractor
        self.model = Model(input=self.model.layers[0].input,
                           output=self.model.get_layer('features').output)

    def extract_features_from_contours(self, contours):
        """ Use trained model to convert list of contours into internal feature representation
        Args:
            contours (list of Contour): Contours
        Returns:
            features (2d np.ndarray): Feature matrix (num_contours x num_features)
            feature_labels (list of strings): Feature dimension labels (num_features)
        """
        num_contours = len(contours)
        t = time.time()

        # convert variable-lenght contours into fixed-size input feature matrix (sub-contours)
        X, targets, contour_id = ContourFeatureExtractorNeuralNetwork.contours_to_feature_matrix(contours,
                                                                                                 np.ones(len(contours)),
                                                                                                 self.min_contour_len,
                                                                                                 self.min_contour_len//2)

        features = self.model.predict(X)

        # average accross contours
        contour_features = []
        for cid in np.sort(np.unique(contour_id)):
            contour_features.append(np.mean(features[contour_id == cid, :], axis=0))
        features = np.vstack(contour_features)

        print('Prediction for {} contours took {:2.2} min'.format(num_contours,
                                                                  (time.time() - t)/60))

        feature_labels = ['cnn_{}'.format(_) for _ in range(len(features))]
        return features, feature_labels

    def save(self, fn_model):
        """ Save model to file
        Args:
            fn_model (string): Filename
        """
        save_model(self.model, fn_model)

    def load(self, fn_model):
        """ Load model from file
        Args:
            fn_model (string): Filename
        """
        self.model = load_model(fn_model)

