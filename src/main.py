import numpy as np
import pickle
import pandas as pd
import os
import datetime
import time

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.metrics import f1_score
from keras.optimizers import Adam, Adadelta

from feature_extractors import ContourFeatureExtractorPymus, \
                               ContourFeatureExtractorBitteli, \
                               ContourFeatureExtractorNeuralNetwork

from tools import assert_increasing_class_ids, \
                  import_and_preprocess_datasets


__author__ = 'Jakob Abesser, abr@idmt.fhg.de'


if __name__ == '__main__':

    t = time.time()

    dir_data = os.path.join('..', "data")
    dir_results = os.path.join('..', "results")
    dir_models = os.path.join('..', "models")

    min_contour_len = 34  # 197,2 ms
    max_contour_len = 344  # 1995,2 ms
    feature_dim_neural_networks = 17
    num_epochs = 1000
    num_folds = 3
    num_random_forest_estimators = 50

    cnn_optimizer = Adam(lr=1E-4)

    feature_extractors = {'pymus': ContourFeatureExtractorPymus(),
                          'bitteli': ContourFeatureExtractorBitteli(),
                          'cnn-1': ContourFeatureExtractorNeuralNetwork(type='CNN',
                                                                        num_feature_blocks=1,
                                                                        feature_dim=feature_dim_neural_networks,
                                                                        num_epochs=num_epochs,
                                                                        optimizer=cnn_optimizer,
                                                                        dropout_ratio=.25),
                          'cnn-2': ContourFeatureExtractorNeuralNetwork(type='CNN',
                                                                        num_feature_blocks=2,
                                                                        feature_dim=feature_dim_neural_networks,
                                                                        num_epochs=num_epochs,
                                                                        optimizer=cnn_optimizer,
                                                                        dropout_ratio=.25)
                          }

    cv_splitter = StratifiedShuffleSplit(n_splits=num_folds, random_state=42, test_size=.2)

    datasets_do_file_aggregation = {'idmt_smt_guitar': False,
                                    'idmt_smt_monotimbral': True,
                                    'melodia_music_genre': True,
                                    'wjd': False}

    datasets_to_import = list(datasets_do_file_aggregation.keys())

    # import and pre-process datasets
    datasets = import_and_preprocess_datasets(dir_data,
                                              datasets_to_import,
                                              min_contour_len,
                                              max_contour_len)

    # Pre-compute features for non-trainable feature extractors
    fn_feat = os.path.join(dir_data, 'dataset_features_non_trainable_feature_extractors.pickle')
    with open(fn_feat, 'rb') as f:
        dataset_features = pickle.load(f)

    # main experiment
    exp_count = 0

    classifier = RandomForestClassifier(n_estimators=num_random_forest_estimators)
    scaler = StandardScaler()

    num_datasets = len(datasets.keys())
    num_extractors = len(feature_extractors.keys())

    ds_labels = sorted(datasets.keys())
    fe_labels = sorted(feature_extractors.keys())

    y_pred_all = {}
    feature_importance = {}
    y_true_all = {}
    file_id_all = {}

    # iterate over datasets
    for d, ds_label in enumerate(ds_labels):

        dataset = datasets[ds_label]

        num_contours = len(dataset)
        all_contour_len = [len(_) for _ in dataset.contours]

        min_contour_len = min(all_contour_len)
        max_contour_len = max(all_contour_len)

        dataset.metadata['class_id'], dataset.id_mapping = assert_increasing_class_ids(dataset.metadata['class_id'].values)
        class_id = dataset.metadata['class_id'].values
        file_id = dataset.metadata['file_id'].values
        num_unique_file_id = len(np.unique(file_id))
        file_id_split_point = int(0.8*num_unique_file_id)

        # iterate over feature extractors
        for f, fe_label in enumerate(fe_labels):

            unique_class_ids = np.unique(class_id)

            for i in range(num_folds):

                # randomly split data into training and test set based on file IDs
                dummy = np.arange(num_unique_file_id)
                np.random.shuffle(dummy)
                file_id_train = dummy[:file_id_split_point]
                file_id_test = dummy[file_id_split_point:]
                train_index = np.where(np.in1d(file_id, file_id_train))[0]
                test_index = np.where(np.in1d(file_id, file_id_test))[0]

                print('-' * 60)
                print('Experiment | feature set {} | dataset {} | fold {}/{}'.format(fe_label,
                                                                                     ds_label,
                                                                                     i + 1,
                                                                                     num_folds))
                print('-' * 60)

                # split targets
                y = np.copy(class_id)
                num_classes = np.max(np.unique(y))+1
                if num_classes != len(np.unique(y)):
                    print('Something is fishy: unique values = {}'.format(np.unique(y)))
                target_train = y[train_index]
                target_test = y[test_index]

                # (1) trainable feature extractors:
                #   -> extract features from training set
                #   -> extract features from test set
                if feature_extractors[fe_label].is_trainable:
                    train_contours = [dataset.contours[_] for _ in train_index]
                    test_contours = [dataset.contours[_] for _ in test_index]

                    fn_model = os.path.join(dir_models, 'model_{}_{}'.format(fe_label, ds_label))
                    feature_extractors[fe_label].load(fn_model)
                    feature_extractors[fe_label].min_contour_len = min_contour_len
                    feature_extractors[fe_label].max_contour_len = max_contour_len
                    feat_train, feature_labels = feature_extractors[fe_label].extract_features_from_contours(train_contours)
                    feat_test, feature_labels = feature_extractors[fe_label].extract_features_from_contours(test_contours)

                # (2) non-trainable feature extractors:
                #   use pre-extracted features
                else:
                    feature_labels = dataset_features[ds_label][fe_label]['feature_labels']
                    X = np.copy(dataset_features[ds_label][fe_label]['features'])
                    X = np.nan_to_num(X)
                    feat_train = X[train_index, :]
                    feat_test = X[test_index, :]

                # feature normalization & model training
                feat_train = scaler.fit_transform(feat_train)
                classifier.fit(feat_train, target_train)

                feature_importance[(d, f, i)] = classifier.feature_importances_
                # feature scaling & prediction
                feat_test = scaler.transform(feat_test)
                y_pred_all[(d, f, i)] = classifier.predict_proba(feat_test)
                y_true_all[(d, f, i)] = y[test_index]
                all_file_id = np.array([dataset.file_label_to_id[_] for _ in dataset.metadata['fn_wav'].tolist()])
                file_id_all[(d, f, i)] = all_file_id[test_index]

    with open(os.path.join(dir_results, 'feature_importance.pckl'), 'wb+') as f:
        pickle.dump(feature_importance, f)

    f1_scores = np.zeros((num_datasets, num_extractors, num_folds))
    f1_scores_file = np.zeros((num_datasets, num_extractors, num_folds))

    # iterate over datasets
    for d, ds_label in enumerate(ds_labels):
        # iterate over feature extractors
        for f, fe_label in enumerate(fe_labels):
            # iterate over CV folds
            for i in range(num_folds):

                # item-wise f1 score
                f1_scores[d, f, i] = f1_score(y_true_all[(d, f, i)],
                                              np.argmax(y_pred_all[(d, f, i)], axis=1),
                                              average="micro")

                # file aggregation
                if datasets_do_file_aggregation[ds_label]:
                    unique_file_ids = np.unique(file_id_all[(d, f, i)])
                    num_ufid = len(unique_file_ids)
                    y_true = np.zeros(num_ufid, dtype=int)
                    y_pred = np.zeros(num_ufid, dtype=int)
                    for u, uid in enumerate(unique_file_ids):
                        mask = file_id_all[(d, f, i)] == uid
                        y_true[u] = y_true_all[(d, f, i)][mask][0]
                        pred_prob = np.mean(y_pred_all[(d, f, i)][mask, :], axis=0, keepdims=True)
                        y_pred[u] = np.argmax(pred_prob, axis=1)[0]
                    f1_scores_file[d, f, i] = f1_score(y_true,
                                                       y_pred,
                                                       average="micro")

    f1_scores_mean = np.mean(f1_scores, axis=2)
    f1_scores_std = np.std(f1_scores, axis=2)
    f1_scores_file_mean = np.mean(f1_scores_file, axis=2)
    f1_scores_file_std = np.std(f1_scores_file, axis=2)

    rows, cols = f1_scores_std.shape

    exp_res_txt = [['{:1.2} ({:1.2})'.format(f1_scores_mean[row, col],
                                             f1_scores_std[row, col]) for col in range(cols)] for row in range(rows)]

    fn_csv = os.path.join(dir_results, 'f1_scores_test.csv')

    df_contour = pd.DataFrame(data=exp_res_txt, columns=fe_labels)
    df_contour.rename(index={_: ds_labels[_] for _ in range(len(ds_labels))}, inplace=True)
    df_contour.to_csv(fn_csv, sep=';')

    exp_res_txt = [['{:1.2} ({:1.2})'.format(f1_scores_file_mean[row, col],
                                             f1_scores_file_std[row, col]) for col in range(cols)] for row in range(rows)]

    fn_csv = os.path.join(dir_results, 'f1_scores_file_test.csv')

    df_file = pd.DataFrame(data=exp_res_txt, columns=fe_labels)
    df_file.rename(index={_: ds_labels[_] for _ in range(len(ds_labels))}, inplace=True)
    df_file.to_csv(fn_csv, sep=';')

    print('='*60)
    print('Overall processing took {} h'.format((time.time()-t)/3600))
    print('='*60)
