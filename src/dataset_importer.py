import os
import pickle
import numpy as np
import pandas as pd


class DatasetImporterPantelliBittnerICASSP2018:
    """ Class loads full set of features & metadata as in
        https://github.com/rabitt/icassp-2017-world-music/blob/master/experiment-scripts/analyze_contours.py
    """

    def load(self):
        dir_data = os.path.join(os.path.dirname(__file__), '../external/icassp-2017-world-music-master/data/')
        fn_contour_data = os.path.join(dir_data, 'contour_data.pickle')
        fn_metadata = os.path.join(dir_data, 'metadata.csv')

        with open(fn_contour_data, 'rb') as f:
            contour_features, contour_files = pickle.load(f, encoding='latin1')

        contour_files = contour_files.astype(str)
        # contour_features (67010 x 30)
        df = pd.read_csv(fn_metadata)

        language = df["Language"].values

        uniq_files = np.unique(contour_files)

        inds = []
        for uniq_file in uniq_files:
            inds.append(np.where(df['Csv'] == uniq_file)[0][0])
        inds = np.array(inds)

        df = df.iloc[inds, :].reset_index()

