import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.kernel_approximation import RBFSampler


def features(df):
    model = Features()
    model.create_features(df)

    return model


class Features:
    def create_features(self, df):
        # standardize the columns to take on values between 0 and 1
        columns = df.columns
        scaler = MinMaxScaler()
        df = scaler.fit_transform(df)
        df = pd.DataFrame(df, columns=columns)

        # train a RBF model
        n_comp = df.shape[1]  # number of random fourier features
        component = RBFSampler(n_components=n_comp, random_state=42)
        component.fit(df)

        # compute features for all the data
        self.features = pd.DataFrame(
            component.transform(df), 
            columns=[f"C{i + 1}" for i in range(n_comp)],
        )
