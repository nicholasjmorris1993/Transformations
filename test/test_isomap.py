import re
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/nick/Transformations/src")
from isomap import features


def corr_plot(df, size=10, method="ward", title="Correlation Plot", save=False):
    # group columns together with hierarchical clustering
    X = df.corr().values
    d = sch.distance.pdist(X)
    L = sch.linkage(d, method=method)
    ind = sch.fcluster(L, 0.5*d.max(), "distance")
    columns = [df.columns.tolist()[i] for i in list((np.argsort(ind)))]
    df = df.reindex(columns, axis=1)
    
    # compute the correlation matrix for the received dataframe
    corr = df.corr()
    
    # plot the correlation matrix
    fig, ax = plt.subplots(figsize=(size, size))
    cax = ax.matshow(corr, cmap="RdYlGn")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90);
    plt.yticks(range(len(corr.columns)), corr.columns);
    
    # add the colorbar legend
    fig.colorbar(cax, ticks=[-1, 0, 1], aspect=40, shrink=.8)

    fig.suptitle(title, y=1.08)
    if save:
        title = re.sub("[^A-Za-z0-9]+", "", title)
        plt.savefig(title + ".png")
    else:
        plt.show()


data = pd.read_csv("/home/nick/Transformations/test/LungCap.csv")
data = data.sample(frac=1, random_state=0).reset_index(drop=True)  # shuffle the data
data = data.drop(columns=["Gender male", "Gender female", "Smoke yes", "Smoke no"])  # remove binary columns

result = features(df=data)

corr_plot(
    df=result.features, 
    size=10, 
    method="ward", 
    title="Lung Capacity Correlation Plot For Isomap Components", 
    save=True,
)
