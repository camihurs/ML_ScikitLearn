#Con este método dejamos que el mismo algoritmo decida la cantidad de clusters
#Funciona mejor para pocos datos

import pandas as pd
from sklearn.cluster import MeanShift

if __name__ == "__main__":
    dataset = pd.read_csv("./data/candy.csv")
    print(dataset.head())

    X = dataset.drop("competitorname", axis = 1)

    meanshift = MeanShift().fit(X)
    print("La cantidad de clusters creados fue: ", max(meanshift.labels_) + 1)
    print("="*64)
    print("Los centros de los tres clusters son: ", meanshift.cluster_centers_)

    dataset['meanshift']=meanshift.labels_
    print("="*64)
    print(dataset.head())