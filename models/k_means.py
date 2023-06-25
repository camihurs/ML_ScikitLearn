import pandas as pd

from sklearn.cluster import MiniBatchKMeans

if __name__ == "__main__":
    dataset = pd.read_csv("./data/candy.csv")
    print(dataset.head())

    #Como es aprendizaje no supervisado, no necesitamos dataset de entrenamiento y pruebas
    #Tampoco necesitamos separar entre features y target

    X = dataset.drop(['competitorname'], axis=1)#Quitamos esta columna porque no aporta nada al modelo
    kmeans = MiniBatchKMeans(n_init = "auto", n_clusters=4, batch_size=8).fit(X)

    print("Total de centros: ", len(kmeans.cluster_centers_))
    print("="*64)
    print(kmeans.predict(X))#Me devuelve la categoría asignada a cada instancia

    dataset['group'] = kmeans.predict(X)

    print(dataset)