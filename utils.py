#Acá vamos a colocar los métodos que vamos a reutilizar constantemente, como el escalamiento por ejemplo.
import pandas as pd

class Utils:
    def load_from_csv(self, path):
        return pd.read_csv(path)

    def load_from_msql(self):
        pass

    def features_target(self, dataset, drop_cols, y):
        X = dataset.drop(drop_cols, axis=1)
        y = dataset[y]
        return X,y

    def model_export(self, clf, score):
        pass
