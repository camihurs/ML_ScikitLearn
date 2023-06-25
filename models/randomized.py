#Vamos a automatizar un poco la selección y optimización de los modelos de ML
import pandas as pd

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

if __name__=="__main__":
    dataset = pd.read_csv("./data/felicidad.csv")
    print(dataset.head())

    X = dataset.drop(['country', 'score', 'rank'], axis = 1)
    y = dataset['score']

    reg = RandomForestRegressor()

    #Vamos a definir la malla de parámetros que va a utilizar nuestro optimizador

    parametros={#Colocamos los parámetros que creemos que vale la pena optimizar
        'n_estimators' : range(4,16), #n_estimators en este caso indica cuántos árboles van a componer nuestro bosque aleatorio
        'criterion' : ['squared_error','absolute_error'], #medida de calidad de los splits que hace el árbol
        'max_depth' : range(2,11) #profundidad del árbol
    }

    rand_est = RandomizedSearchCV(reg,parametros,n_iter=10, cv = 3, scoring='neg_mean_absolute_error').fit(X,y)
    #n_iter va a hacer máximo 10 iteraciones combinando los parámetros

    print(rand_est.best_estimator_)#El resultado es el regresor con sus parámetros correspondientes
    print(rand_est.best_params_)#Arroja lo mismo

    print('Predict: ', rand_est.predict(X.loc[[0]]))#La predicción se hará con el best_estimator
    #Va a predecir el score para el primer país.
    print('Real: ', y[0])