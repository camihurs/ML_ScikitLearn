import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import cross_val_score, KFold

from sklearn.metrics import mean_squared_error


if __name__=="__main__":
    dataset = pd.read_csv("./data/felicidad.csv")

    X = dataset.drop(['country', 'score', 'rank'], axis = 1)

    y = dataset['score']

    model = DecisionTreeRegressor()

    print("Método 1")
    score = cross_val_score(model,X,y,scoring='neg_mean_squared_error')
    print(score)#Acá obtenemos el MSE de 5 pliegues de los datos. 5 es el valor por defecto, lo podemos modificar con el
    #parámetro cv dentro de cross_val_score

    print(np.abs(np.mean(score)))


    print("Método 2")
    kf = KFold(n_splits=3,shuffle=True, random_state=42)#Shuffle es barajar o no los datos, n_splits es el número de pliegues
    mse_values = []

    for train, test in kf.split(dataset):
        print(train)
        print(test)

        X_train = X.iloc[train]
        y_train = y.iloc[train]
        X_test = X.iloc[test]
        y_test = y.iloc[test]


        model = DecisionTreeRegressor().fit(X_train, y_train)
        predict = model.predict(X_test)
        mse_values.append(mean_squared_error(y_test, predict))

    print("Los tres MSE fueron: ", mse_values)
    print("El MSE promedio fue: ", np.mean(mse_values))