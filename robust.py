import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import RANSACRegressor, HuberRegressor

from sklearn.svm import SVR

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

#Lo que estamos haciendo en este script es predecir score de manera lineal con algoritmos que manipulan
#internamente los valores atípitoc, como SVR, RANSAC y HUBER

if __name__ == "__main__":
    dataset = pd.read_csv("./data/felicidad_corrupt.csv")
    print(dataset.head())

    X = dataset.drop(['country', 'score', 'rank', 'high', 'low'], axis = 1)
    y = dataset['score']#Si se colocan dos corchetes, nos arroja una advertencia de dimensiones

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42)

    estimadores = {
        'SVR':SVR(gamma='auto', C=1.0, epsilon=0.1),
        'RANSAC': RANSACRegressor(), #acá se podrían colocar como parámetros otros modelos, como SVR, porque RANSAC es un meta-estimador
        'HUBER': HuberRegressor(epsilon=1.35)#Determina cuándo un dato se considera como outlier
    }

    for name,estimador in estimadores.items():
        estimador.fit(X_train,y_train)
        predictions = estimador.predict(X_test)
        print("=" * 32)
        print(name)
        plt.ylabel('Predicted Score')
        plt.xlabel('Real Score')
        plt.title('Predicted VS Real ' + str(name))
        plt.scatter(y_test, predictions)
        plt.plot(predictions, predictions,'r--')
        plt.show()

        print("="*64)
        print(name)
        print("MSE: ", mean_squared_error(y_test,predictions))