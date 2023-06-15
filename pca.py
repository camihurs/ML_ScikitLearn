import pandas as pd
import sklearn
import matplotlib.pyplot as plt

#Vamos a hacer una comparación entre ambos algoritmos
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA

#La clasificación se hará con una regresión logística
from sklearn.linear_model import LogisticRegression

#Utilidades para preparar los datos antes del entrenamiento
from sklearn.preprocessing import StandardScaler #Para normalizar los datos y que estén en la misma escala de 0 a 1
from sklearn.model_selection import train_test_split


#Si tenemos un archivo con muchos scripts, es normal tener uno principal que sea el que corra todo el flujo de ejecución
#Este archivo lo identificamos con la siguiente línea
if __name__ == "__main__":
    dt_heart=pd.read_csv('./data/heart.csv')

    print(dt_heart.head())

    #Seleccionamos las variables predictoras y la variable de salida
    dt_features = dt_heart.drop(["target"],axis=1)
    dt_target = dt_heart["target"]

    #Tenemos dos variables que son categóricas nominales: cp y thal. cp tiene valores de 0 a 3, y "thal" tiene valores
    #de 1 a 3, pero esos números no significan nada como tal, son diferenciadores, así que tenemos que construir variables dummies
    for i in ['cp', 'thal']:
        dt_features[i] = dt_features[i].astype('category')

    dt_features = pd.get_dummies(dt_features,columns=['cp','thal'])

    print(dt_features.head())

    #Para aplicar el PCA necesitamos siempre normalizar los datos
    #Vamos a usar el StandarScaler
    dt_features = StandardScaler().fit_transform(dt_features)

    #Dividimos el conjunto de datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(dt_features,dt_target,test_size=0.3, random_state=42)

    #Verificamos la división de datos de train y test
    print(X_train.shape)
    print(y_train.shape)

    #Invocamos y configuramos el algoritmo PCA para el conjunto de datos de entrenamiento
    #El 3 de n_componentes fue una suposición nada más.
    pca = PCA(n_components=3)#Pasamos el número de componentes que esperamos, es un parámetro opcional. Por defecto es min(#muestras, #features)
    pca.fit(X_train)

    #Ahora hacemos los mismo con ipca para poder comparar la reducción de dimensionalidad de ambos algoritmos
    ipca = IncrementalPCA(n_components=3, batch_size=10)#el batch es el tamaño del bloque de las observaciones; no las envía todas al mismo tiempo
    ipca.fit(X_train)

    #Ahora vamos a medir la varianza de los componentes extraídos; lo vemos gráficamente
    plt.plot(range(len(pca.explained_variance_)),pca.explained_variance_ratio_, label="PCA")

    #En la gráfica que obtenemos, vemos tres componentes (0, 1 y 2) en el eje horizontal, y en el eje vertical
    #vemos el porcentaje de varianza explicada. La primera componente explica más del 20% de la varianza, la segunda
    #explica como 12% y la tercera menos del 10%

    plt.plot(range(len(ipca.explained_variance_)),ipca.explained_variance_ratio_, label = "iPCA")
    plt.title("PCA vs iPCA")

    plt.legend()
    plt.show()

    #Ahora sí vamos a hacer la clasificación
    logistic = LogisticRegression(solver="lbfgs")
    #solver especifica el algoritmo a utilizar en el problema de optimización. "lbfgs" es un algoritmo de optimización que se utiliza para encontrar 
    #los parámetros óptimos para el modelo de regresión logística. Cada solucionador tiene sus propias ventajas y desventajas y puede ser más adecuado 
    # para ciertos tipos de datos o problemas. El solucionador "lbfgs" es un algoritmo de optimización que aproxima el método de Newton utilizando una
    # cantidad limitada de memoria. Es un buen solucionador predeterminado para problemas de tamaño pequeño a mediano.

    #Aplicamos PCA sobre el conjunto de datos de prueba y entrenamiento (antes sólo habíamos ajustado el algoritmo de PCA, no lo habíamos 
    # aplicado al conjunto de entrenamiento y guardado el resultado)-------------------
    dt_train = pca.transform(X_train)
    dt_test = pca.transform(X_test)

    #Ahora entrenamos el algoritmo de regresión logística
    logistic.fit(dt_train, y_train)

    #Ahora medimos el ajuste del modelo usando el accuracy
    print("Score con PCA = ", logistic.score(dt_test, y_test))


    #Ahora solo hacemos lo mismo con iPCA--------------------
    dt_train = ipca.transform(X_train)
    dt_test = ipca.transform(X_test)

    #Ahora entrenamos el algoritmo de regresión logística
    logistic.fit(dt_train, y_train)

    #Ahora medimos el ajuste del modelo usando el accuracy
    print("Score con PCA = ", logistic.score(dt_test, y_test))