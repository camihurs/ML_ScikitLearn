import pandas as pd
import sklearn

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    dataset = pd.read_csv("./data/felicidad.csv")
    print(dataset.describe())

    X = dataset[['gdp','family','lifexp','freedom','corruption','generosity','dystopia']]
    y = dataset['score']

    print(X.shape)
    print(y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    LinearModel = LinearRegression().fit(X_train, y_train)
    y_predict_linear = LinearModel.predict(X_test)

    LassoModel = Lasso(alpha=0.02).fit(X_train,y_train)#alpha es la penalización
    y_predict_lasso = LassoModel.predict(X_test)

    RidgeModel = Ridge(alpha=1).fit(X_train,y_train)
    y_predict_Ridge = RidgeModel.predict(X_test)

    #Para este ejemplo elegimos calcular nuestra pérdida a través de error cuadrático medio
    #Hay diferentes métricas tanto para clasificación como para regresión que nos pueden dar información
    #diferente sobre los cálculos que estamos realizando
    #Scikit Learn tiene una sección de métricas

    linear_loss = mean_squared_error(y_test,y_predict_linear)
    print("Linear Loss: ", linear_loss)

    Lasso_loss = mean_squared_error(y_test,y_predict_lasso)
    print("Linear Loss: ", Lasso_loss)

    Ridge_loss = mean_squared_error(y_test,y_predict_Ridge)
    print("Linear Loss: ", Ridge_loss)

    print("="*32)
    print("Coeficientes Lasso: ")
    print(LassoModel.coef_)

    print("="*32)
    print("Coeficientes Ridge: ")
    print(RidgeModel.coef_)