import joblib
import numpy as np

from flask import Flask
from flask import jsonify

app = Flask(__name__)

#POSTMAN para pruebas
#El método GET es el que utiliza un navegador web cuando intenta acceder a un recurso que está en un servidor ajeno
#vamos a definir una ruta desde la cual nuestro servidor nos pueda contestar

@app.route('/predict', methods = ['GET']) #El servidor va a responder cuando llamemos a la URL/predict

def predict():
    #La siguiente línea corresponde a la primera fila del dataset felicidad, sin country, rank ni score
    X_test = np.array([7.594444821,7.479555538,1.616463184,1.53352356,0.796666503,0.635422587,0.362012237,0.315963835,2.277026653])
    prediction = model.predict(X_test.reshape(1,-1))#1 fila, y que detecte él mismo cuántas columnas son
    return jsonify({'predicción':list(prediction)})

if __name__ == "__main__":
    model = joblib.load('./models/best_model.pkl')
    app.run(port=8080) #Ponemos a correr el servidor, pero necesitamos que nuestro servidor nos devuelva cosas, nos conteste peticiones