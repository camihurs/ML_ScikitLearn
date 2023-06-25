#En este archivo vamos a tener todo el flujo de ML definido
from utils import Utils

if __name__ == "__main__":
    utils = Utils()

    data = utils.load_from_csv('./data/felicidad.csv')

    print(data)