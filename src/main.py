from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

# Inicjalizacja FastAPI
app = FastAPI()


class IrisData(BaseModel):
    """
    Model danych wejściowych dla API przewidywania gatunków irysa.
    
    Atrybuty:
        sepal_length (float): Długość działki kielicha (cm).
        sepal_width (float): Szerokość działki kielicha (cm).
        petal_length (float): Długość płatka (cm).
        petal_width (float): Szerokość płatka (cm).
    """
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']


@app.post("/predict")
def predict(data: IrisData):
    """
    Endpoint API do przewidywania gatunku irysa na podstawie cech rośliny.

    Parametry:
        data (IrisData): Obiekt zawierający długość i szerokość działek kielicha oraz płatków.

    Zwraca:
        dict: Słownik JSON z przewidywaną nazwą gatunku.
    
    Przykładowe użycie:
        POST /predict
        {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        Odpowiedź:
        {
            "prediction": "Iris-setosa"
        }
    """
    try:
        model = joblib.load('src/model.pkl')
        input_data = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
        prediction = model.predict(input_data)
        class_name = class_names[int(prediction[0])]
        return {"prediction": class_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
