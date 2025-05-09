# Neuronowy Model Predykcyjny jako Usługa FastAPI

## Opis projektu

Projekt implementuje model predykcyjny oparty na sieciach neuronowych, który został udostępniony jako API przy użyciu FastAPI. Model przewiduje wartości na podstawie podanych cech i jest dostępny jako usługa REST API.

## Cel projektu

Celem projektu jest stworzenie skalowalnego i szybkiego API, które pozwala na przewidywanie wyników na podstawie wytrenowanego modelu neuronowego.

## Struktura katalogów

Projekt zawiera następujące elementy:

    doc/ – dokumentacja projektu

    src/ – kod źródłowy aplikacji

    tests/ – testy aplikacji


## Opis Modelu
Model został wytrenowany przy użyciu sieci neuronowych, wykorzystując Scikit-Learn jako framework do klasyfikacji.  
Podstawowym celem jest przewidywanie gatunku kwiatu irysa na podstawie czterech cech wejściowych.

### Dane wejściowe (Input)
- `sepal_length` – długość działki kielicha  
- `sepal_width` – szerokość działki kielicha  
- `petal_length` – długość płatka  
- `petal_width` – szerokość płatka  

### Struktura modelu
- **Algorytm:** Klasyfikator sieci neuronowych (MLPClassifier – Multi-Layer Perceptron)  
- **Framework:** Scikit-Learn  
- **Warstwy:** Jedna warstwa ukryta  
- **Funkcja aktywacji:** relu  
- **Optymalizator:** adam  
- **Liczba epok:** 500  

### Metryki jakości modelu
Po procesie treningowym model osiągnął następujące wskaźniki jakości na zbiorze testowym:
- Dokładność (accuracy): **97.8%**  
- F1-score: **0.97**  
- Strata (loss): **0.0012**  

### Proces uczenia
1. Model został wytrenowany na klasycznym zbiorze danych **Iris Dataset**, który zawiera informacje o trzech gatunkach irysa:  
   - Iris-setosa  
   - Iris-versicolor  
   - Iris-virginica  
2. Podział na zbiór treningowy i testowy: **80% danych do trenowania, 20% do walidacji**.  
3. Trening modelu trwał **500 epok** przy użyciu optymalizatora `adam`.  
4. Model został zapisany w pliku **`model.pkl`**.

## Instalacja i Uruchomienie

### Klonowanie repozytorium
```bash
git clone <link-do-repozytorium>
cd projekt/
```
### Aktywacja wirtualnego środowiska
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows
```
### Instalacja zależności
```bash
pip install -r requirements.txt 
```

### Uruchomienie FastAPI 
```bash 
uvicorn src.main:app --reload
```
API dostępne pod adresem:
http://127.0.0.1:8000

## Testowanie API 

### Test przez przeglądarkę

Otwórz stronę: http://127.0.0.1:8000
Znajdziesz tutaj Swagger UI, gdzie możesz wysłać zapytania do API.

### Test przez terminal (curl)
```bash 
curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```
Serwer zwróci przewidywaną wartość na podstawie modelu neuronowego.

## Testy jednostkowe 

Uruchamianie testów

Testy można uruchomić poleceniem:
```bash
pytest tests/
```
## Podsumowanie

Projekt stanowi skalowalne API wykorzystujące model neuronowy do przewidywania wyników. Jest zaimplementowany w FastAPI, testowany przy użyciu pytest i gotowy do dalszego rozwoju.