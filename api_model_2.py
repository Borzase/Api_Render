from flask import Flask, jsonify, request
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np

app = Flask(__name__)
@app.route("/", methods=["GET"])
def hello():
    return "Bienvenido a mi API de predicciones de adicción a las redes sociales"
@app.route("/api/v1/predict", methods=["GET"])
def predict():
    # Cargar el modelo
    try:
        with open('model_2.pkl', 'rb') as f:
            model_2 = pickle.load(f)
    except Exception as e:
        return jsonify({"error": f"No se pudo cargar el modelo: {str(e)}"}), 500
    # Obtener parámetros
    age = request.args.get('Age', None)
    continent = request.args.get('Continent', None)
    sleep = request.args.get('Sleep_Hours_Per_Night', None)
    # Validar existencia
    if age is None or continent is None or sleep is None:
        return jsonify({"error": "Faltan argumentos. Se requieren Age, Continent y Sleep_Hours_Per_Night."}), 400
    # One-hot encoding de Age
    col_age = ["Age_19", "Age_20", "Age_21", "Age_22", "Age_23", "Age_24"]
    age_dummies = {col: 0 for col in col_age}
    selected_col_age = f"Age_{age}"
    if selected_col_age in age_dummies:
        age_dummies[selected_col_age] = 1
    else:
        return jsonify({"error": f"Valor de edad no permitido: '{age}'"}), 400
    # One-hot encoding de Continent
    col_cont = ["Continent_Asia", "Continent_Europe", "Continent_North America", "Continent_Oceania", "Continent_South America"]
    continent_dummies = {col: 0 for col in col_cont}
    selected_col_cont = f"Continent_{continent}"
    if selected_col_cont in continent_dummies:
        continent_dummies[selected_col_cont] = 1
    else:
        return jsonify({"error": f"Continente no reconocido: '{continent}'"}), 400
    try:
        input_vector = [float(sleep)] + list(continent_dummies.values()) + list(age_dummies.values())
        prediction = model_2.predict([input_vector])
        prediction_value = float(prediction[0])  # Conversión segura a tipo JSON serializable
    except Exception as e:
        return jsonify({"error": f"Error durante la predicción: {str(e)}"}), 500
    return jsonify({'predictions': prediction_value})
if __name__ == '__main__':
    app.run(debug=True)





@app.route("/api/v1/retrain/", methods=["GET"])
def retrain():
    try:
        path = "/data/Students_Social_Media_Addiction.csv"
        if not os.path.exists(path):
            return jsonify({"error": f"Archivo no encontrado en: {path}"}), 404
        data = pd.read_csv(path)
        X = data.drop(columns=["Addicted_Score"])
        y = data["Addicted_Score"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42
        )
        model = Lasso(alpha=6000)
        model.fit(X_train, y_train)
        rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
        mape = mean_absolute_percentage_error(y_test, model.predict(X_test))
        with open("model_2.pkl", "wb") as f:
            pickle.dump(model, f)
        return f"Model retrained. New evaluation metric RMSE: {rmse:.4f}, MAPE: {mape:.4f}"
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)