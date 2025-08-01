from flask import Flask, jsonify, request
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np
from sklearn.linear_model import LinearRegression

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
        path = "data/New_Students_Addiction.csv"
        if not os.path.exists(path):
            return jsonify({"error": f"Archivo no encontrado en: {path}"}), 404

        data = pd.read_csv(path)

        # Variables esperadas
        col_age = ["Age_19", "Age_20", "Age_21", "Age_22", "Age_23", "Age_24"]
        col_cont = ["Continent_Asia", "Continent_Europe", "Continent_North America", "Continent_Oceania", "Continent_South America"]

        # Validación de columnas necesarias
        required_columns = ["Age", "Continent", "Sleep_Hours_Per_Night", "Addicted_Score"]
        for col in required_columns:
            if col not in data.columns:
                return jsonify({"error": f"Columna faltante en CSV: {col}"}), 400

        # Filtrar valores válidos
        df = data.copy()
        df = df[df["Age"].isin([19, 20, 21, 22, 23, 24])]
        df = df[df["Continent"].isin([c.split("_")[1] for c in col_cont])]

        # One-hot encoding
        age_dummies = pd.get_dummies(df["Age"], prefix="Age")
        cont_dummies = pd.get_dummies(df["Continent"], prefix="Continent")
        df = pd.concat([df[["Sleep_Hours_Per_Night", "Addicted_Score"]], age_dummies, cont_dummies], axis=1)

        # Asegurar columnas completas
        for col in col_age + col_cont:
            if col not in df.columns:
                df[col] = 0

        # Orden correcto
        df = df[["Sleep_Hours_Per_Night"] + col_cont + col_age + ["Addicted_Score"]]

        X = df.drop(columns=["Addicted_Score"])
        y = df["Addicted_Score"]

        # Entrenar modelo
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Métricas
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = mean_absolute_percentage_error(y_test, y_pred)

        # Guardar modelo
        with open("model_2.pkl", "wb") as f:
            pickle.dump(model, f)

        return jsonify({
            "message": "Modelo reentrenado correctamente",
            "RMSE": round(rmse, 4),
            "MAPE": round(mape, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True)

