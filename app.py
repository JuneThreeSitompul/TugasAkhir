import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import json
import plotly
import plotly.express as px
from flask import Flask, jsonify, render_template, request, redirect, url_for
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold, train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

app = Flask(__name__)



# Load GeoJSON untuk Provinsi di Indonesia
with open("indonesia_provinces.json", "r") as geojson_file:
    geojson_data = json.load(geojson_file)

# ==========================
#  CLASS UNTUK PREPROCESSING
# ==========================
class Preprocessor:
    def __init__(self, label_encoder_path="labelencode.pkl"):
        self.label_encoder_path = label_encoder_path
        self.le = None

    def fit_label_encoder(self, df, column):
        """Melakukan Label Encoding dan menyimpannya ke file"""
        self.le = LabelEncoder()
        df[column] = self.le.fit_transform(df[column])
        with open(self.label_encoder_path, "wb") as f:
            pickle.dump(self.le, f)
        return df

    def load_label_encoder(self):
        """Memuat LabelEncoder dari file"""
        with open(self.label_encoder_path, "rb") as f:
            self.le = pickle.load(f)

    def transform_label(self, df, column):
        """Mengubah kolom kategorikal menggunakan LabelEncoder yang sudah disimpan"""
        if self.le is None:
            self.load_label_encoder()
        df[column] = self.le.transform(df[column])
        return df

    def clean_data(self, df):
        # 1. Hapus semua spasi dari nama kolom
        df.columns = df.columns.str.strip().str.lower()

        # 2. Hapus spasi & ubah ke huruf kecil di isi sel string
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].str.strip().str.lower()

        # 3. Ubah format angka: "22.450,14" -> 22450.14
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.replace(".", "", regex=False)
                df[col] = df[col].str.replace(",", ".", regex=False)

        # 4. Ubah '-' jadi NaN
        df.replace("-", np.nan, inplace=True)

        # 5. Ubah ke tipe data numerik
        for col in df.columns:
            try:
                df[col] = df[col].astype(float)
            except:
                pass

        # Ubah tahun ke int kalau ada
        if "tahun" in df.columns:
            try:
                df["tahun"] = df["tahun"].astype(int)
            except:
                pass

        # 6. Isi nilai kosong numerik dengan median
        for col in df.select_dtypes(include=['float', 'int']).columns:
            df[col].fillna(df[col].median(), inplace=True)

        return df



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/training')
def training_page():
    return render_template('training.html')

@app.route('/testing')
def testing_page():
    return render_template('testing.html')

@app.route('/prediction')
def predict_page():
    return render_template('prediction.html')

@app.route('/result-testing')
def result_testing_page():
    return render_template('result-testing.html')

@app.route('/result-prediction')
def result_prediction_page():
    return render_template('result-prediction.html')


#PROSES TRAINING
import pandas as pd
import numpy as np
import pickle
import json
import plotly.express as px
import plotly
import joblib
from flask import request, redirect, url_for, render_template
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

@app.route('/model-training', methods=['POST'])
def training():
    file = request.files['dataset']
    if file and file.filename.endswith('.csv'):
        df = pd.read_csv(file, sep=";")

    # === PREPROCESS ===
    preprocessor = Preprocessor()
    df = preprocessor.clean_data(df)
    df = preprocessor.fit_label_encoder(df, "provinsi")

    # Simpan encoder
    with open("labelencode.pkl", "wb") as f:
        pickle.dump(preprocessor.le, f)

    X = df.drop(columns=["ipm"])
    y = df["ipm"]

    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [3, 5, 7],
        "min_samples_split": [2, 4, 6],
        "min_samples_leaf": [1, 2, 3],
        "random_state": [7, 21, 42, 123]
    }

    all_results = []
    best_score = -np.inf
    best_model = None
    best_model_params = None

    for n_splits in [3, 5, 10]:
        tscv = TimeSeriesSplit(n_splits=n_splits)
        model = RandomForestRegressor()
        grid = GridSearchCV(model, param_grid, cv=tscv, scoring="r2", return_train_score=False)
        grid.fit(X, y)

        for i in range(len(grid.cv_results_["params"])):
            params = grid.cv_results_["params"][i]
            mean_score = grid.cv_results_["mean_test_score"][i]
            result_row = {
                "n_splits": n_splits,
                "max_depth": params["max_depth"],
                "min_samples_leaf": params["min_samples_leaf"],
                "min_samples_split": params["min_samples_split"],
                "n_estimators": params["n_estimators"],
                "random_state": params["random_state"],
                "mean_test_score": mean_score
            }
            all_results.append(result_row)

            if mean_score > best_score:
                best_score = mean_score
                best_model = grid.best_estimator_
                best_model_params = result_row.copy()

    # Simpan model terbaik
    with open("best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    # Simpan best model info
    with open("best_model_info.json", "w") as f:
        json.dump(best_model_params, f)

    # Simpan semua hasil grid search
    pd.DataFrame(all_results).to_csv("gridsearch_results.csv", index=False)

    # === Tambahkan Feature Importance Chart ===
    importance = best_model.feature_importances_
    feature_names = X.columns

    importance_df = pd.DataFrame({
        "Indikator": feature_names,
        "Pengaruh": importance
    }).sort_values(by="Pengaruh", ascending=False)

    importance_df.to_csv("feature_importance.csv", index=False)

    feature_chart = px.bar(
        importance_df,
        x="Indikator",
        y="Pengaruh",
        title="Pengaruh Indikator terhadap Prediksi IPM"
    )
    feature_chart_json = json.dumps(feature_chart, cls=plotly.utils.PlotlyJSONEncoder)

    # Simpan chart JSON ke file sementara
    with open("feature_chart.json", "w") as f:
        f.write(feature_chart_json)

    return redirect(url_for("training_result"))

@app.route("/result-training")
def training_result():
    results_df = pd.read_csv("gridsearch_results.csv")
    results = results_df.to_dict(orient="records")

    with open("best_model_info.json", "r") as f:
        best_model = json.load(f)

    with open("feature_chart.json", "r") as f:
        chart = f.read()

    return render_template("result-training.html", results=results, best_model=best_model, chart=chart)



# PROSES TESTING
@app.route("/model-testing", methods=["POST"])
def testing():
    result_json = None
    map_json = None
    chart_json = None
    provinsi = []
    ipm_prediksi = []

    mse = None
    rmse = None
    r2 = None

    if request.method == "POST":
        file = request.files['dataset']
        if file and file.filename.endswith('.csv'):
            df_test = pd.read_csv(file, sep=";")

        # Preprocessing
        preprocessor = Preprocessor()
        df_test = preprocessor.clean_data(df_test)
        df_test = preprocessor.transform_label(df_test, "provinsi")

        # Load model
        model = joblib.load("best_model.pkl", "rb")

        X = df_test.drop(columns=["ipm"], errors='ignore')
        df_test["ipm_prediksi"] = model.predict(X)

        # Evaluasi jika IPM aktual tersedia
        if "ipm" in df_test.columns:
            df_test["Selisih"] = df_test["ipm_prediksi"] - df_test["ipm"]
            mse = mean_squared_error(df_test["ipm"], df_test["ipm_prediksi"])
            rmse = np.sqrt(mse)
            r2 = r2_score(df_test["ipm"], df_test["ipm_prediksi"])

        result_json = df_test.to_dict(orient="records")

        # Bar Chart Data
        provinsi = df_test["provinsi"].tolist()
        ipm_prediksi = df_test["ipm_prediksi"].tolist()

        # Map Visualization
        #fig_map = px.choropleth(
            #df_test,
            #geojson=geojson_data,
            #locations="provinsi",
            #featureidkey="properties.name",
            #color="ipm_prediksi",
            #title="prediksi ipm"
        #)
        #fig_map.update_geos(fitbounds="locations", visible=False)
        #map_json = json.dumps(fig_map, cls=plotly.utils.PlotlyJSONEncoder)

        # Bar Chart Visualization
        #fig_chart = px.bar(
         #   df_test,
          #  x="provinsi",
           # y="ipm_prediksi",
            #title="Prediksi ipm per Provinsi"
        #)
        #chart_json = json.dumps(fig_chart, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template(
        "result-testing.html",
        result=result_json,
        map=map_json,
        chart=chart_json,
        provinsi=provinsi,
        ipm_prediksi=ipm_prediksi,
        mse=round(mse, 2) if mse else None,
        rmse=round(rmse, 2) if rmse else None,
        r2=round(r2, 2) if r2 else None
    )



#PROSES PREDKSI
@app.route("/model-prediction", methods=["POST"])
def prediction():
    result_json = None
    map_json = None
    chart_json = None

    if request.method == "POST":
        file = request.files['dataset']
        if file and file.filename.endswith('.csv'):
            df = pd.read_csv(file, sep=";")

        preprocessor = Preprocessor()
        df = preprocessor.clean_data(df)
        df["provinsi_asli"] = df["provinsi"]
        df = preprocessor.transform_label(df, "provinsi")

        model = joblib.load(open("best_model.pkl", "rb"))
        X = df.drop(columns=["provinsi_asli", "ipm"], errors="ignore")
        df["ipm_prediksi"] = model.predict(X)
        df["provinsi"] = df["provinsi_asli"]
        df.drop(columns=["provinsi_asli"], inplace=True)

        # Gabungkan ke GeoJSON (kode kamu sebelumnya)

        # Untuk tabel preview
        result_json = df.to_dict(orient="records")

        # === Load dan buat grafik feature importance ===
        importance_df = pd.read_csv("feature_importance.csv")
        fig = px.bar(
            importance_df,
            x="Indikator",
            y="Pengaruh",
            title="Pengaruh Indikator terhadap Prediksi IPM"
        )
        chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template(
        "result-prediction.html",
        result=result_json,
        chart=chart_json
    )

if __name__ == "__main__":
    app.run(debug=True)