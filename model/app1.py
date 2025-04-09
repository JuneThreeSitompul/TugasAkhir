import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import json
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
        # Ganti karakter aneh jadi normal, lalu strip dan lowercase
        # df.columns = df.columns.str.replace(r"\s+", " ", regex=True)  # Normalisasi spasi
        # Bersihkan nama kolom (hapus spasi & huruf kecil semua)
        df.columns = df.columns.str.strip().str.lower()

        # Bersihkan nilai string dari whitespace
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

        # Ganti tanda "-" kosong menjadi NaN
        df.replace(r"^\s*-\s*$", np.nan, regex=True, inplace=True)

        
        # Memperbaiki kolom pendapatan
        if 'pendapatan per kapita (ribu rp)' in df.columns:
            df['pendapatan per kapita (ribu rp)'] = (
                df['pendapatan per kapita (ribu rp)']
                .astype(str)
                .str.replace('.', '', regex=False)  # hapus titik ribuan
                .str.replace(',', '.', regex=False)  # ubah koma ke titik
            )
        df= df.replace(',', '.', regex = True)
        #Simpan kolom kategorikal (non-numeric), misal: provinsi
        categorical_cols = df.select_dtypes(include='object').columns

        #Ubah kolom numerik saja ke float
        for col in df.columns.difference(categorical_cols):
            df[col] = pd.to_numeric(df[col], errors='coerce')

        #Isi NaN di kolom numerik dengan median
        df.fillna(df.median(numeric_only=True).round(2), inplace=True)

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

@app.route('/predict')
def predict_page():
    return render_template('prediction.html')

@app.route('/result-testing')
def result_testing_page():
    return render_template('result-testing.html')

@app.route('/result-training')
def result_training_page():
    return render_template('result-training.html')

@app.route('/result-prediction')
def result_prediction_page():
    return render_template('result-prediction.html')


@app.route('/model-training', methods=['POST'])
def training():
        file = request.files['dataset']
        if file and file.filename.endswith('.csv'):
            df = pd.read_csv(file, sep=';')
    
    # Preprocessing
        preprocessor = Preprocessor()
        df = preprocessor.clean_data(df)
        # print("Kolom setelah clean_data:", df.columns.tolist())
        # Tambahkan sebelum fit_label_encoder
        df = preprocessor.fit_label_encoder(df, "provinsi")  # Encode provinsi dan simpan LabelEncoder

        # Pisahkan fitur (X) dan target (y)
        X = df.drop(columns=["ipm"])
        y = df["ipm"]

    # ==========================
    #PROSES MODEL
    # ==========================     
        # Grid Search CV dengan berbagai parameter
        param_grid = {
            "n_estimators": [50, 100, 150],
            "max_depth": [3, 5, 7],
            "min_samples_split": [2, 4, 6],
            "min_samples_leaf": [1, 2, 3],
            "random_state": [7, 21, 42, 123]
        }
        n_splits_list = [3, 5, 7]

        best_model = None
        best_score = float("-inf")
        best_params = None
        best_n_splits = None

        result = []
        grid_results_all = []

        # Asumsikan X dan y sudah ada sebelumnya dan berurutan berdasarkan waktu
        for n_splits in n_splits_list:
            tscv = TimeSeriesSplit(n_splits=n_splits)
            grid_search = GridSearchCV(
                estimator=RandomForestRegressor(),
                param_grid=param_grid,
                cv=tscv,
                scoring="r2",
                n_jobs=-1
            )
            grid_search.fit(X, y)

            # Menyimpan hasil terbaik
            if grid_search.best_score_ > best_score:
                best_score = grid_search.best_score_
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                best_n_splits = n_splits

            # Menyimpan ringkasan hasil terbaik per n_split
            result.append({
                'n_splits': n_splits,
                'r2_score': grid_search.best_score_,
                'max_depth': grid_search.best_params_['max_depth'],
                'min_samples_leaf': grid_search.best_params_['min_samples_leaf'],
                'min_samples_split': grid_search.best_params_['min_samples_split'],
                'n_estimators': grid_search.best_params_['n_estimators'],
                'random_state': grid_search.best_params_['random_state']
            })

            # Simpan semua hasil GridSearchCV
            df = pd.DataFrame(grid_search.cv_results_)
            df["param_n_splits"] = n_splits  # Tambahkan kolom n_splits secara manual
            grid_results_all.append(df)

        # Gabungkan semua DataFrame hasil GridSearchCV
        grid_results_df = pd.concat(grid_results_all, ignore_index=True)

        # Ambil kolom yang ingin ditampilkan
        columns = [
            'param_n_splits', 'param_max_depth', 'param_min_samples_leaf',
            'param_min_samples_split', 'param_n_estimators', 'param_random_state',
            'mean_test_score', 'rank_test_score'
        ]

        # Rename kolom agar mudah dibaca di HTML
        grid_results_df = grid_results_df[columns].rename(columns={
            'param_n_splits': 'n_splits',
            'param_max_depth': 'max_depth',
            'param_min_samples_leaf': 'min_samples_leaf',
            'param_min_samples_split': 'min_samples_split',
            'param_n_estimators': 'n_estimators',
            'param_random_state': 'random_state',
            'mean_test_score': 'mean_test_score',
            'rank_test_score': 'rank_test_score'
        })

        # Urutkan berdasarkan skor terbaik
        grid_results_df = grid_results_df.sort_values(by="mean_test_score", ascending=False)

        # Konversi ke list of dict untuk dikirim ke Jinja
        grid_results = grid_results_df.to_dict(orient='records')

        # Simpan model terbaik
        with open("best_model.pkl", "wb") as f:
            pickle.dump(best_model, f)

        # Muat ulang best_model (opsional)
        with open("best_model.pkl", "rb") as f:
            best_model = pickle.load(f)

        # Kirim semua variabel ke template
        return render_template("result-training.html", 
                            result=result,
                            best_n_splits=best_n_splits,
                            best_params=best_params,
                            best_score=best_score,
                            best_model=best_model,
                            grid_results=grid_results)



# PROSES PREDIKSI
@app.route("/model-prediction", methods=["POST"])
def prediction():
    result_json = None
    map_json = None
    chart_json = None
    provinsi = []
    ipm_prediksi = []

    if request.method == "POST":
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            df_test = pd.read_csv(file)

        # Preprocessing
        preprocessor = Preprocessor()
        df_test = preprocessor.clean_data(df_test)
        df_test = preprocessor.transform_label(df_test, "provinsi")

        # Load model
        model = joblib.load("best_model.pkl", "rb")

        X = df_test.drop(columns=["IPM"], errors='ignore')
        df_test["IPM_Prediksi"] = model.predict(X)

        if "IPM" in df_test.columns:
            df_test["Selisih"] = df_test["IPM_Prediksi"] - df_test["IPM"]

        result_json = df_test.to_dict(orient="records")

        print("Provinsi:", df_test["provinsi"].tolist())  # Cek apakah kolom provinsi ada
        print("IPM Prediksi:", df_test["IPM_Prediksi"].tolist())  # Cek hasil prediksi

        # Bar Chart Data
        provinsi = df_test["provinsi"].tolist()
        ipm_prediksi = df_test["IPM_Prediksi"].tolist()

        # Map Visualization
        fig_map = px.choropleth(
            df_test,
            geojson=geojson_data,
            locations="provinsi",
            featureidkey="properties.name",
            color="IPM_Prediksi",
            title="Prediksi IPM"
        )
        fig_map.update_geos(fitbounds="locations", visible=False)
        map_json = json.dumps(fig_map, cls=plotly.utils.PlotlyJSONEncoder)

        # Bar Chart Visualization (optional, jika ingin dikirim sebagai Plotly chart)
        fig_chart = px.bar(
            df_test,
            x="provinsi",
            y="IPM_Prediksi",
            title="Prediksi IPM per Provinsi"
        )
        chart_json = json.dumps(fig_chart, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template(
        "result-prediction.html",
        result=result_json,
        map=map_json,
        chart=chart_json,
        provinsi=provinsi,
        ipm_prediksi=ipm_prediksi        
    )


if __name__ == "__main__":
    app.run(debug=True)