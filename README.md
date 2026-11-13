# Steam Games Success Predictor

Aplicación web interactiva y modelo de Machine Learning para predecir la probabilidad de éxito comercial de un videojuego en Steam según sus características técnicas y comerciales.

## Enlaces
* **App en Streamlit:** *[Streamlit](https://predictor-de-exito-para-juegos-de-steam.streamlit.app/)*
* **Notebook de Entrenamiento:** [Google Colab](https://colab.research.google.com/drive/126IYgBpHnkDqhlc_QveGBV5meZkTo67i?usp=sharing)
* **Dataset:** [(Kaggle)](https://www.kaggle.com/datasets/fronkongames/steam-games-dataset/data)

## Stack Tecnológico
* **Lenguaje:** Python
* **Datos y ML:** `pandas`, `scikit-learn`, `joblib`
* **Interfaz Web:** `streamlit`

## Modelo
* **Preprocesamiento:** Limpieza de nulos, creación de variable objetivo binaria (`success`) y codificación *One-Hot* para géneros y plataformas.
* **Algoritmo:** **Random Forest Classifier** (86.5% Accuracy / 0.84 F1-Score), seleccionado tras superar al *baseline* de Regresión Logística (74.2%)[cite: 4].

## 🚀 Ejecución Local
```bash
git clone [https://github.com/GabrielPadilla04/Predictor-de-xito-de-juegos-en-steam.git](https://github.com/GabrielPadilla04/Predictor-de-xito-de-juegos-en-steam.git)
cd Predictor-de-xito-de-juegos-en-steam
pip install -r requirements.txt
streamlit run steam-appExitoJuego.py
