from pathlib import Path
import gzip
import pickle
import numpy as np
import pandas as pd
import shap
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title='Credit Risk Analytics', page_icon='📊', layout='wide')
st.markdown('''<style>
.block-container {padding-top:2rem;max-width:1250px}
h1,h2,h3 {letter-spacing:-.025em}
[data-testid="stMetric"] {background:rgba(120,130,150,.08);border:1px solid rgba(120,130,150,.16);padding:1.1rem;border-radius:14px}
</style>''', unsafe_allow_html=True)
BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE.parent / 'app_fastapi_heroku' / 'RFClassifier3.pkl.gz'
DATA_PATH = BASE / 'data_reduced3.csv'

@st.cache_resource
def load_model():
    with gzip.open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

try:
    model = load_model()
    df = load_data()
except (FileNotFoundError, OSError, pickle.UnpicklingError) as exc:
    st.error(f'Impossible de charger les ressources : {exc}')
    st.stop()

features = list(model.feature_names_in_)
missing = [col for col in ['SK_ID_CURR', *features] if col not in df.columns]
if missing:
    st.error(f'Colonnes absentes : {missing}')
    st.stop()
if list(model.classes_) != [0, 1]:
    st.error('Classes inattendues : vérifiez la convention de la classe de défaut.')
    st.stop()

st.title('Credit Risk Analytics')
st.caption('Scoring crédit · Random Forest · Explicabilité SHAP')
with st.sidebar:
    st.header('Paramètres')
    client_id = st.selectbox('Identifiant client', df['SK_ID_CURR'].dropna().astype(int).unique(), index=0)
    threshold = st.slider('Seuil de refus (exploratoire)', 0.01, 0.99, 0.50, 0.01)
    st.caption('Le seuil de 0,50 est provisoire : il ne constitue pas une politique de crédit validée.')

client_rows = df.loc[df['SK_ID_CURR'] == client_id, features]
if client_rows.empty:
    st.warning('Client introuvable.')
    st.stop()
client = client_rows.iloc[[0]].copy()
if client.isna().any().any():
    st.warning('Ce client présente des valeurs manquantes ; vérifiez le prétraitement du modèle.')

probability = float(model.predict_proba(client)[0, 1])
refused = probability >= threshold
col1, col2, col3 = st.columns(3)
col1.metric('Probabilité de défaut estimée', f'{probability:.1%}')
col2.metric('Classification au seuil choisi', 'Risque élevé' if refused else 'Risque faible')
col3.metric('Seuil exploratoire', f'{threshold:.0%}')
st.progress(probability)
st.caption('La classification est indicative. La calibration, le seuil métier et l’équité du modèle restent à valider avant tout usage décisionnel.')

tab1, tab2, tab3 = st.tabs(['Profil client', 'Explication individuelle', 'Importance globale'])
with tab1:
    st.subheader(f'Client {client_id}')
    st.dataframe(client.T.rename(columns={client.index[0]: 'Valeur'}), use_container_width=True)

with tab2:
    st.subheader('Contributions SHAP à la probabilité de défaut')
    with st.spinner('Calcul de l’explication SHAP…'):
        explainer = shap.TreeExplainer(model)
        explanation = explainer(client)
    local = explanation[0, :, 1]
    reconstructed = float(np.asarray(local.base_values).item() + np.asarray(local.values).sum())
    delta = abs(reconstructed - probability)
    st.write(f'Probabilité reconstruite : **{reconstructed:.4f}** · Écart numérique : **{delta:.2e}**')
    if delta > 1e-5:
        st.warning('La décomposition SHAP ne reproduit pas la prédiction : vérifier la configuration.')
    plt.close('all')
    shap.plots.waterfall(
        local,
        max_display=len(features),
        show=False
    )
    fig = plt.gcf()
    fig.set_size_inches(15, 7)
    fig.subplots_adjust(
        left=0.40,
        right=0.90,
        top=0.92,
        bottom=0.16
    )
    st.pyplot(fig, clear_figure=True)
    st.caption('Rouge : augmente le risque prédit ; bleu : le diminue. Ces contributions ne sont pas causales.')

with tab3:
    st.subheader('Importance globale des variables')
    if not hasattr(model, 'feature_importances_'):
        st.info('Le modèle ne fournit pas feature_importances_.')
    else:
        importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
        st.bar_chart(importance)
        st.caption('Importance par diminution d’impureté du Random Forest ; ce graphique ne représente pas des contributions SHAP globales.')
