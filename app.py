import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
import os

# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="NEXHEALTH SURVEY NO TABOO",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== STYLE CSS (CLAIR + SOMBRE) ====================
st.markdown("""
<style>
    /* Fond général - adapté au thème sombre */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8f5e9 100%);
    }
    
    /* Mode sombre : le navigateur gère automatiquement */
    @media (prefers-color-scheme: dark) {
        .stApp {
            background: linear-gradient(135deg, #0a1929 0%, #071a0c 100%);
        }
        .stMarkdown, p, li, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: #e0e0e0 !important;
        }
        .stMetric label, .stMetric div {
            color: #e0e0e0 !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            background-color: #1e2a1e !important;
        }
        .stTabs [data-baseweb="tab"] {
            color: #e0e0e0 !important;
        }
        .interpretation {
            background: #1a2a1a !important;
            color: #e0e0e0 !important;
        }
        .prevention-card {
            background: #1e2a1e !important;
            color: #e0e0e0 !important;
        }
        .stAlert {
            background-color: #2e3a2e !important;
        }
    }
    
    /* Bannière slogan */
    .slogan {
        background: linear-gradient(90deg, #1b5e20, #2e7d32);
        padding: 20px;
        border-radius: 15px;
        color: white !important;
        text-align: center;
        margin-bottom: 25px;
    }
    .slogan p {
        color: white !important;
    }
    
    /* Pied de page */
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background: linear-gradient(90deg, #1b5e20, #2e7d32);
        color: white;
        text-align: center;
        padding: 8px;
        font-size: 0.8rem;
        z-index: 999;
    }
    
    /* Titre principal */
    .main-title {
        text-align: center;
        margin-bottom: 20px;
    }
    .main-title h1 {
        color: #1b5e20;
        font-weight: bold;
    }
    @media (prefers-color-scheme: dark) {
        .main-title h1 {
            color: #4caf50 !important;
        }
    }
    
    /* Cartes prévention */
    .prevention-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border-left: 5px solid #2e7d32;
    }
    
    /* Interprétation */
    .interpretation {
        background: #e3f2fd;
        padding: 15px;
        border-radius: 10px;
        margin: 15px 0;
        border-left: 5px solid #2196f3;
    }
    
    /* Mode selector */
    .mode-banner {
        padding: 15px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 20px;
    }
    .mode-banner h3, .mode-banner p {
        margin: 0;
        font-weight: bold;
    }
    
    /* Boutons carrés dans la sidebar */
    .stButton > button {
        background-color: #1976d2 !important;
        color: white !important;
        border-radius: 15px !important;
        padding: 20px 10px !important;
        height: auto !important;
        min-height: 90px !important;
        white-space: normal !important;
        font-weight: bold !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-3px);
        background-color: #0d47a1 !important;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    /* Bouton AJOUTER mis en avant */
    .stButton:first-child button {
        background-color: #2e7d32 !important;
    }
    .stButton:first-child button:hover {
        background-color: #1b5e20 !important;
    }
    
    /* Onglets stylisés */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f1f5eb;
        padding: 6px;
        border-radius: 40px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 32px !important;
        padding: 8px 20px !important;
        font-weight: 600 !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2e7d32 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ==================== TITRE ET SLOGAN ====================
st.markdown("<h1 style='text-align:center; color:#1b5e20;'>🩺 NEXHEALTH SURVEY NO TABOO</h1>", unsafe_allow_html=True)
st.markdown("""
<div class="slogan">
    <p><i><strong>✨ Parce qu'en santé, il n'y a pas de tabou. ✨</strong></i></p>
    <p><i><strong>🔍 Brisons le silence sur les IST, protégeons notre bien-être.</strong></i></p>
    <p><i><strong>⚖️ La santé sexuelle est un droit, la protection est une responsabilité.</strong></i></p>
</div>
""", unsafe_allow_html=True)

# ==================== BASE DE DONNÉES SQLITE ====================
DB_NAME = "nexhealth.db"

def init_database():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS participants (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT, age INTEGER, sexe TEXT, pays TEXT, profession TEXT, niveau_etude TEXT,
            partenaires_sexuels TEXT, utilisation_preservatifs TEXT, nb_partenaires TEXT,
            rapport_non_protege TEXT, alcool_substances TEXT, connaissance_ist TEXT,
            deja_depiste TEXT, participation_campagnes TEXT, influence_reseaux_sociaux TEXT,
            ist_diagnostiquee TEXT, vaccin_hpv TEXT
        )
    ''')
    conn.commit()
    conn.close()

def sauvegarder_participant(data):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        INSERT INTO participants (
            date, age, sexe, pays, profession, niveau_etude, partenaires_sexuels,
            utilisation_preservatifs, nb_partenaires, rapport_non_protege, alcool_substances,
            connaissance_ist, deja_depiste, participation_campagnes, influence_reseaux_sociaux,
            ist_diagnostiquee, vaccin_hpv
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    ''', data)
    conn.commit()
    conn.close()

def charger_participants():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM participants", conn)
    conn.close()
    return df

def supprimer_toutes_donnees():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("DELETE FROM participants")
    conn.commit()
    conn.close()

init_database()

# ==================== DONNÉES DÉMO (30 EXEMPLES) ====================
def get_demo_data():
    np.random.seed(42)
    demo_data = []
    ages = np.random.choice(range(18, 65), 30)
    sexes = np.random.choice(["Homme", "Femme"], 30)
    pays_list = ["Cameroun", "Sénégal", "Côte d'Ivoire", "Nigéria", "Kenya"]
    prof_list = ["Étudiant", "Employé", "Indépendant", "Fonctionnaire"]
    connais_list = ["Très mauvaise", "Mauvaise", "Moyenne", "Bonne", "Très bonne"]
    preserv_list = ["Jamais", "Rarement", "Parfois", "Souvent", "Systématiquement"]
    partenaires_list = ["0", "1", "2-5", "6-10", "11-20"]  # AJOUT DE L'OPTION "0"
    campagnes_list = ["Jamais", "Rarement", "Parfois", "Souvent", "Très souvent"]
    influence_list = ["Négativement", "Neutre", "Positivement"]
    
    for i in range(30):
        demo_data.append({
            'id': i+1,
            'date': datetime.now().strftime("%Y-%m-%d %H:%M"),
            'age': int(ages[i]),
            'sexe': sexes[i],
            'pays': np.random.choice(pays_list),
            'profession': np.random.choice(prof_list),
            'niveau_etude': "Universitaire",
            'partenaires_sexuels': "Oui",
            'utilisation_preservatifs': np.random.choice(preserv_list, p=[0.15,0.15,0.25,0.25,0.2]),
            'nb_partenaires': np.random.choice(partenaires_list, p=[0.1,0.3,0.4,0.15,0.05]),
            'rapport_non_protege': np.random.choice(["Jamais", "Une fois", "Plusieurs fois"], p=[0.4,0.35,0.25]),
            'alcool_substances': np.random.choice(["Jamais", "Rarement", "Parfois", "Souvent"], p=[0.5,0.3,0.15,0.05]),
            'connaissance_ist': np.random.choice(connais_list, p=[0.1,0.2,0.3,0.25,0.15]),
            'deja_depiste': np.random.choice(["Jamais", "Une fois", "Plusieurs fois", "Régulièrement"], p=[0.4,0.3,0.2,0.1]),
            'participation_campagnes': np.random.choice(campagnes_list, p=[0.25,0.2,0.25,0.2,0.1]),
            'influence_reseaux_sociaux': np.random.choice(influence_list, p=[0.2,0.4,0.4]),
            'ist_diagnostiquee': np.random.choice(["Non", "Faible", "Modéré", "Élevé"], p=[0.6,0.2,0.1,0.1]),
            'vaccin_hpv': np.random.choice(["Non", "Oui", "Je ne sais pas"], p=[0.6,0.2,0.2])
        })
    return pd.DataFrame(demo_data)

# ==================== MODES ====================
if 'mode' not in st.session_state:
    st.session_state.mode = "demo"
if 'page' not in st.session_state:
    st.session_state.page = "ajouter"

# ==================== SIDEBAR AVEC 4 BOUTONS CARRÉS ====================
with st.sidebar:
    st.markdown("### 👩‍💻 Réalisé par")
    st.markdown("**MADJOU FORTUNE NESLINE - 24G2876**")
    st.markdown("📚 **Programme INF232 EC2**")
    st.markdown("---")
    
    # Sélecteur de mode (BLEU POUR LES DEUX)
    st.markdown("**⚙️ MODE DE L'APPLICATION**")
    mode_col1, mode_col2 = st.columns(2)
    with mode_col1:
        if st.button("🔬 DÉMO", use_container_width=True, type="primary" if st.session_state.mode == "demo" else "secondary"):
            st.session_state.mode = "demo"
            st.rerun()
    with mode_col2:
        if st.button("📝 NORMAL", use_container_width=True, type="primary" if st.session_state.mode == "normal" else "secondary"):
            st.session_state.mode = "normal"
            st.rerun()
    
    st.markdown("---")
    st.markdown("### 📂 MENU PRINCIPAL")
    
    # 4 BOUTONS CARRÉS (AJOUTER mis en avant automatiquement par CSS)
    if st.button("✏️➕ AJOUTER UN PARTICIPANT", use_container_width=True):
        st.session_state.page = "ajouter"
        st.rerun()
    
    if st.button("📋📊 ENREGISTREMENTS", use_container_width=True):
        st.session_state.page = "participants"
        st.rerun()
    
    if st.button("📈🔬 ANALYSES AVANCÉES", use_container_width=True):
        st.session_state.page = "analyses"
        st.rerun()
    
    if st.button("🛡️📚 PRÉVENTION IST", use_container_width=True):
        st.session_state.page = "prevention"
        st.rerun()
    
    st.markdown("---")
    
    # Formulaire de collecte (visible uniquement si page = ajouter)
    if st.session_state.page == "ajouter":
        st.header("📝 Formulaire de collecte")
        st.markdown("*Toutes vos réponses sont anonymes*")
        
        with st.form("collecte"):
            st.markdown("### 👤 Votre profil")
            age = st.slider("Âge", 15, 95, 25)
            sexe = st.radio("Sexe", ["Homme", "Femme", "Autre"])
            pays = st.selectbox("Pays", ["Cameroun", "Sénégal", "Côte d'Ivoire", "Nigéria", "Kenya", "Autre"])
            profession = st.selectbox("Profession", ["Étudiant", "Employé", "Indépendant", "Fonctionnaire", "Sans emploi"])
            niveau_etude = st.selectbox("Niveau d'étude", ["Secondaire", "Universitaire", "Supérieur", "Aucun"])
            
            st.markdown("---")
            st.markdown("### 💕 Habitudes")
            utilisation_preservatifs = st.select_slider("Utilisation des préservatifs", options=["Jamais", "Rarement", "Parfois", "Souvent", "Systématiquement"])
            nb_partenaires = st.selectbox("Nombre de partenaires", ["0", "1", "2-5", "6-10", "11-20", "20+"])  # AJOUT DE "0"
            rapport_non_protege = st.selectbox("Avez-vous eu un rapport non protégé ?", ["Jamais", "Une fois", "Plusieurs fois"])
            
            st.markdown("---")
            st.markdown("### 🏥 Connaissance")
            connaissance_ist = st.select_slider("Connaissance des IST", options=["Très mauvaise", "Mauvaise", "Moyenne", "Bonne", "Très bonne"])
            deja_depiste = st.radio("Déjà dépisté ?", ["Jamais", "Une fois", "Plusieurs fois", "Régulièrement"])
            participation_campagnes = st.select_slider("Participation aux campagnes", options=["Jamais", "Rarement", "Parfois", "Souvent", "Très souvent"])
            
            st.markdown("---")
            st.markdown("### 📱 Réseaux sociaux")
            influence_reseaux = st.select_slider("Influence des réseaux sociaux", options=["Négativement", "Neutre", "Positivement"])
            
            submit = st.form_submit_button("✅ Envoyer")
            
            if submit:
                if st.session_state.mode == "normal":
                    data = (
                        datetime.now().strftime("%Y-%m-%d %H:%M"),
                        age, sexe, pays, profession, niveau_etude,
                        "Oui", utilisation_preservatifs, nb_partenaires,
                        rapport_non_protege, "Parfois", connaissance_ist,
                        deja_depiste, participation_campagnes, influence_reseaux,
                        "Non renseigné", "Non"
                    )
                    sauvegarder_participant(data)
                    st.success("✅ Merci ! Votre réponse est enregistrée.")
                    st.balloons()
                else:
                    st.info("ℹ️ Mode Démo actif : Les données ne sont pas sauvegardées. Passez en Mode Normal pour enregistrer.")
    
    st.metric("👥 Participants", len(st.session_state.donnees) if 'donnees' in st.session_state else 0)

# ==================== AFFICHAGE DU MODE ACTUEL (BLEU) ====================
if st.session_state.mode == "demo":
    st.markdown("""
    <div class="mode-banner" style="background:#1565c0;">
        <h3 style="color:white;">🔬 MODE DÉMONSTRATION ACTIF</h3>
        <p style="color:#e3f2fd;">30 exemples fictifs - Aucune donnée sauvegardée</p>
    </div>
    """, unsafe_allow_html=True)
    df = get_demo_data()
else:
    st.markdown("""
    <div class="mode-banner" style="background:#1565c0;">
        <h3 style="color:white;">📝 MODE NORMAL ACTIF</h3>
        <p style="color:#e3f2fd;">Données réelles sauvegardées dans la base SQLite</p>
    </div>
    """, unsafe_allow_html=True)
    df = charger_participants()
    if len(df) == 0:
        st.info("📭 Aucune donnée réelle pour le moment. Utilisez le formulaire pour ajouter des participants.")
        df = pd.DataFrame(columns=[
            'id', 'date', 'age', 'sexe', 'pays', 'profession', 'niveau_etude',
            'partenaires_sexuels', 'utilisation_preservatifs', 'nb_partenaires',
            'rapport_non_protege', 'alcool_substances', 'connaissance_ist',
            'deja_depiste', 'participation_campagnes', 'influence_reseaux_sociaux',
            'ist_diagnostiquee', 'vaccin_hpv'
        ])

# ==================== PRÉPARATION DES DONNÉES ====================
df['Age'] = pd.to_numeric(df['age'], errors='coerce')
df['Connaissance_num'] = df['connaissance_ist'].map({'Très mauvaise':1,'Mauvaise':2,'Moyenne':3,'Bonne':4,'Très bonne':5})
df['Preservatifs_num'] = df['utilisation_preservatifs'].map({'Jamais':1,'Rarement':2,'Parfois':3,'Souvent':4,'Systématiquement':5})
df['Campagnes_num'] = df['participation_campagnes'].map({'Jamais':1,'Rarement':2,'Parfois':3,'Souvent':4,'Très souvent':5})
df['Influence_num'] = df['influence_reseaux_sociaux'].map({'Négativement':1,'Neutre':2,'Positivement':3})
df['Partenaires_num'] = df['nb_partenaires'].map({'0':0,'1':1,'2-5':2,'6-10':3,'11-20':4,'20+':5})

# Score de risque
df['Score_Risque'] = (
    (6 - df['Preservatifs_num']) * 2 +
    df['Partenaires_num'] * 1.5 +
    df['rapport_non_protege'].map({'Jamais':0, 'Une fois':1, 'Plusieurs fois':2}) * 2 +
    (df['Connaissance_num'] < 3).astype(int) * 2
)
df['Categorie_Risque'] = df['Score_Risque'].apply(lambda x: 'Faible' if x <= 8 else ('Modéré' if x <= 15 else 'Élevé'))

df_clean = df.dropna(subset=['Age', 'Connaissance_num', 'Preservatifs_num', 'Campagnes_num', 'Partenaires_num'])

# ==================== ONGLETS ====================
tab1 = "📋 Participants"
tab2 = "📈 Régression simple"
tab3 = "🔬 Régression multiple"
tab4 = "🎯 PCA"
tab5 = "🏷️ Classification (Risque IST)"
tab6 = "🔄 Clustering"
tab7 = "📊 Graphiques"
tab8 = "🛡️ PRÉVENTION IST"

tabs = st.tabs([tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8])

# ============================================================
# TAB 0 : PARTICIPANTS
# ============================================================
with tabs[0]:
    st.header("📋 Participants à l'étude")
    if len(df_clean) > 0:
        st.dataframe(df_clean, use_container_width=True)
        csv = df_clean.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Télécharger les données (CSV)", csv, "donnees_ist.csv", "text/csv")
    else:
        st.info("Aucune donnée disponible pour le moment.")

# ============================================================
# TAB 1 : RÉGRESSION SIMPLE
# ============================================================
with tabs[1]:
    st.header("📈 Régression simple : Âge → Connaissance des IST")
    st.markdown("**Objectif :** Vérifier si l'âge influence le niveau de connaissance des IST.")
    
    if len(df_clean) >= 3:
        X = df_clean[['Age']].values
        y = df_clean['Connaissance_num'].values
        modele = LinearRegression().fit(X, y)
        
        fig = px.scatter(df_clean, x='Age', y='Connaissance_num', 
                         title="Âge vs Connaissance des IST",
                         labels={'Connaissance_num': 'Niveau (1=Très mauvaise, 5=Très bonne)'},
                         color='Categorie_Risque', hover_data=['profession'])
        x_range = np.linspace(df_clean['Age'].min(), df_clean['Age'].max(), 100)
        y_pred = modele.predict(x_range.reshape(-1, 1))
        fig.add_trace(go.Scatter(x=x_range, y=y_pred, mode='lines', 
                                name='Tendance', line=dict(color='#2e7d32', width=3)))
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📐 Coefficient", f"{modele.coef_[0]:.3f}")
            if modele.coef_[0] > 0:
                st.success("✅ Plus on est âgé, meilleure est la connaissance des IST")
            else:
                st.info("⚠️ Les jeunes ont une meilleure connaissance des IST")
        with col2:
            r2 = r2_score(y, modele.predict(X))
            st.metric("🎯 R² (qualité)", f"{r2:.3f}")
            if r2 > 0.5:
                st.success("✅ L'âge explique bien les différences de connaissance")
            else:
                st.info("⚠️ D'autres facteurs que l'âge influencent la connaissance")
        
        st.info("""
        **📖 Interprétation :**
        - Chaque point représente un participant
        - La ligne verte montre la tendance générale
        - Les couleurs indiquent le niveau de risque IST estimé
        - R² proche de 1 = l'âge est un bon prédicteur
        """)
    else:
        st.warning(f"⚠️ Besoin d'au moins 3 participants. Actuellement : {len(df_clean)}")

# ============================================================
# TAB 2 : RÉGRESSION MULTIPLE
# ============================================================
with tabs[2]:
    st.header("🔬 Régression multiple : Facteurs influençant la connaissance des IST")
    st.markdown("**Objectif :** Identifier quels comportements sont liés à une meilleure connaissance des IST.")
    
    if len(df_clean) >= 5:
        X = df_clean[['Age', 'Preservatifs_num', 'Partenaires_num', 'Campagnes_num']].values
        y = df_clean['Connaissance_num'].values
        modele = LinearRegression().fit(X, y)
        
        coef_df = pd.DataFrame({
            'Facteur': ['Âge', 'Utilisation préservatifs', 'Nombre de partenaires', 'Participation campagnes'],
            'Coefficient': modele.coef_,
            'Impact': ['Positif' if c > 0 else 'Négatif' for c in modele.coef_]
        })
        st.dataframe(coef_df, use_container_width=True)
        
        predictions = modele.predict(X)
        fig = px.scatter(x=y, y=predictions, color=df_clean['Categorie_Risque'],
                         title="Qualité du modèle : prédictions vs réalité",
                         labels={'x': 'Connaissance réelle', 'y': 'Connaissance prédite'})
        fig.add_trace(go.Scatter(x=[1,5], y=[1,5], mode='lines', 
                                name='Prédiction parfaite', line=dict(dash='dash', color='#2e7d32')))
        st.plotly_chart(fig, use_container_width=True)
        
        r2 = r2_score(y, predictions)
        st.metric("📊 R² du modèle", f"{r2:.3f}")
        
        st.info("""
        **📖 Interprétation :**
        - Un coefficient POSITIF = plus le facteur augmente, meilleure est la connaissance
        - Un coefficient NÉGATIF = plus le facteur augmente, moins bonne est la connaissance
        - Les points proches de la ligne verte indiquent une bonne prédiction
        - Un R² élevé (proche de 1) signifie que ces facteurs expliquent bien les différences
        """)
    else:
        st.warning(f"⚠️ Besoin d'au moins 5 participants. Actuellement : {len(df_clean)}")

# ============================================================
# TAB 3 : PCA
# ============================================================
with tabs[3]:
    st.header("🎯 Analyse en Composantes Principales (PCA)")
    st.markdown("**Objectif :** Visualiser les profils similaires dans un espace réduit à 2 dimensions.")
    
    if len(df_clean) >= 4:
        features = ['Age', 'Connaissance_num', 'Preservatifs_num', 'Partenaires_num', 'Campagnes_num']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_clean[features])
        
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(X_scaled)
        
        df_viz = pd.DataFrame({
            'PC1': pca_result[:, 0],
            'PC2': pca_result[:, 1],
            'Risque': df_clean['Categorie_Risque'],
            'Âge': df_clean['Age']
        })
        
        fig = px.scatter(df_viz, x='PC1', y='PC2', color='Risque', size='Âge',
                         title="Projection des profils (les points proches se ressemblent)",
                         labels={'PC1': f'Dimension 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                                'PC2': f'Dimension 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)'})
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        **📖 Interprétation :**
        - Les POINTS PROCHES ont des comportements similaires face aux IST
        - Les COULEURS indiquent le niveau de risque estimé
        - Plus la variance expliquée est élevée, plus la projection est fidèle à la réalité
        - Observez si les points de même couleur ont tendance à se regrouper
        """)
    else:
        st.warning(f"⚠️ Besoin d'au moins 4 participants. Actuellement : {len(df_clean)}")

# ============================================================
# TAB 4 : CLASSIFICATION (PRÉDICTION DU RISQUE AVEC CONSEILS PERSONNALISÉS)
# ============================================================
with tabs[4]:
    st.header("🏷️ Classification : Prédire son risque de contracter une IST")
    st.markdown("**Objectif :** Le modèle apprend à estimer votre niveau de risque selon vos habitudes.")
    
    if len(df_clean) >= 6:
        df_clean['Cible_Risque'] = (df_clean['Categorie_Risque'] == 'Élevé').astype(int)
        
        X = df_clean[['Age', 'Preservatifs_num', 'Partenaires_num', 'Campagnes_num', 'Connaissance_num']].values
        y = df_clean['Cible_Risque'].values
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=4)
        rf.fit(X, y)
        
        # Importance des facteurs
        importance_df = pd.DataFrame({
            'Facteur': ['Âge', 'Préservatifs', 'Nombre de partenaires', 'Participation campagnes', 'Connaissance IST'],
            'Importance (%)': (rf.feature_importances_ * 100).round(1)
        }).sort_values('Importance (%)', ascending=False)
        st.dataframe(importance_df, use_container_width=True)
        
        fig_imp = px.bar(importance_df, x='Importance (%)', y='Facteur', orientation='h',
                         title="Facteurs influençant le risque IST")
        st.plotly_chart(fig_imp, use_container_width=True)
        
        # Section test interactif
        st.subheader("🔮 Évaluez VOTRE niveau de risque")
        st.markdown("Renseignez vos habitudes ci-dessous pour une estimation personnalisée.")
        
        col1, col2 = st.columns(2)
        with col1:
            age_test = st.slider("Votre âge", 18, 65, 25, key="risk_age")
            preserv_test = st.select_slider("Utilisation des préservatifs", 
                                           options=["Systématiquement", "Souvent", "Parfois", "Rarement", "Jamais"],
                                           key="risk_preserv")
            nb_partenaires_test = st.select_slider("Nombre de partenaires (environ)", options=["0","1", "2-5", "6-10", "11-20", "20+"],
                                                   key="risk_partenaires")
        with col2:
            campagnes_test = st.select_slider("Participation aux campagnes de dépistage",
                                             options=["Très souvent", "Souvent", "Parfois", "Rarement", "Jamais"],
                                             key="risk_campagnes")
            connais_test = st.select_slider("Connaissance des IST",
                                           options=["Très bonne", "Bonne", "Moyenne", "Mauvaise", "Très mauvaise"],
                                           key="risk_connais")
        
        if st.button("🔮 Estimer mon risque", key="predict_risk"):
            # Conversion
            preserv_map = {"Systématiquement":5, "Souvent":4, "Parfois":3, "Rarement":2, "Jamais":1}
            campagnes_map = {"Très souvent":5, "Souvent":4, "Parfois":3, "Rarement":2, "Jamais":1}
            connais_map = {"Très bonne":5, "Bonne":4, "Moyenne":3, "Mauvaise":2, "Très mauvaise":1}
            partenaires_map = {"0":0, "1":1, "2-5":2, "6-10":3, "11-20":4, "20+":5}
            
            preserv_val = preserv_map[preserv_test]
            camp_val = campagnes_map[campagnes_test]
            connais_val = connais_map[connais_test]
            partenaires_val = partenaires_map[nb_partenaires_test]
            
            pred = rf.predict([[age_test, preserv_val, partenaires_val, camp_val, connais_val]])[0]
            proba = rf.predict_proba([[age_test, preserv_val, partenaires_val, camp_val, connais_val]]).max()
            
            # IDENTIFICATION DES FACTEURS DE RISQUE
            facteurs_risque = []
            if preserv_val <= 2:
                facteurs_risque.append("⚠️ Utilisation irrégulière des préservatifs")
            if partenaires_val >= 4:
                facteurs_risque.append("⚠️ Nombre élevé de partenaires")
            if camp_val <= 2:
                facteurs_risque.append("⚠️ Dépistage rare ou absent")
            if connais_val <= 2:
                facteurs_risque.append("⚠️ Con