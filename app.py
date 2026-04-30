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

# ==================== CONFIGURATION ====================
st.set_page_config(page_title="NEXHEALTH SURVEY NO TABOO", page_icon="🩺", layout="wide")

# ==================== CSS POUR DESIGN 4 BOUTONS CARRÉS ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;500;600;700;800&display=swap');
    * {
        font-family: 'Manrope', sans-serif !important;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f0f7f0 0%, #e8f5e9 100%);
    }
    
    .mode-indicator {
        background: linear-gradient(90deg, #1b5e20, #2e7d32);
        padding: 15px;
        border-radius: 30px;
        text-align: center;
        margin: 15px 0 25px 0;
        color: white;
        font-weight: bold;
        font-size: 1.3rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .mode-indicator-demo {
        background: linear-gradient(90deg, #2e7d32, #4caf50);
    }
    
    .mode-indicator-normal {
        background: linear-gradient(90deg, #1565c0, #2196f3);
    }
    
    .square-btn {
        background: white;
        border-radius: 20px;
        padding: 30px 10px;
        text-align: center;
        transition: all 0.3s ease;
        border: 1px solid #e0e4da;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        cursor: pointer;
        margin: 10px;
    }
    
    .square-btn:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(27, 94, 32, 0.15);
        border-color: #2e7d32;
    }
    
    .square-btn-selected {
        background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
        border: 2px solid #2e7d32;
    }
    
    .square-icon {
        font-size: 3rem;
        margin-bottom: 15px;
    }
    
    .square-title {
        font-size: 1rem;
        font-weight: 700;
        color: #1b5e20;
    }
    
    .footer {
        text-align: center;
        padding: 20px;
        margin-top: 40px;
        background: linear-gradient(90deg, #1b5e20, #2e7d32);
        color: white;
        border-radius: 20px;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ==================== TITRE ET SLOGAN ====================
st.title("🩺 NEXHEALTH SURVEY NO TABOO")
st.markdown("""
*✨ Parce qu'en santé, il n'y a pas de tabou. ✨*  
*🔍 Brisons le silence sur les IST, protégeons notre bien-être.*  
*⚖️ La santé sexuelle est un droit, la protection est une responsabilité.*
""")

# ==================== SESSION STATE ====================
if 'mode' not in st.session_state:
    st.session_state.mode = "demo"
if 'page' not in st.session_state:
    st.session_state.page = "ajouter"

# ==================== SQLITE (Mode Normal) ====================
DB_NAME = "nexhealth.db"

def init_database():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS participants (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Date TEXT, Age INTEGER, Sexe TEXT, Pays TEXT, Profession TEXT,
            Niveau_Etude TEXT, Partenaires_Sexuels TEXT, Utilisation_Preservatifs TEXT,
            Nb_Partenaires TEXT, Rapport_Non_Protege TEXT, Alcool_Substances TEXT,
            Connaissance_IST TEXT, Deja_Depiste TEXT, Participation_Campagnes TEXT,
            Influence_Reseaux_Sociaux TEXT, IST_Diagnostiquee TEXT, Vaccin_HPV TEXT
        )
    ''')
    conn.commit()
    conn.close()

def sauvegarder_participant(data):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        INSERT INTO participants (
            Date, Age, Sexe, Pays, Profession, Niveau_Etude, Partenaires_Sexuels,
            Utilisation_Preservatifs, Nb_Partenaires, Rapport_Non_Protege, Alcool_Substances,
            Connaissance_IST, Deja_Depiste, Participation_Campagnes, Influence_Reseaux_Sociaux,
            IST_Diagnostiquee, Vaccin_HPV
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
    partenaires_list = ["1", "2-5", "6-10", "11-20"]
    campagnes_list = ["Jamais", "Rarement", "Parfois", "Souvent", "Très souvent"]
    influence_list = ["Négativement", "Neutre", "Positivement"]
    
    for i in range(30):
        demo_data.append({
            'Date': datetime.now().strftime("%Y-%m-%d %H:%M"),
            'Age': int(ages[i]),
            'Sexe': sexes[i],
            'Pays': np.random.choice(pays_list),
            'Profession': np.random.choice(prof_list),
            'Niveau_Etude': "Universitaire",
            'Partenaires_Sexuels': "Oui",
            'Utilisation_Preservatifs': np.random.choice(preserv_list, p=[0.15,0.15,0.25,0.25,0.2]),
            'Nb_Partenaires': np.random.choice(partenaires_list, p=[0.3,0.4,0.2,0.1]),
            'Rapport_Non_Protege': np.random.choice(["Jamais", "Une fois", "Plusieurs fois"], p=[0.4,0.35,0.25]),
            'Alcool_Substances': np.random.choice(["Jamais", "Rarement", "Parfois", "Souvent"], p=[0.5,0.3,0.15,0.05]),
            'Connaissance_IST': np.random.choice(connais_list, p=[0.1,0.2,0.3,0.25,0.15]),
            'Deja_Depiste': np.random.choice(["Jamais", "Une fois", "Plusieurs fois", "Régulièrement"], p=[0.4,0.3,0.2,0.1]),
            'Participation_Campagnes': np.random.choice(campagnes_list, p=[0.25,0.2,0.25,0.2,0.1]),
            'Influence_Reseaux_Sociaux': np.random.choice(influence_list, p=[0.2,0.4,0.4]),
            'IST_Diagnostiquee': np.random.choice(["Non", "Faible", "Modéré", "Élevé"], p=[0.6,0.2,0.1,0.1]),
            'Vaccin_HPV': np.random.choice(["Non", "Oui", "Je ne sais pas"], p=[0.6,0.2,0.2])
        })
    return pd.DataFrame(demo_data)

# ==================== CHARGEMENT DES DONNÉES SELON LE MODE ====================
def get_current_data():
    if st.session_state.mode == "demo":
        return get_demo_data()
    else:
        df = charger_participants()
        if len(df) == 0:
            st.warning("Aucune donnée en mode Normal. Utilisez le formulaire pour ajouter des participants.")
        return df

# ==================== AFFICHAGE DU MODE ACTUEL ====================
if st.session_state.mode == "demo":
    st.markdown("""
    <div class="mode-indicator mode-indicator-demo">
        🔬 MODE DÉMO ACTIF (30 exemples fictifs) 🔬
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="mode-indicator mode-indicator-normal">
        📝 MODE NORMAL ACTIF (Données réelles sauvegardées) 📝
    </div>
    """, unsafe_allow_html=True)

# ==================== SIDEBAR MODE SELECTOR ====================
with st.sidebar:
    st.markdown("### 👩‍💻 **MADJOU FORTUNE NESLINE - 24G2876**")
    st.markdown("📚 **Programme INF232 EC2**")
    st.markdown("---")
    
    st.markdown("### ⚙️ **Changer de mode**")
    col_mode1, col_mode2 = st.columns(2)
    with col_mode1:
        if st.button("🔬 Mode Démo", use_container_width=True, 
                     type="primary" if st.session_state.mode == "demo" else "secondary"):
            st.session_state.mode = "demo"
            st.rerun()
    with col_mode2:
        if st.button("📝 Mode Normal", use_container_width=True,
                     type="primary" if st.session_state.mode == "normal" else "secondary"):
            st.session_state.mode = "normal"
            st.rerun()
    
    st.markdown("---")
    
    st.markdown("### 👤 **Mon profil**")
    age_sidebar = st.slider("Âge", 15, 95, 25)
    sexe_sidebar = st.radio("Sexe", ["Homme", "Femme", "Autre"])
    pays_sidebar = st.selectbox("Pays", ["Cameroun", "Sénégal", "Côte d'Ivoire", "Nigéria", "Kenya", "Autre"])
    profession_sidebar = st.selectbox("Profession", ["Étudiant", "Employé", "Indépendant", "Fonctionnaire", "Sans emploi"])
    
    st.markdown("---")
    st.markdown("### 💕 **Habitudes**")
    preservatifs_sidebar = st.select_slider("Utilisation des préservatifs", options=["Jamais", "Rarement", "Parfois", "Souvent", "Systématiquement"])
    nb_partenaires_sidebar = st.selectbox("Nombre de partenaires", ["1", "2-5", "6-10", "11-20", "20+"])
    rapport_non_protege_sidebar = st.selectbox("Avez-vous eu un rapport non protégé ?", ["Jamais", "Une fois", "Plusieurs fois"])
    
    st.markdown("---")
    st.markdown("### 🏥 **Connaissance**")
    connaissance_ist_sidebar = st.select_slider("Connaissance des IST", options=["Très mauvaise", "Mauvaise", "Moyenne", "Bonne", "Très bonne"])
    deja_depiste_sidebar = st.radio("Déjà dépisté ?", ["Jamais", "Une fois", "Plusieurs fois", "Régulièrement"])
    participation_campagnes_sidebar = st.select_slider("Participation aux campagnes", options=["Jamais", "Rarement", "Parfois", "Souvent", "Très souvent"])
    
    st.markdown("---")
    st.markdown("### 📱 **Réseaux sociaux**")
    influence_reseaux_sidebar = st.select_slider("Influence des réseaux sociaux", options=["Négativement", "Neutre", "Positivement"])
    
    st.markdown("---")
    submit_sidebar = st.button("✅ Envoyer ma participation", use_container_width=True)
    
    if submit_sidebar:
        if st.session_state.mode == "normal":
            nouvelle = pd.DataFrame([{
                'Date': datetime.now().strftime("%Y-%m-%d %H:%M"),
                'Age': age_sidebar, 'Sexe': sexe_sidebar, 'Pays': pays_sidebar, 'Profession': profession_sidebar,
                'Niveau_Etude': "Universitaire", 'Partenaires_Sexuels': "Oui",
                'Utilisation_Preservatifs': preservatifs_sidebar, 'Nb_Partenaires': nb_partenaires_sidebar,
                'Rapport_Non_Protege': rapport_non_protege_sidebar, 'Alcool_Substances': "Parfois",
                'Connaissance_IST': connaissance_ist_sidebar, 'Deja_Depiste': deja_depiste_sidebar,
                'Participation_Campagnes': participation_campagnes_sidebar, 'Influence_Reseaux_Sociaux': influence_reseaux_sidebar,
                'IST_Diagnostiquee': "Non renseigné", 'Vaccin_HPV': "Non"
            }])
            data_tuple = tuple(nouvelle.iloc[0].values)
            sauvegarder_participant(data_tuple)
            st.success("✅ Merci ! Votre réponse est enregistrée.")
            st.balloons()
        else:
            st.info("ℹ️ Mode Démo actif : Les données ne sont pas sauvegardées. Passez en Mode Normal pour enregistrer.")

# ==================== MENU 4 BOUTONS CARRÉS ====================
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("✏️🫂\n**AJOUTER**\n*une personne*", use_container_width=True,
                 type="primary" if st.session_state.page == "ajouter" else "secondary"):
        st.session_state.page = "ajouter"
        st.rerun()

with col2:
    if st.button("📋👥\n**ENREGISTREMENTS**\n*participants*", use_container_width=True,
                 type="primary" if st.session_state.page == "participants" else "secondary"):
        st.session_state.page = "participants"
        st.rerun()

with col3:
    if st.button("📈🔬\n**ANALYSES**\n*avancées*", use_container_width=True,
                 type="primary" if st.session_state.page == "analyses" else "secondary"):
        st.session_state.page = "analyses"
        st.rerun()

with col4:
    if st.button("📚🩺\n**PRÉVENTION**\n*IST*", use_container_width=True,
                 type="primary" if st.session_state.page == "prevention" else "secondary"):
        st.session_state.page = "prevention"
        st.rerun()

st.markdown("---")

# ==================== CHARGEMENT DES DONNÉES ====================
df = get_current_data()

# Préparation des données (identique au code original)
if len(df) > 0:
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df['Connaissance_num'] = df['Connaissance_IST'].map({'Très mauvaise':1,'Mauvaise':2,'Moyenne':3,'Bonne':4,'Très bonne':5})
    df['Preservatifs_num'] = df['Utilisation_Preservatifs'].map({'Jamais':1,'Rarement':2,'Parfois':3,'Souvent':4,'Systématiquement':5})
    df['Campagnes_num'] = df['Participation_Campagnes'].map({'Jamais':1,'Rarement':2,'Parfois':3,'Souvent':4,'Très souvent':5})
    df['Influence_num'] = df['Influence_Reseaux_Sociaux'].map({'Négativement':1,'Neutre':2,'Positivement':3})
    df['Partenaires_num'] = df['Nb_Partenaires'].map({'1':1,'2-5':2,'6-10':3,'11-20':4,'20+':5})

    df['Score_Risque'] = (
        (6 - df['Preservatifs_num']) * 2 +
        df['Partenaires_num'] * 1.5 +
        df['Rapport_Non_Protege'].map({'Jamais':0, 'Une fois':1, 'Plusieurs fois':2}) * 2 +
        (df['Connaissance_num'] < 3).astype(int) * 2
    )
    df['Categorie_Risque'] = df['Score_Risque'].apply(lambda x: 'Faible' if x <= 8 else ('Modéré' if x <= 15 else 'Élevé'))

    df_clean = df.dropna(subset=['Age', 'Connaissance_num', 'Preservatifs_num', 'Campagnes_num', 'Partenaires_num'])
else:
    df_clean = pd.DataFrame()

# ==================== PAGE 1 : AJOUTER UNE PERSONNE ====================
if st.session_state.page == "ajouter":
    st.header("✏️🫂 Ajouter une nouvelle personne")
    st.markdown("*Toutes vos réponses sont anonymes.*")
    
    if st.session_state.mode == "demo":
        st.warning("⚠️ **Mode Démo actif** : Les données ne seront pas sauvegardées. Passez en Mode Normal pour enregistrer.")
    else:
        st.info("📝 **Mode Normal** : Vos réponses seront sauvegardées dans la base de données.")
    
    with st.form("form_ajout", clear_on_submit=False if st.session_state.mode == "normal" else True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**👤 Profil**")
            age_f = st.slider("Âge", 15, 95, 25, key="f_age")
            sexe_f = st.radio("Sexe", ["Homme", "Femme", "Autre"], key="f_sexe")
            pays_f = st.selectbox("Pays", ["Cameroun", "Sénégal", "Côte d'Ivoire", "Nigéria", "Kenya", "Autre"], key="f_pays")
            profession_f = st.selectbox("Profession", ["Étudiant", "Employé", "Indépendant", "Fonctionnaire", "Sans emploi"], key="f_profession")
        with col2:
            st.markdown("**💕 Habitudes**")
            preservatifs_f = st.select_slider("Utilisation des préservatifs", options=["Jamais", "Rarement", "Parfois", "Souvent", "Systématiquement"], key="f_preserv")
            nb_partenaires_f = st.selectbox("Nombre de partenaires", ["1", "2-5", "6-10", "11-20", "20+"], key="f_partenaires")
            rapport_f = st.selectbox("Rapport non protégé ?", ["Jamais", "Une fois", "Plusieurs fois"], key="f_rapport")
            st.markdown("**🏥 Connaissance**")
            connais_f = st.select_slider("Connaissance des IST", options=["Très mauvaise", "Mauvaise", "Moyenne", "Bonne", "Très bonne"], key="f_connais")
            depist_f = st.radio("Déjà dépisté ?", ["Jamais", "Une fois", "Plusieurs fois", "Régulièrement"], key="f_depist")
            camp_f = st.select_slider("Participation aux campagnes", options=["Jamais", "Rarement", "Parfois", "Souvent", "Très souvent"], key="f_camp")
            st.markdown("**📱 Réseaux sociaux**")
            influ_f = st.select_slider("Influence des réseaux sociaux", options=["Négativement", "Neutre", "Positivement"], key="f_influ")
        
        submit_f = st.form_submit_button("✅ Envoyer", use_container_width=True)
        if submit_f and st.session_state.mode == "normal":
            nouvelle = pd.DataFrame([{
                'Date': datetime.now().strftime("%Y-%m-%d %H:%M"),
                'Age': age_f, 'Sexe': sexe_f, 'Pays': pays_f, 'Profession': profession_f,
                'Niveau_Etude': "Universitaire", 'Partenaires_Sexuels': "Oui",
                'Utilisation_Preservatifs': preservatifs_f, 'Nb_Partenaires': nb_partenaires_f,
                'Rapport_Non_Protege': rapport_f, 'Alcool_Substances': "Parfois",
                'Connaissance_IST': connais_f, 'Deja_Depiste': depist_f,
                'Participation_Campagnes': camp_f, 'Influence_Reseaux_Sociaux': influ_f,
                'IST_Diagnostiquee': "Non renseigné", 'Vaccin_HPV': "Non"
            }])
            data_tuple = tuple(nouvelle.iloc[0].values)
            sauvegarder_participant(data_tuple)
            st.success("✅ Merci ! Votre réponse a été enregistrée.")
            st.balloons()
            st.rerun()
        elif submit_f and st.session_state.mode == "demo":
            st.info("ℹ️ Mode Démo actif : Les données ne sont pas sauvegardées.")

# ==================== PAGE 2 : ENREGISTREMENTS ====================
elif st.session_state.page == "participants":
    st.header("📋👥 Participants enregistrés")
    
    if st.session_state.mode == "demo":
        st.info("📊 **Mode Démo** : Affichage des 30 exemples fictifs")
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Télécharger les données (CSV)", csv, "donnees_demo.csv", "text/csv")
    else:
        if len(df) == 0:
            st.info("📭 Aucun participant enregistré. Utilisez le formulaire pour ajouter des participants.")
        else:
            st.metric("Total participants", len(df))
            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False).encode('utf-8')
            col1, col2 = st.columns(2)
            with col1:
                st.download_button("📥 Exporter les données (CSV)", csv, "donnees_reelles.csv", "text/csv")
            with col2:
                if st.button("🗑️ Supprimer toutes les données", use_container_width=True):
                    supprimer_toutes_donnees()
                    st.success("Toutes les données ont été supprimées !")
                    st.rerun()

# ==================== PAGE 3 : ANALYSES AVANCÉES ====================
elif st.session_state.page == "analyses":
    st.header("📈🔬 Analyses avancées")
    
    if len(df_clean) < 3:
        st.warning(f"⚠️ Besoin d'au moins 3 participants pour les analyses. Actuellement : {len(df_clean)} participant(s).")
    else:
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📈 Régression simple", "🔬 Régression multiple", "🎯 PCA",
            "🏷️ Classification (Risque)", "🔄 Clustering", "📊 Graphiques"
        ])
        
        # Régression simple
        with tab1:
            with tab1:
                st.header("📈 Régression simple : Âge → Connaissance des IST")
                st.markdown("*Objectif :* Vérifier si l'âge influence le niveau de connaissance des IST.")
                
                if len(df_clean) >= 3:
                    X = df_clean[['Age']].values
                    y = df_clean['Connaissance_num'].values
                    modele = LinearRegression().fit(X, y)
                    
                    fig = px.scatter(df_clean, x='Age', y='Connaissance_num', 
                                     title="Âge vs Connaissance des IST",
                                     labels={'Connaissance_num': 'Niveau (1=Très mauvaise, 5=Très bonne)'},
                                     color='Categorie_Risque', hover_data=['Profession'])
                    x_range = np.linspace(df_clean['Age'].min(), df_clean['Age'].max(), 100)
                    y_pred = modele.predict(x_range.reshape(-1, 1))
                    fig.add_trace(go.Scatter(x=x_range, y=y_pred, mode='lines', 
                                            name='Tendance', line=dict(color='red', width=3)))
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
                    *📖 Interprétation :*
                    - Chaque point représente un participant
                    - La ligne rouge montre la tendance générale
                    - Les couleurs indiquent le niveau de risque IST estimé
                    - R² proche de 1 = bonne prédiction
                    """)
                else:
                    st.warning(f"⚠️ Besoin d'au moins 3 participants. Actuellement : {len(df_clean)}")
        
        # Régression multiple
        with tab2:
            with tab2:
                st.header("🔬 Régression multiple : Facteurs influençant la connaissance des IST")
                st.markdown("*Objectif :* Identifier quels comportements sont liés à une meilleure connaissance des IST.")
                
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
                                            name='Prédiction parfaite', line=dict(dash='dash', color='red')))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    r2 = r2_score(y, predictions)
                    st.metric("📊 R² du modèle", f"{r2:.3f}")
                    
                    st.info("""
                    *📖 Interprétation :*
                    - Un coefficient POSITIF = plus le facteur augmente, meilleure est la connaissance
                    - Un coefficient NÉGATIF = plus le facteur augmente, moins bonne est la connaissance
                    - Les points proches de la ligne rouge indiquent une bonne prédiction
                    """)
                else:
                    st.warning(f"⚠️ Besoin d'au moins 5 participants. Actuellement : {len(df_clean)}")
        
        # PCA
        with tab3:
            with tab3:
                st.header("🎯 Analyse en Composantes Principales (PCA)")
                st.markdown("*Objectif :* Visualiser les profils similaires dans un espace réduit à 2 dimensions.")
                
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
                    *📖 Interprétation :*
                    - Les POINTS PROCHES ont des comportements similaires
                    - Les COULEURS indiquent le niveau de risque IST
                    - Plus la variance expliquée est élevée, plus la projection est fidèle
                    """)
                else:
                    st.warning(f"⚠️ Besoin d'au moins 4 participants. Actuellement : {len(df_clean)}")
        
        # Classification
with tab4:
    with tab4:
        st.header("🏷️ Classification : Prédire son risque de contracter une IST")
        st.markdown("*Objectif :* Le modèle apprend à estimer votre niveau de risque selon vos habitudes.")
        
        if len(df_clean) >= 6:
            df_clean['Cible_Risque'] = (df_clean['Categorie_Risque'] == 'Élevé').astype(int)
            
            X = df_clean[['Age', 'Preservatifs_num', 'Partenaires_num', 'Campagnes_num', 'Connaissance_num']].values
            y = df_clean['Cible_Risque'].values
            
            rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=4)
            rf.fit(X, y)
            
            importance_df = pd.DataFrame({
                'Facteur': ['Âge', 'Préservatifs', 'Nombre de partenaires', 'Participation campagnes', 'Connaissance IST'],
                'Importance (%)': (rf.feature_importances_ * 100).round(1)
            }).sort_values('Importance (%)', ascending=False)
            st.dataframe(importance_df, use_container_width=True)
            
            fig_imp = px.bar(importance_df, x='Importance (%)', y='Facteur', orientation='h',
                             title="Facteurs influençant le risque IST")
            st.plotly_chart(fig_imp, use_container_width=True)
            
            st.subheader("🔮 Évaluez VOTRE niveau de risque")
            st.markdown("Renseignez vos habitudes ci-dessous pour une estimation personnalisée.")
            
            col1, col2 = st.columns(2)
            with col1:
                age_test = st.slider("Votre âge", 18, 65, 25, key="risk_age")
                preserv_test = st.select_slider("Utilisation des préservatifs", 
                                               options=["Systématiquement", "Souvent", "Parfois", "Rarement", "Jamais"],
                                               key="risk_preserv")
                nb_partenaires_test = st.select_slider("Nombre de partenaires (environ)", options=["1", "2-5", "6-10", "11-20", "20+"],
                                                       key="risk_partenaires")
            with col2:
                campagnes_test = st.select_slider("Participation aux campagnes de dépistage",
                                                 options=["Très souvent", "Souvent", "Parfois", "Rarement", "Jamais"],
                                                 key="risk_campagnes")
                connais_test = st.select_slider("Connaissance des IST",
                                               options=["Très bonne", "Bonne", "Moyenne", "Mauvaise", "Très mauvaise"],
                                               key="risk_connais")
            
            if st.button("🔮 Estimer mon risque", key="predict_risk"):
                preserv_map = {"Systématiquement":5, "Souvent":4, "Parfois":3, "Rarement":2, "Jamais":1}
                campagnes_map = {"Très souvent":5, "Souvent":4, "Parfois":3, "Rarement":2, "Jamais":1}
                connais_map = {"Très bonne":5, "Bonne":4, "Moyenne":3, "Mauvaise":2, "Très mauvaise":1}
                partenaires_map = {"1":1, "2-5":2, "6-10":3, "11-20":4, "20+":5}
                
                preserv_val = preserv_map[preserv_test]
                camp_val = campagnes_map[campagnes_test]
                connais_val = connais_map[connais_test]
                partenaires_val = partenaires_map[nb_partenaires_test]
                
                pred = rf.predict([[age_test, preserv_val, partenaires_val, camp_val, connais_val]])[0]
                proba = rf.predict_proba([[age_test, preserv_val, partenaires_val, camp_val, connais_val]]).max()
                
                if pred == 1:
                    st.error(f"⚠️ **Risque ÉLEVÉ** (confiance : {proba:.1%})")
                    st.markdown("""
                    **💡 Recommandations pour réduire votre risque :**
                    - Utilisez des préservatifs à chaque rapport
                    - Réduisez le nombre de partenaires
                    - Faites-vous dépister régulièrement (2 fois par an)
                    - Participez aux campagnes de sensibilisation
                    - Consultez un professionnel de santé
                    """)
                else:
                    st.success(f"✅ **Risque FAIBLE à MODÉRÉ** (confiance : {proba:.1%})")
                    st.markdown("""
                    **💡 Pour rester protégé(e) :**
                    - Continuez les bonnes pratiques
                    - Maintenez un dépistage régulier
                    - Sensibilisez votre entourage
                    """)
            
            try:
                scores = cross_val_score(rf, X, y, cv=min(3, len(np.unique(y))))
                st.caption(f"📊 Précision du modèle : {scores.mean():.1%} (basé sur {len(df_clean)} participants)")
            except:
                pass
        else:
            st.warning(f"⚠️ Besoin d'au moins 6 participants. Actuellement : {len(df_clean)}")