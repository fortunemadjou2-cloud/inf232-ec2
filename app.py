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

# ==================== CSS DESIGN ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;500;600;700;800&display=swap');
    * { font-family: 'Manrope', sans-serif !important; }
    .stApp { background: linear-gradient(135deg, #f0f7f0 0%, #e8f5e9 100%); }
    .mode-indicator {
        padding: 15px;
        border-radius: 30px;
        text-align: center;
        margin: 15px 0 25px 0;
        color: white;
        font-weight: bold;
        font-size: 1.3rem;
    }
    .mode-indicator-demo { background: linear-gradient(90deg, #2e7d32, #4caf50); }
    .mode-indicator-normal { background: linear-gradient(90deg, #1565c0, #2196f3); }
    .footer {
        text-align: center;
        padding: 20px;
        margin-top: 40px;
        background: linear-gradient(90deg, #1b5e20, #2e7d32);
        color: white;
        border-radius: 20px;
    }
    .stButton > button {
        background-color: #2e7d32 !important;
        color: white !important;
        border-radius: 40px !important;
        padding: 10px 20px !important;
        font-weight: 600 !important;
    }
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
    .interpretation-box {
        background: #e8f5e9;
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        border-left: 5px solid #2e7d32;
        color: #1b5e20;
    }
    .advice-box {
        background: #e3f2fd;
        padding: 15px;
        border-radius: 12px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== TITRE ====================
st.markdown("<h1 style='text-align:center; color:#1b5e20;'>🩺 NEXHEALTH SURVEY NO TABOO</h1>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align:center; font-style:italic;'>✨ Parce qu'en santé, il n'y a pas de tabou. ✨</p>
<p style='text-align:center; font-style:italic;'>🔍 Brisons le silence sur les IST, protégeons notre bien-être.</p>
<p style='text-align:center; font-style:italic;'>⚖️ La santé sexuelle est un droit, la protection est une responsabilité.</p>
""", unsafe_allow_html=True)

# ==================== SESSION STATE ====================
if 'mode' not in st.session_state:
    st.session_state.mode = "demo"
if 'page' not in st.session_state:
    st.session_state.page = "ajouter"

# ==================== SQLITE ====================
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

# ==================== DONNÉES DÉMO ====================
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

def get_current_data():
    if st.session_state.mode == "demo":
        return get_demo_data()
    else:
        return charger_participants()

# ==================== AFFICHAGE MODE ====================
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

# ==================== SIDEBAR ====================
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
    st.markdown("### 📝 Formulaire de collecte")
    
    with st.form("collecte_sidebar"):
        st.markdown("### 👤 Votre profil")
        age = st.slider("Âge", 15, 95, 25)
        sexe = st.radio("Sexe", ["Homme", "Femme", "Autre"])
        pays = st.selectbox("Pays", ["Cameroun", "Sénégal", "Côte d'Ivoire", "Nigéria", "Kenya", "Autre"])
        profession = st.selectbox("Profession", ["Étudiant", "Employé", "Indépendant", "Fonctionnaire", "Sans emploi"])
        
        st.markdown("---")
        st.markdown("### 💕 Habitudes")
        utilisation_preservatifs = st.select_slider("Utilisation des préservatifs", options=["Jamais", "Rarement", "Parfois", "Souvent", "Systématiquement"])
        nb_partenaires = st.selectbox("Nombre de partenaires", ["1", "2-5", "6-10", "11-20", "20+"])
        rapport_non_protege = st.selectbox("Rapport non protégé ?", ["Jamais", "Une fois", "Plusieurs fois"])
        
        st.markdown("---")
        st.markdown("### 🏥 Connaissance")
        connaissance_ist = st.select_slider("Connaissance des IST", options=["Très mauvaise", "Mauvaise", "Moyenne", "Bonne", "Très bonne"])
        deja_depiste = st.radio("Déjà dépisté ?", ["Jamais", "Une fois", "Plusieurs fois", "Régulièrement"])
        participation_campagnes = st.select_slider("Participation aux campagnes", options=["Jamais", "Rarement", "Parfois", "Souvent", "Très souvent"])
        
        st.markdown("---")
        st.markdown("### 📱 Réseaux sociaux")
        influence_reseaux = st.select_slider("Influence des réseaux sociaux", options=["Négativement", "Neutre", "Positivement"])
        
        if st.form_submit_button("✅ Envoyer"):
            if st.session_state.mode == "normal":
                nouvelle = pd.DataFrame([{
                    'Date': datetime.now().strftime("%Y-%m-%d %H:%M"), 'Age': age, 'Sexe': sexe,
                    'Pays': pays, 'Profession': profession, 'Niveau_Etude': "Universitaire",
                    'Partenaires_Sexuels': "Oui", 'Utilisation_Preservatifs': utilisation_preservatifs,
                    'Nb_Partenaires': nb_partenaires, 'Rapport_Non_Protege': rapport_non_protege,
                    'Alcool_Substances': "Parfois", 'Connaissance_IST': connaissance_ist,
                    'Deja_Depiste': deja_depiste, 'Participation_Campagnes': participation_campagnes,
                    'Influence_Reseaux_Sociaux': influence_reseaux, 'IST_Diagnostiquee': "Non renseigné",
                    'Vaccin_HPV': "Non"
                }])
                sauvegarder_participant(tuple(nouvelle.iloc[0].values))
                st.success("✅ Merci ! Votre réponse est enregistrée.")
                st.balloons()
            else:
                st.info("ℹ️ Mode Démo actif : Les données ne sont pas sauvegardées.")
    
    st.metric("👥 Participants", len(get_current_data()))

# ==================== 4 BOUTONS CARRÉS ====================
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

if len(df) > 0:
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df['Connaissance_num'] = df['Connaissance_IST'].map({'Très mauvaise':1,'Mauvaise':2,'Moyenne':3,'Bonne':4,'Très bonne':5})
    df['Preservatifs_num'] = df['Utilisation_Preservatifs'].map({'Jamais':1,'Rarement':2,'Parfois':3,'Souvent':4,'Systématiquement':5})
    df['Campagnes_num'] = df['Participation_Campagnes'].map({'Jamais':1,'Rarement':2,'Parfois':3,'Souvent':4,'Très souvent':5})
    df['Influence_num'] = df['Influence_Reseaux_Sociaux'].map({'Négativement':1,'Neutre':2,'Positivement':3})
    df['Partenaires_num'] = df['Nb_Partenaires'].map({'1':1,'2-5':2,'6-10':3,'11-20':4,'20+':5})
    df['Rapport_num'] = df['Rapport_Non_Protege'].map({'Jamais':0, 'Une fois':1, 'Plusieurs fois':2})
    df['Alcool_num'] = df['Alcool_Substances'].map({'Jamais':0, 'Rarement':1, 'Parfois':2, 'Souvent':3, 'Très souvent':4})
    
    df['Score_Risque'] = (6 - df['Preservatifs_num']) * 2 + df['Partenaires_num'] * 1.5 + df['Rapport_num'] * 2 + (df['Connaissance_num'] < 3).astype(int) * 2 + df['Alcool_num'] * 0.5
    df['Categorie_Risque'] = df['Score_Risque'].apply(lambda x: 'Faible' if x <= 8 else ('Modéré' if x <= 15 else 'Élevé'))
    df_clean = df.dropna(subset=['Age', 'Connaissance_num', 'Preservatifs_num', 'Campagnes_num', 'Partenaires_num', 'Rapport_num', 'Alcool_num'])
else:
    df_clean = pd.DataFrame()

# ==================== PAGE 1 : AJOUTER ====================
if st.session_state.page == "ajouter":
    st.header("✏️🫂 Ajouter une nouvelle personne")
    
    if st.session_state.mode == "demo":
        st.warning("⚠️ Mode Démo actif : Les données ne seront pas sauvegardées.")
    else:
        st.info("📝 Mode Normal : Vos réponses seront sauvegardées.")
    
    with st.form("form_ajout"):
        col1, col2 = st.columns(2)
        with col1:
            age_f = st.slider("Âge", 15, 95, 25, key="f_age")
            sexe_f = st.radio("Sexe", ["Homme", "Femme", "Autre"], key="f_sexe")
            pays_f = st.selectbox("Pays", ["Cameroun", "Sénégal", "Côte d'Ivoire", "Nigéria", "Kenya", "Autre"], key="f_pays")
            profession_f = st.selectbox("Profession", ["Étudiant", "Employé", "Indépendant", "Fonctionnaire", "Sans emploi"], key="f_prof")
        with col2:
            preservatifs_f = st.select_slider("Utilisation des préservatifs", options=["Jamais", "Rarement", "Parfois", "Souvent", "Systématiquement"], key="f_preserv")
            nb_partenaires_f = st.selectbox("Nombre de partenaires", ["1", "2-5", "6-10", "11-20", "20+"], key="f_nb")
            rapport_f = st.selectbox("Rapport non protégé ?", ["Jamais", "Une fois", "Plusieurs fois"], key="f_rapport")
            connais_f = st.select_slider("Connaissance des IST", options=["Très mauvaise", "Mauvaise", "Moyenne", "Bonne", "Très bonne"], key="f_connais")
            depist_f = st.radio("Déjà dépisté ?", ["Jamais", "Une fois", "Plusieurs fois", "Régulièrement"], key="f_depist")
            camp_f = st.select_slider("Participation aux campagnes", options=["Jamais", "Rarement", "Parfois", "Souvent", "Très souvent"], key="f_camp")
            influ_f = st.select_slider("Influence des réseaux sociaux", options=["Négativement", "Neutre", "Positivement"], key="f_influ")
        
        if st.form_submit_button("✅ Envoyer", use_container_width=True):
            if st.session_state.mode == "normal":
                nouvelle = pd.DataFrame([{
                    'Date': datetime.now().strftime("%Y-%m-%d %H:%M"), 'Age': age_f, 'Sexe': sexe_f,
                    'Pays': pays_f, 'Profession': profession_f, 'Niveau_Etude': "Universitaire",
                    'Partenaires_Sexuels': "Oui", 'Utilisation_Preservatifs': preservatifs_f,
                    'Nb_Partenaires': nb_partenaires_f, 'Rapport_Non_Protege': rapport_f,
                    'Alcool_Substances': "Parfois", 'Connaissance_IST': connais_f,
                    'Deja_Depiste': depist_f, 'Participation_Campagnes': camp_f,
                    'Influence_Reseaux_Sociaux': influ_f, 'IST_Diagnostiquee': "Non renseigné", 'Vaccin_HPV': "Non"
                }])
                sauvegarder_participant(tuple(nouvelle.iloc[0].values))
                st.success("✅ Merci ! Votre réponse a été enregistrée.")
                st.balloons()
                st.rerun()
            else:
                st.info("ℹ️ Mode Démo actif : Les données ne sont pas sauvegardées.")

# ==================== PAGE 2 : ENREGISTREMENTS ====================
elif st.session_state.page == "participants":
    st.header("📋👥 Participants enregistrés")
    
    if st.session_state.mode == "demo":
        st.info("📊 Mode Démo : 30 exemples fictifs")
        st.dataframe(df, use_container_width=True)
        st.download_button("📥 Télécharger CSV", df.to_csv(index=False).encode('utf-8'), "donnees_demo.csv")
    else:
        if len(df) == 0:
            st.info("📭 Aucun participant. Utilisez le formulaire pour ajouter.")
        else:
            st.metric("Total participants", len(df))
            st.dataframe(df, use_container_width=True)
            col1, col2 = st.columns(2)
            with col1:
                st.download_button("📥 Exporter CSV", df.to_csv(index=False).encode('utf-8'), "donnees_reelles.csv")
            with col2:
                if st.button("🗑️ Supprimer toutes les données"):
                    supprimer_toutes_donnees()
                    st.rerun()

# ==================== PAGE 3 : ANALYSES ====================
elif st.session_state.page == "analyses":
    st.header("📈🔬 Analyses avancées")
    
    if len(df_clean) < 3:
        st.warning(f"⚠️ Besoin d'au moins 3 participants. Actuellement : {len(df_clean)}")
    else:
        t1, t2, t3, t4, t5, t6 = st.tabs([
            "📈 Régression simple", "🔬 Régression multiple", "🎯 PCA",
            "🏷️ Classification", "🔄 Clustering", "📊 Graphiques"
        ])
        
        # ========== REGRESSION SIMPLE ==========
        with t1:
            st.subheader("📈 Relation entre l'âge et la connaissance des IST")
            X = df_clean[['Age']].values
            y = df_clean['Connaissance_num'].values
            model = LinearRegression().fit(X, y)
            
            fig = px.scatter(df_clean, x='Age', y='Connaissance_num', color='Categorie_Risque',
                            title="Âge vs Connaissance des IST",
                            labels={'Connaissance_num': 'Niveau (1=Très mauvaise, 5=Très bonne)'})
            x_range = np.linspace(df_clean['Age'].min(), df_clean['Age'].max(), 100)
            fig.add_trace(go.Scatter(x=x_range, y=model.predict(x_range.reshape(-1,1)), mode='lines', name='Tendance', line=dict(color='red', width=3)))
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📐 Coefficient", f"{model.coef_[0]:.3f}")
                if model.coef_[0] > 0:
                    st.success("✅ Plus on est âgé, meilleure est la connaissance")
            with col2:
                st.metric("🎯 R²", f"{r2_score(y, model.predict(X)):.3f}")
            
            st.markdown("""
            <div class="interpretation-box">
            <b>📖 INTERPRÉTATION :</b><br>
            - Chaque point représente un participant<br>
            - La ligne rouge montre la tendance générale<br>
            - R² proche de 1 = l'âge explique bien les différences de connaissance
            </div>
            """, unsafe_allow_html=True)
        
        # ========== REGRESSION MULTIPLE ==========
        with t2:
            st.subheader("🔬 Facteurs influençant la connaissance des IST")
            X = df_clean[['Age', 'Preservatifs_num', 'Partenaires_num']].values
            y = df_clean['Connaissance_num'].values
            model = LinearRegression().fit(X, y)
            
            coef_df = pd.DataFrame({'Facteur':['Âge','Préservatifs','Partenaires'], 'Coefficient':model.coef_})
            st.dataframe(coef_df, use_container_width=True)
            
            pred = model.predict(X)
            fig = px.scatter(x=y, y=pred, title="Prédictions vs Réalité", labels={'x':'Réel', 'y':'Prédit'})
            fig.add_trace(go.Scatter(x=[1,5], y=[1,5], mode='lines', name='Parfait', line=dict(dash='dash', color='red')))
            st.plotly_chart(fig, use_container_width=True)
            st.metric("R²", f"{r2_score(y, pred):.3f}")
            
            st.markdown("""
            <div class="interpretation-box">
            <b>📖 INTERPRÉTATION :</b><br>
            - Coefficient POSITIF = ce facteur améliore la connaissance<br>
            - Coefficient NÉGATIF = ce facteur diminue la connaissance<br>
            - Points proches de la diagonale = bonne prédiction
            </div>
            """, unsafe_allow_html=True)
        
        # ========== PCA ==========
        with t3:
            st.subheader("🎯 Projection des profils (PCA)")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_clean[['Age', 'Connaissance_num', 'Preservatifs_num']])
            pca = PCA(n_components=2)
            result = pca.fit_transform(X_scaled)
            fig = px.scatter(x=result[:,0], y=result[:,1], color=df_clean['Categorie_Risque'],
                            title="Les points proches ont des profils similaires",
                            labels={'x': f'Dimension 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                                   'y': f'Dimension 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)'})
            st.plotly_chart(fig, use_container_width=True)
            st.info("Les points proches ont des comportements similaires face aux IST.")
        
        # ========== CLASSIFICATION AVEC 7 QUESTIONS ==========
        with t4:
            st.subheader("🏷️ Évaluez votre risque de contracter une IST")
            st.markdown("📊 **Ce modèle est basé sur les données enregistrées**")
            
            if len(df_clean) >= 6:
                df_clean['Cible'] = (df_clean['Categorie_Risque'] == 'Élevé').astype(int)
                X = df_clean[['Age', 'Preservatifs_num', 'Partenaires_num', 'Campagnes_num', 'Connaissance_num', 'Rapport_num', 'Alcool_num']].values
                y = df_clean['Cible'].values
                rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)
                
                st.subheader("🔮 7 questions pour évaluer votre risque")
                col1, col2 = st.columns(2)
                with col1:
                    age_t = st.slider("1. Votre âge", 18, 65, 25, key="age_t")
                    preserv_t = st.select_slider("2. Utilisation des préservatifs", options=["Systématiquement","Souvent","Parfois","Rarement","Jamais"], key="preserv_t")
                    nb_t = st.select_slider("3. Nombre de partenaires (dernière année)", options=["1","2-5","6-10","11-20","20+"], key="nb_t")
                    rapport_t = st.select_slider("4. Rapports non protégés récents", options=["Jamais","Une fois","Plusieurs fois"], key="rapport_t")
                with col2:
                    camp_t = st.select_slider("5. Participation aux campagnes", options=["Très souvent","Souvent","Parfois","Rarement","Jamais"], key="camp_t")
                    connais_t = st.select_slider("6. Connaissance des IST", options=["Très bonne","Bonne","Moyenne","Mauvaise","Très mauvaise"], key="connais_t")
                    alcool_t = st.select_slider("7. Alcool/substances avant rapports", options=["Jamais","Rarement","Parfois","Souvent","Très souvent"], key="alcool_t")
                
                if st.button("🔮 Estimer mon risque"):
                    p_map = {"Systématiquement":5,"Souvent":4,"Parfois":3,"Rarement":2,"Jamais":1}
                    k_map = {"1":1,"2-5":2,"6-10":3,"11-20":4,"20+":5}
                    c_map = {"Très souvent":5,"Souvent":4,"Parfois":3,"Rarement":2,"Jamais":1}
                    r_map = {"Jamais":0,"Une fois":1,"Plusieurs fois":2}
                    a_map = {"Jamais":0,"Rarement":1,"Parfois":2,"Souvent":3,"Très souvent":4}
                    con_map = {"Très bonne":5,"Bonne":4,"Moyenne":3,"Mauvaise":2,"Très mauvaise":1}
                    
                    pred = rf.predict([[age_t, p_map[preserv_t], k_map[nb_t], c_map[camp_t], con_map[connais_t], r_map[rapport_t], a_map[alcool_t]]])[0]
                    proba = rf.predict_proba([[age_t, p_map[preserv_t], k_map[nb_t], c_map[camp_t], con_map[connais_t], r_map[rapport_t], a_map[alcool_t]]]).max()
                    
                    if pred == 1:
                        st.error(f"⚠️ **Risque ÉLEVÉ** (confiance : {proba:.1%})")
                        st.markdown("""
                        <div class="advice-box">
                        <b>💡 SUGGESTIONS POUR RÉDUIRE VOTRE RISQUE :</b><br><br>
                        - ✅ Utilisez des préservatifs à chaque rapport sexuel<br>
                        - ✅ Réduisez votre nombre de partenaires<br>
                        - ✅ Faites-vous dépister régulièrement (2 fois par an)<br>
                        - ✅ Participez aux campagnes de sensibilisation<br>
                        - ✅ Informez-vous sur les IST (onglet Prévention)<br>
                        - ✅ Consultez un médecin pour la vaccination HPV<br>
                        - ✅ Évitez l'alcool ou les substances avant les rapports
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.success(f"✅ **Risque FAIBLE à MODÉRÉ** (confiance : {proba:.1%})")
                        st.markdown("""
                        <div class="advice-box">
                        <b>💡 CONSEILS POUR RESTER PROTÉGÉ(E) :</b><br><br>
                        - ✅ Continuez les bonnes pratiques<br>
                        - ✅ Maintenez un dépistage régulier<br>
                        - ✅ Restez informé(e) sur les IST<br>
                        - ✅ Sensibilisez votre entourage
                        </div>
                        """, unsafe_allow_html=True)
                
                try:
                    scores = cross_val_score(rf, X, y, cv=min(3, len(np.unique(y))))
                    st.caption(f"📊 Précision du modèle : {scores.mean():.1%} (basé sur {len(df_clean)} participants)")
                except:
                    pass
            else:
                st.warning(f"⚠️ Besoin d'au moins 6 participants. Actuellement : {len(df_clean)}")
        
        # ========== CLUSTERING ==========
        with t5:
            st.subheader("🔄 Segmentation des profils (K-Means)")
            if len(df_clean) >= 5:
                X = df_clean[['Age', 'Connaissance_num', 'Preservatifs_num']].values
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                k = st.slider("Nombre de segments", 2, 4, 3)
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(X_scaled)
                fig = px.scatter(df_clean, x='Age', y='Connaissance_num', color=clusters.astype(str), size='Preservatifs_num',
                                title=f"Segmentation en {k} groupes", hover_data=['Profession'])
                st.plotly_chart(fig, use_container_width=True)
                
                profil = df_clean.groupby(clusters)[['Age', 'Connaissance_num', 'Preservatifs_num']].mean().round(1)
                profil.columns = ['Âge moyen', 'Connaissance (1-5)', 'Préservatifs (1-5)']
                st.dataframe(profil)
                st.info("Chaque couleur représente un groupe aux habitudes similaires.")
            else:
                st.warning("Ajoutez au moins 5 participants")
        
        # ========== GRAPHIQUES ==========
        with t6:
            st.subheader("📊 Graphiques statistiques")
            col1, col2 = st.columns(2)
            with col1:
                fig_hist = px.histogram(df_clean, x='Age', nbins=15, title="Distribution des âges")
                st.plotly_chart(fig_hist, use_container_width=True)
                
                # Correction de l'erreur value_counts
                connais_counts = df_clean['Connaissance_IST'].value_counts().reset_index()
                connais_counts.columns = ['Niveau', 'Nombre']
                fig_bar = px.bar(connais_counts, x='Niveau', y='Nombre', title="Niveau de connaissance des IST")
                st.plotly_chart(fig_bar, use_container_width=True)
                
                fig_pie = px.pie(df_clean, names='Sexe', title="Répartition par sexe")
                st.plotly_chart(fig_pie, use_container_width=True)
            with col2:
                preserv_counts = df_clean['Utilisation_Preservatifs'].value_counts().reset_index()
                preserv_counts.columns = ['Fréquence', 'Nombre']
                fig_preserv = px.bar(preserv_counts, x='Fréquence', y='Nombre', title="Utilisation des préservatifs")
                st.plotly_chart(fig_preserv, use_container_width=True)
                
                fig_risk = px.pie(df_clean, names='Categorie_Risque', title="Niveau de risque IST")
                st.plotly_chart(fig_risk, use_container_width=True)

# ==================== PAGE 4 : PRÉVENTION ====================
elif st.session_state.page == "prevention":
    st.header("📚🩺 ESPACE PRÉVENTION IST")
    
    st.warning("⚠️ Ces informations ne remplacent pas l'avis d'un médecin.")
    
    with st.expander("📖 Définition et modes de contraction", expanded=True):
        st.markdown("""
        **IST (Infections Sexuellement Transmissibles)** : Infections transmises lors de rapports sexuels non protégés.
        
        **Modes de contraction :**
        - Rapports vaginaux, anaux, oraux non protégés
        - Partage de seringues contaminées
        - Transmission mère-enfant (grossesse, accouchement, allaitement)
        - Contact direct avec des lésions
        """)
    
    with st.expander("🦠 Principales IST (7 exemples)", expanded=True):
        st.markdown("""
        **1. VIH/Sida** - *Virus (VIH)* : Destruction immunitaire. Prévention : préservatifs, PrEP.
        
        **2. Syphilis** - *Bactérie (Treponema pallidum)* : Lésions, complications neuro. Guérissable.
        
        **3. Gonorrhée** - *Bactérie (Neisseria gonorrhoeae)* : Écoulements, douleurs, infertilité.
        
        **4. Chlamydia** - *Bactérie (Chlamydia trachomatis)* : Asymptomatique, stérilité.
        
        **5. HPV** - *Virus (Papillomavirus)* : Verrues, cancers. Vaccination préventive.
        
        **6. Herpès génital** - *Virus (HSV-1/HSV-2)* : Vésicules douloureuses, récurrences.
        
        **7. Hépatite B** - *Virus (VHB)* : Fatigue, jaunisse, cirrhose. Vaccination.
        """)
    
    with st.expander("🚨 Symptômes évocateurs"):
        st.markdown("""
        - Écoulements anormaux (urètre, vagin, anus)
        - Douleurs ou brûlures en urinant
        - Lésions, boutons, ulcères ou verrues
        - Démangeaisons intenses
        - Ganglions gonflés dans l'aine
        - Fièvre inexpliquée
        """)
        st.warning("⚠️ **Dépistage régulier indispensable (2x par an)**")
    
    with st.expander("🛡️ Moyens de prévention"):
        st.markdown("""
        - **Préservatifs masculins et féminins** (protection contre la plupart des IST)
        - **Dépistage régulier** (au moins 2 fois par an)
        - **Vaccination** (HPV, Hépatite B)
        - **Communication ouverte** avec le/la partenaire
        - **Réduction du nombre de partenaires**
        """)
    
    with st.expander("📍 Où se dépister ?"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🇨🇲 Cameroun :** Hôpital Général Yaoundé, Hôpital Laquintinie Douala")
            st.markdown("**🇸🇳 Sénégal :** Hôpital Fann Dakar, ALCS")
        with col2:
            st.markdown("**🇨🇮 Côte d'Ivoire :** INHP Abidjan")
            st.markdown("**🌍 Autres :** Hôpitaux publics, Croix-Rouge")

# ==================== FOOTER ====================
st.markdown("""
<div class="footer">
    📌 MADJOU FORTUNE NESLINE (24G2876) - INF232 EC2
</div>
""", unsafe_allow_html=True)