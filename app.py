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
from sklearn.metrics import r2_score, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

# ==================== CONFIGURATION ====================
st.set_page_config(page_title="NEXHEALTH SURVEY NO TABOO", page_icon="🩺", layout="wide")

# ==================== CSS DESIGN ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;500;600;700;800&display=swap');
    * { font-family: 'Manrope', sans-serif !important; }
    
    .stApp { background: linear-gradient(135deg, #f0f7f0 0%, #e8f5e9 100%); }
    
    h1, h2, h3, h4, h5, h6, p, span, label, .stMarkdown { color: #1b5e20 !important; }
    
    @media (prefers-color-scheme: dark) {
        .stApp { background: linear-gradient(135deg, #0a2a0a 0%, #0d3b0d 100%) !important; }
        h1, h2, h3, h4, h5, h6, p, span, label, .stMarkdown { color: #c8e6c9 !important; }
        .stTabs [data-baseweb="tab-list"] { background-color: #1b5e20 !important; }
        .stTabs [data-baseweb="tab"] { color: #e8f5e9 !important; }
        .stAlert { background-color: #1b5e20 !important; color: #e8f5e9 !important; }
    }
    
    .mode-indicator {
        background: linear-gradient(90deg, #1b5e20, #2e7d32);
        padding: 15px;
        border-radius: 30px;
        text-align: center;
        margin: 15px 0 25px 0;
        color: white !important;
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
    .footer p { color: white !important; }
    
    .interpretation-box {
        background: #e8f5e9;
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        border-left: 5px solid #2e7d32;
    }
    @media (prefers-color-scheme: dark) {
        .interpretation-box { background: #1b5e20; border-left: 5px solid #4caf50; }
        .interpretation-box p, .interpretation-box li { color: #e8f5e9 !important; }
    }
</style>
""", unsafe_allow_html=True)

# ==================== TITRE ====================
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
    st.markdown("### 👤 **Mon profil**")
    age_sb = st.slider("Âge", 15, 95, 25, key="sb_age")
    sexe_sb = st.radio("Sexe", ["Homme", "Femme", "Autre"], key="sb_sexe")
    pays_sb = st.selectbox("Pays", ["Cameroun", "Sénégal", "Côte d'Ivoire", "Nigéria", "Kenya", "Autre"], key="sb_pays")
    profession_sb = st.selectbox("Profession", ["Étudiant", "Employé", "Indépendant", "Fonctionnaire", "Sans emploi"], key="sb_prof")
    
    st.markdown("---")
    st.markdown("### 💕 **Habitudes**")
    preservatifs_sb = st.select_slider("Utilisation des préservatifs", options=["Jamais", "Rarement", "Parfois", "Souvent", "Systématiquement"], key="sb_preserv")
    nb_partenaires_sb = st.selectbox("Nombre de partenaires", ["1", "2-5", "6-10", "11-20", "20+"], key="sb_nb")
    rapport_sb = st.selectbox("Rapport non protégé ?", ["Jamais", "Une fois", "Plusieurs fois"], key="sb_rapport")
    
    st.markdown("---")
    st.markdown("### 🏥 **Connaissance**")
    connais_sb = st.select_slider("Connaissance des IST", options=["Très mauvaise", "Mauvaise", "Moyenne", "Bonne", "Très bonne"], key="sb_connais")
    depist_sb = st.radio("Déjà dépisté ?", ["Jamais", "Une fois", "Plusieurs fois", "Régulièrement"], key="sb_depist")
    camp_sb = st.select_slider("Participation aux campagnes", options=["Jamais", "Rarement", "Parfois", "Souvent", "Très souvent"], key="sb_camp")
    
    st.markdown("---")
    st.markdown("### 📱 **Réseaux sociaux**")
    influ_sb = st.select_slider("Influence des réseaux sociaux", options=["Négativement", "Neutre", "Positivement"], key="sb_influ")
    
    if st.button("✅ Envoyer ma participation", use_container_width=True, key="sb_submit"):
        if st.session_state.mode == "normal":
            nouvelle = pd.DataFrame([{
                'Date': datetime.now().strftime("%Y-%m-%d %H:%M"), 'Age': age_sb, 'Sexe': sexe_sb,
                'Pays': pays_sb, 'Profession': profession_sb, 'Niveau_Etude': "Universitaire",
                'Partenaires_Sexuels': "Oui", 'Utilisation_Preservatifs': preservatifs_sb,
                'Nb_Partenaires': nb_partenaires_sb, 'Rapport_Non_Protege': rapport_sb,
                'Alcool_Substances': "Parfois", 'Connaissance_IST': connais_sb,
                'Deja_Depiste': depist_sb, 'Participation_Campagnes': camp_sb,
                'Influence_Reseaux_Sociaux': influ_sb, 'IST_Diagnostiquee': "Non renseigné", 'Vaccin_HPV': "Non"
            }])
            data_tuple = tuple(nouvelle.iloc[0].values)
            sauvegarder_participant(data_tuple)
            st.success("✅ Merci ! Votre réponse est enregistrée.")
            st.balloons()
        else:
            st.info("ℹ️ Mode Démo actif : Les données ne sont pas sauvegardées.")

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

# ==================== CHARGEMENT DONNÉES ====================
df = get_current_data()

if len(df) > 0:
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df['Connaissance_num'] = df['Connaissance_IST'].map({'Très mauvaise':1,'Mauvaise':2,'Moyenne':3,'Bonne':4,'Très bonne':5})
    df['Preservatifs_num'] = df['Utilisation_Preservatifs'].map({'Jamais':1,'Rarement':2,'Parfois':3,'Souvent':4,'Systématiquement':5})
    df['Campagnes_num'] = df['Participation_Campagnes'].map({'Jamais':1,'Rarement':2,'Parfois':3,'Souvent':4,'Très souvent':5})
    df['Partenaires_num'] = df['Nb_Partenaires'].map({'1':1,'2-5':2,'6-10':3,'11-20':4,'20+':5})
    df['Rapport_num'] = df['Rapport_Non_Protege'].map({'Jamais':0, 'Une fois':1, 'Plusieurs fois':2})
    
    df['Score_Risque'] = (6 - df['Preservatifs_num']) * 2 + df['Partenaires_num'] * 1.5 + df['Rapport_num'] * 2 + (df['Connaissance_num'] < 3).astype(int) * 2
    df['Categorie_Risque'] = df['Score_Risque'].apply(lambda x: 'Faible' if x <= 8 else ('Modéré' if x <= 15 else 'Élevé'))
    df_clean = df.dropna(subset=['Age', 'Connaissance_num', 'Preservatifs_num', 'Campagnes_num', 'Partenaires_num'])
else:
    df_clean = pd.DataFrame()

# ==================== PAGE 1 : AJOUTER ====================
if st.session_state.page == "ajouter":
    st.header("✏️🫂 Ajouter une nouvelle personne")
    
    if st.session_state.mode == "demo":
        st.warning("⚠️ Mode Démo actif : Les données ne seront pas sauvegardées.")
    else:
        st.info("📝 Mode Normal : Vos réponses seront sauvegardées.")
    
    with st.form("form_ajout", clear_on_submit=False):
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
        
        submit_f = st.form_submit_button("✅ Envoyer", use_container_width=True)
        if submit_f and st.session_state.mode == "normal":
            nouvelle = pd.DataFrame([{
                'Date': datetime.now().strftime("%Y-%m-%d %H:%M"), 'Age': age_f, 'Sexe': sexe_f,
                'Pays': pays_f, 'Profession': profession_f, 'Niveau_Etude': "Universitaire",
                'Partenaires_Sexuels': "Oui", 'Utilisation_Preservatifs': preservatifs_f,
                'Nb_Partenaires': nb_partenaires_f, 'Rapport_Non_Protege': rapport_f,
                'Alcool_Substances': "Parfois", 'Connaissance_IST': connais_f,
                'Deja_Depiste': depist_f, 'Participation_Campagnes': camp_f,
                'Influence_Reseaux_Sociaux': influ_f, 'IST_Diagnostiquee': "Non renseigné", 'Vaccin_HPV': "Non"
            }])
            data_tuple = tuple(nouvelle.iloc[0].values)
            sauvegarder_participant(data_tuple)
            st.success("✅ Merci ! Votre réponse est enregistrée.")
            st.balloons()
            st.rerun()
        elif submit_f and st.session_state.mode == "demo":
            st.info("ℹ️ Mode Démo actif : Les données ne sont pas sauvegardées.")

# ==================== PAGE 2 : ENREGISTREMENTS ====================
elif st.session_state.page == "participants":
    st.header("📋👥 Participants enregistrés")
    
    if st.session_state.mode == "demo":
        st.info("📊 Mode Démo : 30 exemples fictifs")
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Télécharger CSV", csv, "donnees_demo.csv")
    else:
        if len(df) == 0:
            st.info("📭 Aucun participant. Utilisez le formulaire pour ajouter.")
        else:
            st.metric("Total participants", len(df))
            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False).encode('utf-8')
            col1, col2 = st.columns(2)
            with col1:
                st.download_button("📥 Exporter CSV", csv, "donnees_reelles.csv")
            with col2:
                if st.button("🗑️ Supprimer toutes les données"):
                    supprimer_toutes_donnees()
                    st.success("Données supprimées !")
                    st.rerun()

# ==================== PAGE 3 : ANALYSES ====================
elif st.session_state.page == "analyses":
    st.header("📈🔬 Analyses avancées")
    
    if len(df_clean) < 3:
        st.warning(f"⚠️ Besoin d'au moins 3 participants. Actuellement : {len(df_clean)}")
    else:
        tabs = st.tabs(["📈 Régression simple", "🔬 Régression multiple", "🎯 PCA", "🏷️ Classification", "🔄 Clustering", "📊 Graphiques"])
        
        # ========== REGRESSION SIMPLE ==========
        with tabs[0]:
            if len(df_clean) >= 3:
                X = df_clean[['Age']].values
                y = df_clean['Connaissance_num'].values
                model = LinearRegression().fit(X, y)
                fig = px.scatter(df_clean, x='Age', y='Connaissance_num', color='Categorie_Risque',
                                 title="Relation entre l'âge et la connaissance des IST")
                x_range = np.linspace(df_clean['Age'].min(), df_clean['Age'].max(), 100)
                fig.add_trace(go.Scatter(x=x_range, y=model.predict(x_range.reshape(-1,1)), 
                                        mode='lines', name='Tendance', line=dict(color='red', width=3)))
                st.plotly_chart(fig, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Coefficient directeur", f"{model.coef_[0]:.3f}")
                with col2:
                    r2 = r2_score(y, model.predict(X))
                    st.metric("R² (qualité)", f"{r2:.3f}")
                
                st.markdown("""
                <div class="interpretation-box">
                <b>📖 INTERPRÉTATION DE LA RÉGRESSION SIMPLE</b><br><br>
                - Chaque point coloré représente un participant.<br>
                - La ligne rouge montre la tendance générale.<br>
                - Les couleurs indiquent le niveau de risque.<br>
                - R² proche de 1 = bonne prédiction.
                </div>
                """, unsafe_allow_html=True)
        
        # ========== REGRESSION MULTIPLE ==========
        with tabs[1]:
            if len(df_clean) >= 4:
                X = df_clean[['Age', 'Preservatifs_num', 'Partenaires_num']].values
                y = df_clean['Connaissance_num'].values
                model = LinearRegression().fit(X, y)
                
                coef_df = pd.DataFrame({
                    'Facteur': ['Âge', 'Utilisation des préservatifs', 'Nombre de partenaires'],
                    'Coefficient': [model.coef_[0], model.coef_[1], model.coef_[2]]
                })
                st.dataframe(coef_df, use_container_width=True)
                
                pred = model.predict(X)
                fig = px.scatter(x=y, y=pred, color=df_clean['Categorie_Risque'],
                                 title="Prédictions du modèle vs Valeurs réelles")
                fig.add_trace(go.Scatter(x=[1,5], y=[1,5], mode='lines', 
                                        name='Prédiction parfaite', line=dict(dash='dash', color='red')))
                st.plotly_chart(fig, use_container_width=True)
                st.metric("R² du modèle", f"{r2_score(y, pred):.3f}")
                
                st.markdown("""
                <div class="interpretation-box">
                <b>📖 INTERPRÉTATION DE LA RÉGRESSION MULTIPLE</b><br><br>
                - Les coefficients indiquent l'importance de chaque facteur.<br>
                - Un coefficient positif = plus le facteur augmente, meilleure est la connaissance.<br>
                - Les points proches de la diagonale = bonne prédiction.
                </div>
                """, unsafe_allow_html=True)
        
        # ========== PCA ==========
        with tabs[2]:
            if len(df_clean) >= 4:
                features = ['Age', 'Connaissance_num', 'Preservatifs_num']
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(df_clean[features])
                pca = PCA(n_components=2)
                result = pca.fit_transform(X_scaled)
                
                df_viz = pd.DataFrame({
                    'PC1': result[:,0],
                    'PC2': result[:,1],
                    'Risque': df_clean['Categorie_Risque'],
                    'Âge': df_clean['Age']
                })
                
                fig = px.scatter(df_viz, x='PC1', y='PC2', color='Risque', size='Âge',
                                 title="Projection PCA - Visualisation des profils")
                fig.update_layout(
                    xaxis_title=f"Composante 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
                    yaxis_title=f"Composante 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown(f"""
                <div class="interpretation-box">
                <b>📖 INTERPRÉTATION DE L'ACP</b><br><br>
                - Composante 1 : {pca.explained_variance_ratio_[0]*100:.1f}% des informations<br>
                - Composante 2 : {pca.explained_variance_ratio_[1]*100:.1f}% des informations<br>
                - Les points proches = profils similaires.
                </div>
                """, unsafe_allow_html=True)
        
        # ========== CLASSIFICATION ==========
        with tabs[3]:
            if len(df_clean) >= 5:
                df_clean['Cible'] = (df_clean['Categorie_Risque'] == 'Élevé').astype(int)
                X = df_clean[['Age', 'Preservatifs_num', 'Partenaires_num']].values
                y = df_clean['Cible'].values
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                rf.fit(X_train, y_train)
                y_pred = rf.predict(X_test)
                
                importance_df = pd.DataFrame({
                    'Facteur': ['Âge', 'Préservatifs', 'Partenaires'],
                    'Importance (%)': (rf.feature_importances_ * 100).round(1)
                }).sort_values('Importance (%)', ascending=False)
                st.dataframe(importance_df, use_container_width=True)
                
                fig_imp = px.bar(importance_df, x='Importance (%)', y='Facteur', orientation='h',
                                 title="Facteurs influençant le risque")
                st.plotly_chart(fig_imp, use_container_width=True)
                
                cm = confusion_matrix(y_test, y_pred)
                fig_cm = px.imshow(cm, text_auto=True, aspect="auto",
                                   labels=dict(x="Prédiction", y="Réalité"),
                                   x=['Risque faible', 'Risque élevé'],
                                   y=['Risque faible', 'Risque élevé'],
                                   title="Matrice de confusion")
                st.plotly_chart(fig_cm, use_container_width=True)
                st.metric("Précision du modèle", f"{accuracy_score(y_test, y_pred):.1%}")
                
                st.subheader("🔮 Testez votre risque")
                col1, col2 = st.columns(2)
                with col1:
                    age_t = st.slider("Âge", 18, 65, 25, key="t_age")
                    p_t = st.select_slider("Préservatifs", options=["Systématiquement","Souvent","Parfois","Rarement","Jamais"], key="t_preserv")
                with col2:
                    k_t = st.select_slider("Partenaires", options=["1","2-5","6-10","11-20","20+"], key="t_part")
                
                if st.button("🔮 Estimer mon risque", key="btn_risk"):
                    p_map = {"Systématiquement":5,"Souvent":4,"Parfois":3,"Rarement":2,"Jamais":1}
                    k_map = {"1":1,"2-5":2,"6-10":3,"11-20":4,"20+":5}
                    pred_risk = rf.predict([[age_t, p_map[p_t], k_map[k_t]]])[0]
                    if pred_risk == 1:
                        st.error("⚠️ Risque ÉLEVÉ - Consultez l'onglet Prévention")
                    else:
                        st.success("✅ Risque FAIBLE à MODÉRÉ - Continuez vos bonnes pratiques")
        
        # ========== CLUSTERING ==========
        with tabs[4]:
            if len(df_clean) >= 5:
                X = df_clean[['Age', 'Connaissance_num', 'Preservatifs_num']].values
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                k = st.slider("Nombre de segments (clusters)", 2, 4, 3)
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(X_scaled)
                df_clean['Segment'] = clusters
                
                fig = px.scatter(df_clean, x='Age', y='Connaissance_num', 
                                 color=clusters.astype(str), size='Preservatifs_num',
                                 title=f"Segmentation des participants en {k} groupes")
                st.plotly_chart(fig, use_container_width=True)
                
                profil = df_clean.groupby('Segment')[['Age', 'Connaissance_num', 'Preservatifs_num']].mean().round(1)
                profil.columns = ['Âge moyen', 'Connaissance (1-5)', 'Préservatifs (1-5)']
                st.dataframe(profil)
                
                st.markdown("""
                <div class="interpretation-box">
                <b>📖 INTERPRÉTATION DU CLUSTERING</b><br><br>
                - Chaque couleur = un groupe aux habitudes similaires.<br>
                - Le tableau montre le profil moyen de chaque groupe.<br>
                - Identifiez à quel groupe vous ressemblez.
                </div>
                """, unsafe_allow_html=True)
        
        # ========== GRAPHIQUES ==========
        with tabs[5]:
            col1, col2 = st.columns(2)
            with col1:
                fig_hist = px.histogram(df_clean, x='Age', nbins=15, title="Distribution des âges")
                st.plotly_chart(fig_hist, use_container_width=True)
                connais_counts = df_clean['Connaissance_IST'].value_counts().reset_index()
                connais_counts.columns = ['Niveau', 'Nombre']
                fig_bar = px.bar(connais_counts, x='Niveau', y='Nombre', title="Niveau de connaissance")
                st.plotly_chart(fig_bar, use_container_width=True)
            with col2:
                fig_pie = px.pie(df_clean, names='Categorie_Risque', title="Répartition par niveau de risque")
                st.plotly_chart(fig_pie, use_container_width=True)
                preserv_counts = df_clean['Utilisation_Preservatifs'].value_counts().reset_index()
                preserv_counts.columns = ['Fréquence', 'Nombre']
                fig_preserv = px.bar(preserv_counts, x='Fréquence', y='Nombre', title="Utilisation des préservatifs")
                st.plotly_chart(fig_preserv, use_container_width=True)

# ==================== PAGE 4 : PREVENTION ====================
elif st.session_state.page == "prevention":
    st.header("📚🩺 ESPACE PRÉVENTION IST")
    
    st.warning("⚠️ **Ces informations ne remplacent pas l'avis d'un médecin. Consultez un professionnel de santé.**")
    
    with st.expander("📖 Qu'est-ce qu'une IST ?", expanded=False):
        st.markdown("""
        **Définition :** Les IST (Infections Sexuellement Transmissibles) sont des infections transmises lors de rapports sexuels non protégés.
        
        **Modes de contamination :**
        - Rapports vaginaux, anaux, oraux non protégés
        - Contact direct avec des lésions
        - Partage de seringues
        - Transmission mère-enfant
        """)
    
    with st.expander("🦠 Principales IST (7 exemples)", expanded=False):
        st.markdown("""
        **1. VIH/Sida** - *Virus* : Destruction immunitaire. Prévention : préservatifs.
        
        **2. Syphilis** - *Bactérie* : Lésions, complications neuro. Guérissable.
        
        **3. Gonorrhée** - *Bactérie* : Écoulements, douleurs. Risque d'infertilité.
        
        **4. Chlamydia** - *Bactérie* : Asymptomatique. Peut rendre stérile.
        
        **5. HPV** - *Virus* : Verrues, cancers. Vaccination préventive.
        
        **6. Herpès** - *Virus* : Vésicules douloureuses. Récurrences.
        
        **7. Hépatite B** - *Virus* : Fatigue, jaunisse, cirrhose. Vaccination.
        """)
    
    with st.expander("🚨 Symptômes évocateurs", expanded=False):
        st.markdown("""
        - Écoulements anormaux
        - Douleurs en urinant
        - Lésions ou verrues
        - Démangeaisons
        - Ganglions gonflés
        """)
        st.warning("⚠️ Dépistage régulier indispensable (2x par an)")
    
    with st.expander("🛡️ Prévention", expanded=False):
        st.markdown("""
        - Préservatifs (masculins et féminins)
        - Dépistage régulier
        - Vaccination HPV
        - Communication avec le partenaire
        """)
    
    with st.expander("📍 Dépistage", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Cameroun :** Hôpital Général Yaoundé, Hôpital Laquintinie Douala")
            st.markdown("**Sénégal :** Hôpital Fann Dakar, ALCS")
        with col2:
            st.markdown("**Côte d'Ivoire :** INHP Abidjan")
            st.markdown("**Autres :** Hôpitaux publics, Croix-Rouge")

# ==================== FOOTER ====================
st.markdown("""
<div class="footer">
    📌 MADJOU FORTUNE NESLINE (24G2876) - INF232 EC2
</div>
""", unsafe_allow_html=True)