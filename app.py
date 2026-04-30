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
        background: linear-gradient(90deg, #1b5e20, #2e7d32);
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
    
    .square-btn {
        background: white;
        border-radius: 20px;
        padding: 30px 10px;
        text-align: center;
        transition: all 0.3s ease;
        border: 1px solid #e0e4da;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        margin: 10px;
    }
    .square-btn:hover { transform: translateY(-5px); box-shadow: 0 15px 30px rgba(27,94,32,0.15); border-color: #2e7d32; }
    
    .footer {
        text-align: center;
        padding: 20px;
        margin-top: 40px;
        background: linear-gradient(90deg, #1b5e20, #2e7d32);
        color: white;
        border-radius: 20px;
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
        df = charger_participants()
        return df

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
                'Date': datetime.now().strftime("%Y-%m-%d %H:%M"),
                'Age': age_sb, 'Sexe': sexe_sb, 'Pays': pays_sb, 'Profession': profession_sb,
                'Niveau_Etude': "Universitaire", 'Partenaires_Sexuels': "Oui",
                'Utilisation_Preservatifs': preservatifs_sb, 'Nb_Partenaires': nb_partenaires_sb,
                'Rapport_Non_Protege': rapport_sb, 'Alcool_Substances': "Parfois",
                'Connaissance_IST': connais_sb, 'Deja_Depiste': depist_sb,
                'Participation_Campagnes': camp_sb, 'Influence_Reseaux_Sociaux': influ_sb,
                'IST_Diagnostiquee': "Non renseigné", 'Vaccin_HPV': "Non"
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
    df['Influence_num'] = df['Influence_Reseaux_Sociaux'].map({'Négativement':1,'Neutre':2,'Positivement':3})
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
        
        # Regression simple
        with tabs[0]:
            if len(df_clean) >= 3:
                X = df_clean[['Age']].values
                y = df_clean['Connaissance_num'].values
                model = LinearRegression().fit(X, y)
                fig = px.scatter(df_clean, x='Age', y='Connaissance_num', color='Categorie_Risque')
                x_range = np.linspace(df_clean['Age'].min(), df_clean['Age'].max(), 100)
                fig.add_trace(go.Scatter(x=x_range, y=model.predict(x_range.reshape(-1,1)), mode='lines', name='Tendance', line=dict(color='red')))
                st.plotly_chart(fig, use_container_width=True)
                st.metric("R²", f"{r2_score(y, model.predict(X)):.3f}")
        
        # Regression multiple
        with tabs[1]:
            if len(df_clean) >= 4:
                X = df_clean[['Age', 'Preservatifs_num', 'Partenaires_num']].values
                y = df_clean['Connaissance_num'].values
                model = LinearRegression().fit(X, y)
                st.dataframe(pd.DataFrame({'Facteur':['Age','Preservatifs','Partenaires'],'Coefficient':model.coef_}))
                pred = model.predict(X)
                fig = px.scatter(x=y, y=pred)
                fig.add_trace(go.Scatter(x=[1,5], y=[1,5], mode='lines', name='Parfait', line=dict(dash='dash')))
                st.plotly_chart(fig, use_container_width=True)
                st.metric("R²", f"{r2_score(y, pred):.3f}")
        
        # PCA
        with tabs[2]:
            if len(df_clean) >= 4:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(df_clean[['Age', 'Connaissance_num', 'Preservatifs_num']])
                pca = PCA(n_components=2)
                result = pca.fit_transform(X_scaled)
                fig = px.scatter(x=result[:,0], y=result[:,1], color=df_clean['Categorie_Risque'])
                st.plotly_chart(fig, use_container_width=True)
        
        # Classification
        with tabs[3]:
            if len(df_clean) >= 5:
                df_clean['Cible'] = (df_clean['Categorie_Risque'] == 'Élevé').astype(int)
                X = df_clean[['Age', 'Preservatifs_num', 'Partenaires_num']].values
                y = df_clean['Cible'].values
                rf = RandomForestClassifier().fit(X, y)
                
                st.subheader("Testez votre risque")
                col1, col2 = st.columns(2)
                with col1:
                    age_t = st.slider("Age", 18, 65, 25, key="t_age")
                    p_t = st.select_slider("Preservatifs", options=["Systématiquement","Souvent","Parfois","Rarement","Jamais"], key="t_preserv")
                with col2:
                    k_t = st.select_slider("Partenaires", options=["1","2-5","6-10","11-20","20+"], key="t_part")
                
                if st.button("Estimer mon risque", key="btn_risk"):
                    p_map = {"Systématiquement":5,"Souvent":4,"Parfois":3,"Rarement":2,"Jamais":1}
                    k_map = {"1":1,"2-5":2,"6-10":3,"11-20":4,"20+":5}
                    pred = rf.predict([[age_t, p_map[p_t], k_map[k_t]]])[0]
                    if pred == 1:
                        st.error("⚠️ Risque ÉLEVÉ")
                    else:
                        st.success("✅ Risque FAIBLE à MODÉRÉ")
        
        # Clustering
        with tabs[4]:
            if len(df_clean) >= 5:
                X = df_clean[['Age', 'Connaissance_num', 'Preservatifs_num']].values
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                k = st.slider("Nombre de clusters", 2, 4, 3)
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(X_scaled)
                fig = px.scatter(df_clean, x='Age', y='Connaissance_num', color=clusters.astype(str), size='Preservatifs_num')
                st.plotly_chart(fig, use_container_width=True)
        
        # Graphiques
        with tabs[5]:
            col1, col2 = st.columns(2)
            with col1:
                fig_hist = px.histogram(df_clean, x='Age', nbins=15, title="Distribution des âges")
                st.plotly_chart(fig_hist, use_container_width=True)
                fig_bar = px.bar(df_clean['Connaissance_IST'].value_counts().reset_index(), x='index', y='Connaissance_IST', title="Niveau de connaissance")
                st.plotly_chart(fig_bar, use_container_width=True)
            with col2:
                fig_pie = px.pie(df_clean, names='Categorie_Risque', title="Risque IST")
                st.plotly_chart(fig_pie, use_container_width=True)
                fig_preserv = px.bar(df_clean['Utilisation_Preservatifs'].value_counts().reset_index(), x='index', y='Utilisation_Preservatifs', title="Préservatifs")
                st.plotly_chart(fig_preserv, use_container_width=True)

# ==================== PAGE 4 : PRÉVENTION ====================
elif st.session_state.page == "prevention":
    st.header("📚🩺 ESPACE PRÉVENTION IST")
    
    st.warning("⚠️ Ces informations ne remplacent pas l'avis d'un médecin.")
    
    with st.expander("📖 Définition et modes de contraction"):
        st.markdown("""
        **IST (Infections Sexuellement Transmissibles)** : Infections transmises lors de rapports sexuels non protégés.
        
        **Modes de contraction :**
        - Rapports vaginaux, anaux, oraux non protégés
        - Partage de seringues contaminées
        - Transmission mère-enfant (grossesse, accouchement, allaitement)
        - Contact direct avec des lésions
        """)
    
    with st.expander("🦠 Principales IST (7 exemples)"):
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
        st.markdown("- Écoulements anormaux\n- Douleurs en urinant\n- Lésions ou verrues\n- Démangeaisons\n- Ganglions gonflés")
        st.warning("⚠️ Dépistage régulier indispensable (2x par an)")
    
    with st.expander("🛡️ Moyens de prévention"):
        st.markdown("- Préservatifs (masculins et féminins)\n- Dépistage régulier\n- Vaccination (HPV, Hépatite B)\n- Communication avec le partenaire")
    
    with st.expander("📍 Où se dépister ?"):
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