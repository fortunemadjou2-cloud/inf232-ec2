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

# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="NEXHEALTH SURVEY NO TABOO",
    page_icon="🫂🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== STYLE CSS ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;500;600;700;800&display=swap');
    .stApp, .main, .block-container { font-family: 'Manrope', sans-serif !important; }
    .nexhealth-banner {
        background: linear-gradient(90deg, #1b5e20 0%, #2e7d32 100%);
        border-radius: 24px;
        padding: 40px;
        margin-bottom: 20px;
        color: #ffffff !important;
    }
    .nexhealth-banner h1, .nexhealth-banner p { color: #ffffff !important; }
    .mode-selector {
        margin-bottom: 20px;
        padding: 15px;
        border-radius: 20px;
        text-align: center;
    }
    .mode-demo { background: linear-gradient(90deg, #e8f5e9, #c8e6c9); border: 2px solid #2e7d32; }
    .mode-normal { background: linear-gradient(90deg, #e3f2fd, #bbdef5); border: 2px solid #1565c0; }
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
    .stTabs [aria-selected="true"] { background-color: #2e7d32 !important; color: white !important; }
    .interpretation-box {
        background: #e8f5e9;
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        border-left: 5px solid #2e7d32;
        color: #1b5e20;
    }
    .nexhealth-footer {
        text-align: center;
        padding: 20px;
        margin-top: 48px;
        background: linear-gradient(90deg, #1b5e20, #2e7d32);
        color: white;
        border-radius: 16px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ==================== BANNIÈRE ====================
st.markdown("""
<div class="nexhealth-banner">
    <h1 style="margin:0; font-size:2rem;">🫂🩺 NEXHEALTH SURVEY NO TABOO</h1>
    <p style="margin:10px 0 0 0; font-style:italic;">✨ Parce qu'en santé, il n'y a pas de tabou. ✨</p>
    <p style="margin:5px 0 0 0; font-size:0.9rem;">🔍 Brisons le silence sur les IST, protégeons notre bien-être.</p>
    <p style="margin:5px 0 0 0; font-size:0.9rem;">⚖️ La santé sexuelle est un droit, la protection est une responsabilité.</p>
</div>
""", unsafe_allow_html=True)

# ==================== SESSION STATE ====================
if 'page' not in st.session_state:
    st.session_state.page = "ajouter"
if 'mode' not in st.session_state:
    st.session_state.mode = "demo"

# ==================== MODE SELECTOR ====================
mode_col1, mode_col2, mode_col3 = st.columns([1, 2, 1])
with mode_col2:
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔬 Mode Démo (30 exemples)", use_container_width=True,
                     type="primary" if st.session_state.mode == "demo" else "secondary"):
            st.session_state.mode = "demo"
            st.rerun()
    with col_b:
        if st.button("📝 Mode Normal (Données réelles)", use_container_width=True,
                     type="primary" if st.session_state.mode == "normal" else "secondary"):
            st.session_state.mode = "normal"
            st.rerun()

if st.session_state.mode == "demo":
    st.markdown("""
    <div class="mode-selector mode-demo" style="text-align:center; border-radius:15px; padding:10px; margin-bottom:20px;">
        <span style="font-size:1.2rem;">🔬 MODE DÉMO ACTIF</span><br>
        <span style="font-size:0.9rem;">Visualisation avec 30 exemples fictifs (aucune donnée sauvegardée)</span>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="mode-selector mode-normal" style="text-align:center; border-radius:15px; padding:10px; margin-bottom:20px;">
        <span style="font-size:1.2rem;">📝 MODE NORMAL ACTIF</span><br>
        <span style="font-size:0.9rem;">Collecte et analyse de données réelles (sauvegardées dans la base)</span>
    </div>
    """, unsafe_allow_html=True)

# ==================== BASE SQLITE ====================
DB_NAME = "nexhealth.db"

def init_database():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS participants (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT, age INTEGER, sexe TEXT, pays TEXT, profession TEXT,
            niveau_etude TEXT, partenaires_sexuels TEXT, utilisation_preservatifs TEXT,
            nb_partenaires TEXT, rapport_non_protege TEXT, alcool_substances TEXT,
            connaissance_ist TEXT, ist_connues TEXT, connaissance_asymptomatique TEXT,
            deja_depiste TEXT, savoir_depistage_gratuit TEXT, frequence_depistage TEXT,
            moyens_prevention TEXT, participation_campagnes TEXT,
            influence_reseaux_sociaux TEXT, ist_diagnostiquee TEXT, vaccin_hpv TEXT,
            consultation_medecin TEXT
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
            connaissance_ist, ist_connues, connaissance_asymptomatique, deja_depiste,
            savoir_depistage_gratuit, frequence_depistage, moyens_prevention,
            participation_campagnes, influence_reseaux_sociaux, ist_diagnostiquee,
            vaccin_hpv, consultation_medecin
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
    partenaires_list = ["0", "1", "2-5", "6-10", "11-20", "20+"]
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
            'nb_partenaires': np.random.choice(partenaires_list, p=[0.1,0.3,0.35,0.15,0.05,0.05]),
            'rapport_non_protege': np.random.choice(["Jamais", "Une fois", "Plusieurs fois"], p=[0.4,0.35,0.25]),
            'alcool_substances': np.random.choice(["Jamais", "Rarement", "Parfois", "Souvent"], p=[0.5,0.3,0.15,0.05]),
            'connaissance_ist': np.random.choice(connais_list, p=[0.1,0.2,0.3,0.25,0.15]),
            'ist_connues': "",
            'connaissance_asymptomatique': np.random.choice(["Oui", "Non", "Je ne sais pas"], p=[0.6,0.2,0.2]),
            'deja_depiste': np.random.choice(["Jamais", "Une fois", "Plusieurs fois", "Régulièrement"], p=[0.4,0.3,0.2,0.1]),
            'savoir_depistage_gratuit': np.random.choice(["Oui", "Non", "Je ne sais pas"], p=[0.5,0.25,0.25]),
            'frequence_depistage': np.random.choice(["Jamais", "1 fois par an", "2 fois par an"], p=[0.5,0.35,0.15]),
            'moyens_prevention': "",
            'participation_campagnes': np.random.choice(campagnes_list, p=[0.25,0.2,0.25,0.2,0.1]),
            'influence_reseaux_sociaux': np.random.choice(influence_list, p=[0.2,0.4,0.4]),
            'ist_diagnostiquee': np.random.choice(["Non", "Oui, guérie"], p=[0.85,0.15]),
            'vaccin_hpv': np.random.choice(["Non", "Oui", "Je ne sais pas"], p=[0.6,0.2,0.2]),
            'consultation_medecin': np.random.choice(["Jamais", "Rarement", "Parfois"], p=[0.5,0.3,0.2])
        })
    return pd.DataFrame(demo_data)

# ==================== CHARGEMENT DES DONNÉES ====================
if st.session_state.mode == "demo":
    df = get_demo_data()
else:
    df = charger_participants()

# ==================== MENU 4 CARRÉS ====================
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("✏️🫂\nAJOUTER\nPARTICIPANT", use_container_width=True, 
                 type="primary" if st.session_state.page == "ajouter" else "secondary"):
        st.session_state.page = "ajouter"
        st.rerun()

with col2:
    if st.button("📋👥\nPARTICIPANTS\nENREGISTRÉS", use_container_width=True,
                 type="primary" if st.session_state.page == "participants" else "secondary"):
        st.session_state.page = "participants"
        st.rerun()

with col3:
    if st.button("📈🔬\nANALYSES\nAVANCÉES", use_container_width=True,
                 type="primary" if st.session_state.page == "analyses" else "secondary"):
        st.session_state.page = "analyses"
        st.rerun()

with col4:
    if st.button("🛡️🩺\nESPACE\nPRÉVENTION IST", use_container_width=True,
                 type="primary" if st.session_state.page == "prevention" else "secondary"):
        st.session_state.page = "prevention"
        st.rerun()

st.markdown("---")

# ==================== PAGE 1 : AJOUTER PARTICIPANT ====================
if st.session_state.page == "ajouter":
    st.header("🫂✏️ Ajouter un nouveau participant")
    st.markdown("*Toutes vos réponses sont anonymes et seront conservées dans la base de données.*")
    
    if st.session_state.mode == "demo":
        st.warning("⚠️ Mode Démo actif : Les données ne seront pas sauvegardees. Passez en Mode Normal pour enregistrer.")
    
    with st.form("collecte", clear_on_submit=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**👤 Votre profil**")
            age = st.slider("Âge", 15, 95, 25)
            sexe = st.radio("Sexe", ["Homme", "Femme", "Autre", "Préfère ne pas répondre"])
            pays = st.selectbox("Pays", ["Sénégal", "Côte d'Ivoire", "Nigéria", "Kenya", "Ghana", "Maroc", "Cameroun", "Autre"])
            profession = st.selectbox("Profession", ["Étudiant", "Employé", "Indépendant", "Fonctionnaire", "Sans emploi", "Retraité", "Autre"])
            niveau_etude = st.selectbox("Niveau d'étude", ["Aucun", "Primaire", "Secondaire", "Universitaire", "Supérieur", "Autre"])
        
        with col2:
            st.markdown("**💕 Habitudes et comportements**")
            partenaires_sexuels = st.radio("Avez-vous déjà eu un ou plusieurs partenaires sexuels ?", ["Oui", "Non", "Préfère ne répondre"])
            utilisation_preservatifs = st.select_slider("Utilisez-vous systématiquement des préservatifs ?",
                options=["Jamais", "Rarement", "Parfois", "Souvent", "Systématiquement", "Pas concerné(e)"])
            nb_partenaires = st.selectbox("Combien de partenaires différents avez-vous eus (environ) ?",
                ["0", "1", "2-5", "6-10", "11-20", "20+", "Préfère ne répondre"])
            rapport_non_protege = st.selectbox("Avez-vous déjà eu un rapport non protégé ?", ["Jamais", "Une fois", "Plusieurs fois", "Préfère ne répondre"])
            alcool_substances = st.select_slider("Consommez-vous de l'alcool ou des substances avant des rapports ?",
                options=["Jamais", "Rarement", "Parfois", "Souvent", "Très souvent"])
        
        st.markdown("---")
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("**🏥 Connaissance et dépistage**")
            connaissance_ist = st.select_slider("Comment évaluez-vous votre connaissance des IST ?",
                options=["Très mauvaise", "Mauvaise", "Moyenne", "Bonne", "Très bonne"])
            ist_connues = st.multiselect("Quelles IST connaissez-vous ?",
                ["Sida/VIH", "Syphilis", "Gonorrhée", "Chlamydia", "HPV", "Herpès", "Hépatite B"])
            ist_autres = st.text_input("Autres IST que vous connaissez (facultatif)")
            connaissance_asympto = st.radio("Savez-vous que certaines IST peuvent être asymptomatiques ?",
                ["Oui", "Non", "Je ne sais pas"])
            deja_depiste = st.radio("Vous êtes-vous déjà fait dépister pour une IST ?",
                ["Jamais", "Une fois", "Plusieurs fois", "Régulièrement"])
            savoir_depistage_gratuit = st.radio("Savez-vous où vous faire dépister gratuitement ?", ["Oui", "Non", "Je ne sais pas"])
        
        with col4:
            st.markdown("**📊 Actions de prévention**")
            frequence_depistage = st.selectbox("À quelle fréquence vous faites-vous dépister (par an) ?",
                ["Jamais", "1 fois par an", "2 fois par an", "3 fois par an ou plus"])
            moyens_prevention = st.multiselect("Quels moyens de prévention utilisez-vous ?",
                ["Préservatifs", "Dépistage régulier", "Vaccination (HPV, Hépatite B)", "Abstinence", "Fidélité mutuelle", "Aucun"])
            prevention_autres = st.text_input("Autres moyens de prévention (facultatif)")
            participation_campagnes = st.select_slider("Participez-vous régulièrement aux campagnes de dépistage des IST ?",
                options=["Jamais", "Rarement", "Parfois", "Souvent", "Très souvent"])
            
            st.markdown("**📱 Réseaux sociaux**")
            influence_reseaux = st.select_slider("Pensez-vous que les réseaux sociaux influencent votre vie sexuelle ?",
                options=["Très négativement", "Négativement", "Neutre", "Positivement", "Très positivement"])
        
        st.markdown("---")
        col5, col6 = st.columns(2)
        
        with col5:
            st.markdown("**📊 Données médicales (optionnelles)**")
            ist_diagnostiquee = st.radio("Avez-vous déjà eu une IST diagnostiquée ?",
                ["Non", "Oui, guérie", "Oui, en traitement", "Préfère ne répondre"])
            vaccin_hpv = st.radio("Avez-vous été vacciné(e) contre le HPV (Papillomavirus) ?",
                ["Oui", "Non", "Je ne sais pas", "En cours"])
            consultation_medecin = st.select_slider("Consultez-vous un gynécologue/urologue régulièrement ?",
                options=["Jamais", "Rarement", "Parfois", "Régulièrement"])
        
        submit = st.form_submit_button("✅ Envoyer ma participation", use_container_width=True)
        
        if submit and st.session_state.mode == "normal":
            ist_connues_str = ", ".join(ist_connues)
            if ist_autres:
                ist_connues_str += f", {ist_autres}"
            prevention_str = ", ".join(moyens_prevention)
            if prevention_autres:
                prevention_str += f", {prevention_autres}"
            
            data = (
                datetime.now().strftime("%Y-%m-%d %H:%M"),
                age, sexe, pays, profession, niveau_etude,
                partenaires_sexuels, utilisation_preservatifs, nb_partenaires,
                rapport_non_protege, alcool_substances, connaissance_ist,
                ist_connues_str, connaissance_asympto, deja_depiste,
                savoir_depistage_gratuit, frequence_depistage, prevention_str,
                participation_campagnes, influence_reseaux, ist_diagnostiquee,
                vaccin_hpv, consultation_medecin
            )
            sauvegarder_participant(data)
            st.success("✅ Merci ! Votre reponse a ete enregistree.")
            st.balloons()
        elif submit and st.session_state.mode == "demo":
            st.info("ℹ️ Mode Demo actif : Les donnees ne sont pas sauvegardees.")

# ==================== PAGE 2 : PARTICIPANTS ====================
elif st.session_state.page == "participants":
    st.header("📋👥 Participants enregistres")
    
    if st.session_state.mode == "demo":
        st.info("📊 Mode Demo : Voici 30 exemples fictifs de participants")
        st.dataframe(df, use_container_width=True)
    else:
        if len(df) == 0:
            st.info("📭 Aucun participant. Utilisez 'AJOUTER PARTICIPANT'.")
        else:
            st.metric("Total participants", len(df))
            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False).encode('utf-8')
            col1, col2 = st.columns(2)
            with col1:
                st.download_button("📥 Export CSV", csv, "participants_ist.csv")
            with col2:
                if st.button("🗑️ Supprimer tout"):
                    supprimer_toutes_donnees()
                    st.rerun()

# ==================== PAGE 3 : ANALYSES ====================
elif st.session_state.page == "analyses":
    st.header("📈🔬 Analyses avancees des donnees")
    
    if st.session_state.mode == "demo":
        st.info("📊 Mode Demo : Analyses basees sur 30 exemples fictifs")
    
    if len(df) < 3:
        st.warning(f"⚠️ Besoin d'au moins 3 participants. Actuellement : {len(df)}")
    else:
        # Conversion des donnees
        df['Age'] = pd.to_numeric(df['age'], errors='coerce')
        df['Connaissance_num'] = df['connaissance_ist'].map({'Tres mauvaise':1,'Mauvaise':2,'Moyenne':3,'Bonne':4,'Tres bonne':5})
        df['Preservatifs_num'] = df['utilisation_preservatifs'].map({'Jamais':1,'Rarement':2,'Parfois':3,'Souvent':4,'Systematiquement':5})
        df['Partenaires_num'] = df['nb_partenaires'].map({'0':0,'1':1,'2-5':2,'6-10':3,'11-20':4,'20+':5})
        df_clean = df.dropna(subset=['Age', 'Connaissance_num', 'Preservatifs_num'])
        
        df_clean['Score_Risque'] = (6 - df_clean['Preservatifs_num']) * 2 + df_clean['Partenaires_num'] * 1.5
        df_clean['Categorie_Risque'] = df_clean['Score_Risque'].apply(lambda x: 'Faible' if x <= 8 else ('Modere' if x <= 15 else 'Eleve'))
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Regression simple", "Regression multiple", "PCA", "Classification", "Graphiques"])
        
        with tab1:
            if len(df_clean) >= 3:
                X = df_clean[['Age']].values
                y = df_clean['Connaissance_num'].values
                modele = LinearRegression().fit(X, y)
                fig = px.scatter(df_clean, x='Age', y='Connaissance_num', color='Categorie_Risque')
                x_range = np.linspace(df_clean['Age'].min(), df_clean['Age'].max(), 100)
                y_pred = modele.predict(x_range.reshape(-1, 1))
                fig.add_trace(go.Scatter(x=x_range, y=y_pred, mode='lines', name='Tendance', line=dict(color='#2e7d32')))
                st.plotly_chart(fig, use_container_width=True)
                st.metric("R²", f"{r2_score(y, modele.predict(X)):.3f}")
                st.info("Plus le R² est proche de 1, plus l'age explique les differences.")
        
        with tab2:
            if len(df_clean) >= 4:
                X = df_clean[['Age', 'Preservatifs_num', 'Partenaires_num']].values
                y = df_clean['Connaissance_num'].values
                modele = LinearRegression().fit(X, y)
                st.dataframe(pd.DataFrame({'Facteur':['Age','Preservatifs','Partenaires'],'Coefficient':modele.coef_}))
                predictions = modele.predict(X)
                fig = px.scatter(x=y, y=predictions)
                fig.add_trace(go.Scatter(x=[1,5], y=[1,5], mode='lines', name='Parfait', line=dict(dash='dash')))
                st.plotly_chart(fig, use_container_width=True)
                st.metric("R²", f"{r2_score(y, predictions):.3f}")
        
        with tab3:
            if len(df_clean) >= 4:
                features = ['Age', 'Connaissance_num', 'Preservatifs_num']
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(df_clean[features])
                pca = PCA(n_components=2)
                result = pca.fit_transform(X_scaled)
                fig = px.scatter(x=result[:,0], y=result[:,1], color=df_clean['Categorie_Risque'])
                st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            if len(df_clean) >= 5:
                df_clean['Cible'] = (df_clean['Categorie_Risque'] == 'Eleve').astype(int)
                X = df_clean[['Age', 'Preservatifs_num', 'Partenaires_num']].values
                y = df_clean['Cible'].values
                rf = RandomForestClassifier().fit(X, y)
                
                st.subheader("Testez votre risque")
                col1, col2 = st.columns(2)
                with col1:
                    age_t = st.slider("Age", 18, 65, 25, key="age_t")
                    p_t = st.select_slider("Preservatifs", options=["Systematiquement","Souvent","Parfois","Rarement","Jamais"], key="p_t")
                with col2:
                    k_t = st.select_slider("Partenaires", options=["1","2-5","6-10","11-20","20+"], key="k_t")
                
                if st.button("Estimer mon risque"):
                    p_map = {"Systematiquement":5,"Souvent":4,"Parfois":3,"Rarement":2,"Jamais":1}
                    k_map = {"1":1,"2-5":2,"6-10":3,"11-20":4,"20+":5}
                    pred = rf.predict([[age_t, p_map[p_t], k_map[k_t]]])[0]
                    if pred == 1:
                        st.error("Risque ELEVE - Consultez l'onglet Prevention")
                    else:
                        st.success("Risque FAIBLE a MODERE")
        
        with tab5:
            col1, col2 = st.columns(2)
            with col1:
                fig_hist = px.histogram(df_clean, x='Age', nbins=15, title="Distribution des ages")
                st.plotly_chart(fig_hist, use_container_width=True)
                connais_counts = df_clean['connaissance_ist'].value_counts().reset_index()
                connais_counts.columns = ['Niveau', 'Nombre']
                fig_bar = px.bar(connais_counts, x='Niveau', y='Nombre', title="Niveau de connaissance")
                st.plotly_chart(fig_bar, use_container_width=True)
            with col2:
                fig_pie = px.pie(df_clean, names='Categorie_Risque', title="Repartition par risque")
                st.plotly_chart(fig_pie, use_container_width=True)
                preserv_counts = df_clean['utilisation_preservatifs'].value_counts().reset_index()
                preserv_counts.columns = ['Frequence', 'Nombre']
                fig_preserv = px.bar(preserv_counts, x='Frequence', y='Nombre', title="Utilisation des preservatifs")
                st.plotly_chart(fig_preserv, use_container_width=True)

# ==================== PAGE 4 : PREVENTION ====================
elif st.session_state.page == "prevention":
    st.header("🛡️🩺 ESPACE PREVENTION IST")
    
    st.warning("Ces informations ne remplacent pas l'avis d'un medecin. Consultez un professionnel de sante.")
    
    with st.expander("Qu'est-ce qu'une IST ?", expanded=True):
        st.markdown("""
        Les IST (Infections Sexuellement Transmissibles) sont des infections qui se transmettent lors de rapports sexuels non proteges.
        
        **Modes de transmission :** rapports vaginaux, anaux, oraux, partage de seringues, mere-enfant.
        
        **Caracteristiques :**
        - Certaines sont asymptomatiques
        - Toutes peuvent avoir des consequences graves
        - La plupart sont evitables ou soignables
        """)
    
    with st.expander("Principales IST"):
        st.markdown("""
        **VIH :** Destruction immunitaire. Prevention : preservatifs, PrEP.
        **Syphilis :** Lesions, complications neurologiques. Guerissable.
        **Gonorrhee :** Ecoulements, douleurs. Risque d'infertilite.
        **Chlamydia :** Asymptomatique. Peut rendre sterile.
        **HPV :** Verrues, cancers. Vaccination preventive.
        **Herpes :** Vesicules douloureuses. Recurrences possibles.
        **Hepatite B :** Fatigue, jaunisse. Vaccination disponible.
        """)
    
    with st.expander("Symptomes evocateurs"):
        st.markdown("- Ecoulements anormaux\n- Douleurs en urinant\n- Lesions ou verrues\n- Demangeaisons\n- Ganglions gonfles")
        st.warning("Depistage regulier indispensable (2x par an)")
    
    with st.expander("Prevention"):
        st.markdown("- Preservatifs\n- Depistage regulier\n- Vaccination HPV\n- Communication avec le/la partenaire")
    
    with st.expander("Depistage"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Cameroun :** Hopital General Yaounde, Hopital Laquintinie Douala")
            st.markdown("**Senegal :** Hopital Fann Dakar, ALCS")
        with col2:
            st.markdown("**Cote d'Ivoire :** INHP Abidjan")
            st.markdown("**Autres :** Hopitaux publics, Croix-Rouge")

# ==================== FOOTER ====================
st.markdown("""
<div class="nexhealth-footer">
    MADJOU FORTUNE NESLINE (24G2876) - INF232 EC2
</div>
""", unsafe_allow_html=True)