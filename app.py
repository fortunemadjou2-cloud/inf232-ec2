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
import hashlib
import os

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
    /* Bannière */
    .banner {
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 50%, #4caf50 100%);
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        color: white;
    }
    .banner h1 {
        font-size: 2.5rem;
        font-weight: bold;
        font-style: italic;
        margin: 0;
    }
    .banner p {
        font-size: 1.1rem;
        font-weight: bold;
        font-style: italic;
        margin: 10px 0 0 0;
    }
    .banner-sub {
        font-size: 0.95rem;
        margin: 5px 0;
    }
    
    /* Cartes prévention */
    .risk-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
        border-left: 5px solid #2e7d32;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    .risk-card h4 {
        color: #1b5e20;
        margin-top: 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 20px;
        margin-top: 40px;
        background: linear-gradient(135deg, #1b5e20, #2e7d32);
        color: white;
        border-radius: 15px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==================== BANNIÈRE ====================
st.markdown("""
<div class="banner">
    <h1>🫂🩺 NEXHEALTH SURVEY NO TABOO</h1>
    <p>✨ Parce qu'en santé, il n'y a pas de tabou. ✨</p>
    <div class="banner-sub">🔍 Brisons le silence sur les IST, protégeons notre bien-être.</div>
    <div class="banner-sub">⚖️ La santé sexuelle est un droit, la protection est une responsabilité.</div>
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
            connaissance_ist TEXT, deja_depiste TEXT, participation_campagnes TEXT,
            influence_reseaux_sociaux TEXT, ist_diagnostiquee TEXT, vaccin_hpv TEXT
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

# ==================== SESSION STATE ====================
if 'page' not in st.session_state:
    st.session_state.page = "ajouter"

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
    if st.button("📚🩺\nPRÉVENTION\n& RISQUES", use_container_width=True,
                 type="primary" if st.session_state.page == "prevention" else "secondary"):
        st.session_state.page = "prevention"
        st.rerun()

st.markdown("---")

# ==================== PAGE AJOUTER ====================
if st.session_state.page == "ajouter":
    st.header("🫂✏️ Ajouter un nouveau participant")
    
    with st.form("collecte", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Âge", 15, 95, 25)
            sexe = st.radio("Sexe", ["Homme", "Femme", "Autre"])
            pays = st.selectbox("Pays", ["Cameroun", "Sénégal", "Côte d'Ivoire", "Nigéria", "Kenya", "Autre"])
            profession = st.selectbox("Profession", ["Étudiant", "Employé", "Indépendant", "Fonctionnaire", "Sans emploi"])
        with col2:
            utilisation_preservatifs = st.select_slider("Utilisation des préservatifs", 
                options=["Jamais", "Rarement", "Parfois", "Souvent", "Systématiquement"])
            nb_partenaires = st.selectbox("Nombre de partenaires", ["1", "2-5", "6-10", "11-20", "20+"])
            connaissance_ist = st.select_slider("Connaissance des IST",
                options=["Très mauvaise", "Mauvaise", "Moyenne", "Bonne", "Très bonne"])
            deja_depiste = st.radio("Déjà dépisté ?", ["Jamais", "Une fois", "Plusieurs fois", "Régulièrement"])
        
        if st.form_submit_button("✅ Enregistrer", use_container_width=True):
            data = (
                datetime.now().strftime("%Y-%m-%d %H:%M"), age, sexe, pays, profession, "Universitaire",
                "Oui", utilisation_preservatifs, nb_partenaires, "Jamais", "Parfois",
                connaissance_ist, deja_depiste, "Parfois", "Neutre", "Non", "Non"
            )
            sauvegarder_participant(data)
            st.success("✅ Participant enregistré !")
            st.balloons()

# ==================== PAGE PARTICIPANTS ====================
elif st.session_state.page == "participants":
    st.header("📋👥 Participants enregistrés")
    df = charger_participants()
    if len(df) == 0:
        st.info("📭 Aucun participant. Utilisez 'AJOUTER PARTICIPANT'.")
    else:
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Exporter CSV", csv, "participants.csv")

# ==================== PAGE ANALYSES ====================
elif st.session_state.page == "analyses":
    st.header("📈🔬 Analyses avancées")
    df = charger_participants()
    
    if len(df) < 3:
        st.warning(f"⚠️ Besoin d'au moins 3 participants. Actuellement : {len(df)}")
    else:
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
        df['Connaissance_num'] = df['connaissance_ist'].map(
            {'Très mauvaise':1,'Mauvaise':2,'Moyenne':3,'Bonne':4,'Très bonne':5})
        df['Preservatifs_num'] = df['utilisation_preservatifs'].map(
            {'Jamais':1,'Rarement':2,'Parfois':3,'Souvent':4,'Systématiquement':5})
        df['Nb_partenaires_num'] = df['nb_partenaires'].map(
            {'1':1,'2-5':2,'6-10':3,'11-20':4,'20+':5})
        
        df_clean = df.dropna()
        df_clean['Score_Risque'] = (6 - df_clean['Preservatifs_num']) * 2 + df_clean['Nb_partenaires_num'] * 1.5
        df_clean['Categorie_Risque'] = df_clean['Score_Risque'].apply(
            lambda x: 'Faible' if x <= 8 else ('Modéré' if x <= 15 else 'Élevé'))
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Régression simple", "🔬 Régression multiple", "🎯 PCA", "🏷️ Classification", "🔄 Clustering"
        ])
        
        with tab1:
            X = df_clean[['Age']].values
            y = df_clean['Connaissance_num'].values
            modele = LinearRegression().fit(X, y)
            fig = px.scatter(df_clean, x='Age', y='Connaissance_num', 
                             color='Categorie_Risque', title="Âge vs Connaissance")
            x_range = np.linspace(df_clean['Age'].min(), df_clean['Age'].max(), 100)
            y_pred = modele.predict(x_range.reshape(-1, 1))
            fig.add_trace(go.Scatter(x=x_range, y=y_pred, mode='lines', 
                                    name='Tendance', line=dict(color='#2e7d32')))
            st.plotly_chart(fig, use_container_width=True)
            st.metric("R²", f"{r2_score(y, modele.predict(X)):.3f}")
        
        with tab2:
            X = df_clean[['Age', 'Preservatifs_num', 'Nb_partenaires_num']].values
            y = df_clean['Connaissance_num'].values
            modele = LinearRegression().fit(X, y)
            st.dataframe(pd.DataFrame({'Facteur': ['Âge', 'Préservatifs', 'Partenaires'],
                                       'Coefficient': modele.coef_}))
            predictions = modele.predict(X)
            fig = px.scatter(x=y, y=predictions, title="Prédictions vs Réalité")
            fig.add_trace(go.Scatter(x=[1,5], y=[1,5], mode='lines', name='Parfait',
                                    line=dict(dash='dash', color='#2e7d32')))
            st.plotly_chart(fig, use_container_width=True)
            st.metric("R²", f"{r2_score(y, predictions):.3f}")
        
        with tab3:
            features = ['Age', 'Connaissance_num', 'Preservatifs_num', 'Nb_partenaires_num']
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_clean[features])
            pca = PCA(n_components=2)
            result = pca.fit_transform(X_scaled)
            df_viz = pd.DataFrame({'PC1': result[:,0], 'PC2': result[:,1], 
                                   'Risque': df_clean['Categorie_Risque']})
            fig = px.scatter(df_viz, x='PC1', y='PC2', color='Risque', title="PCA")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            df_clean['Cible'] = (df_clean['Categorie_Risque'] == 'Élevé').astype(int)
            X = df_clean[['Age', 'Preservatifs_num', 'Nb_partenaires_num', 'Connaissance_num']].values
            y = df_clean['Cible'].values
            rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=4).fit(X, y)
            importance = pd.DataFrame({'Facteur': ['Âge', 'Préservatifs', 'Partenaires', 'Connaissance'],
                                      'Importance %': (rf.feature_importances_ * 100).round(1)})
            st.dataframe(importance)
            
            st.subheader("🔮 Testez votre risque")
            col1, col2 = st.columns(2)
            with col1:
                age_t = st.slider("Âge", 18, 65, 25, key="risk_age")
                preserv_t = st.select_slider("Préservatifs", 
                    options=["Systématiquement", "Souvent", "Parfois", "Rarement", "Jamais"], key="risk_preserv")
            with col2:
                partenaires_t = st.select_slider("Partenaires", options=["1", "2-5", "6-10", "11-20", "20+"], key="risk_part")
                connais_t = st.select_slider("Connaissance", 
                    options=["Très bonne", "Bonne", "Moyenne", "Mauvaise", "Très mauvaise"], key="risk_connais")
            
            if st.button("Estimer mon risque"):
                p_map = {"Systématiquement":5, "Souvent":4, "Parfois":3, "Rarement":2, "Jamais":1}
                k_map = {"1":1, "2-5":2, "6-10":3, "11-20":4, "20+":5}
                c_map = {"Très bonne":5, "Bonne":4, "Moyenne":3, "Mauvaise":2, "Très mauvaise":1}
                pred = rf.predict([[age_t, p_map[preserv_t], k_map[partenaires_t], c_map[connais_t]]])[0]
                if pred == 1:
                    st.error("⚠️ Risque ÉLEVÉ - Consultez la rubrique Prévention")
                else:
                    st.success("✅ Risque FAIBLE à MODÉRÉ - Continuez les bonnes pratiques")
        
        with tab5:
            X = df_clean[['Age', 'Connaissance_num', 'Preservatifs_num']].values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            k = st.slider("Nombre de segments", 2, 4, 3)
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)
            fig = px.scatter(df_clean, x='Age', y='Connaissance_num', 
                             color=clusters.astype(str), title=f"Segmentation en {k} groupes",
                             size='Preservatifs_num')
            st.plotly_chart(fig, use_container_width=True)

# ==================== PAGE PRÉVENTION ====================
elif st.session_state.page == "prevention":
    st.header("📚🩺 Prévention, conséquences et risques")
    
    with st.expander("🏠 Conséquences sur la vie quotidienne", expanded=True):
        st.markdown("""
        - **Douleurs chroniques** : Douleurs abdominales, pelviennes ou lors des rapports sexuels
        - **Fatigue persistante** : Épuisement physique et mental récurrent
        - **Impact sur la vie sexuelle** : Baisse de libido, douleurs, peur de transmettre
        - **Stigmatisation sociale** : Isolement, jugement, difficultés relationnelles
        - **Anxiété et dépression** : Stress lié au diagnostic
        """)
    
    with st.expander("⏰ Conséquences à long terme", expanded=True):
        st.markdown("""
        - **Infertilité** (homme et femme)
        - **Grossesses extra-utérines**
        - **Cancers** : HPV responsable de cancers du col de l'utérus, de l'anus, du pénis
        - **Transmission mère-enfant**
        - **Complications neurologiques** (syphilis tardive)
        """)
    
    with st.expander("⚠️ Risques des IST non traitées", expanded=True):
        st.markdown("""
        - Aggravation des symptômes
        - Propagation à d'autres personnes
        - Développement de résistances aux traitements
        - Augmentation du risque VIH (jusqu'à 10 fois plus élevé)
        - Complications irréversibles (stérilité, lésions neurologiques)
        """)
    
    with st.expander("🚨 Symptômes évocateurs", expanded=True):
        st.markdown("""
        - Écoulements anormaux
        - Douleurs ou brûlures en urinant
        - Lésions, boutons, ulcères ou verrues
        - Démangeaisons intenses
        - Ganglions gonflés dans l'aine
        - Fièvre inexpliquée
        """)
        st.warning("⚠️ Certaines IST sont asymptomatiques → Dépistage régulier indispensable")
    
    with st.expander("🛡️ Moyens de prévention", expanded=True):
        st.markdown("""
        - **Préservatifs masculins et féminins**
        - **Dépistage régulier** (2x par an si vie sexuelle active)
        - **Vaccination** (HPV, Hépatite B)
        - **Communication** avec le/la partenaire
        - **Réduction du nombre de partenaires**
        """)
    
    with st.expander("📍 Où se faire dépister ?", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🇨🇲 Cameroun** : Hôpital Général Yaoundé, Hôpital Laquintinie Douala")
            st.markdown("**🇸🇳 Sénégal** : Hôpital Fann Dakar, ALCS")
        with col2:
            st.markdown("**🇨🇮 Côte d'Ivoire** : INHP Abidjan, Centre Treichville")
            st.markdown("**🌍 Autres** : Hôpitaux publics, Croix-Rouge")

# ==================== FOOTER ====================
st.markdown("""
<div class="footer">
    📌 MADJOU FORTUNE NESLINE (24G2876) - INF232 EC2
</div>
""", unsafe_allow_html=True)