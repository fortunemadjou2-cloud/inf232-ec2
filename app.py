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

# ==================== STYLE CSS MODERNE ====================
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
        font-size: 2.8rem;
        font-weight: bold;
        font-style: italic;
        margin: 0;
        letter-spacing: 2px;
    }
    .banner p {
        font-size: 1.2rem;
        font-weight: bold;
        font-style: italic;
        margin: 10px 0 0 0;
        opacity: 0.95;
    }
    .banner-sub {
        font-size: 1rem;
        margin: 5px 0;
        opacity: 0.9;
    }
    
    /* Carrés menu */
    .menu-card {
        background: white;
        padding: 25px 15px;
        border-radius: 20px;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
        margin-bottom: 20px;
    }
    .menu-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(0,0,0,0.15);
        border-color: #2e7d32;
    }
    .menu-card-selected {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border: 2px solid #2e7d32;
        box-shadow: 0 10px 25px rgba(46,125,50,0.2);
    }
    .menu-icon {
        font-size: 3rem;
        margin-bottom: 10px;
    }
    .menu-title {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1b5e20;
        margin: 0;
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

# ==================== INITIALISATION DE LA BASE DE DONNÉES SQLITE ====================
DB_NAME = "nexhealth.db"

def init_database():
    """Crée la base de données et la table si elles n'existent pas"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS participants (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            age INTEGER,
            sexe TEXT,
            pays TEXT,
            profession TEXT,
            niveau_etude TEXT,
            partenaires_sexuels TEXT,
            utilisation_preservatifs TEXT,
            nb_partenaires TEXT,
            rapport_non_protege TEXT,
            alcool_substances TEXT,
            connaissance_ist TEXT,
            deja_depiste TEXT,
            participation_campagnes TEXT,
            influence_reseaux_sociaux TEXT,
            ist_diagnostiquee TEXT,
            vaccin_hpv TEXT
        )
    ''')
    conn.commit()
    conn.close()

def sauvegarder_participant(data):
    """Sauvegarde un participant dans la base SQLite"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        INSERT INTO participants (
            date, age, sexe, pays, profession, niveau_etude,
            partenaires_sexuels, utilisation_preservatifs, nb_partenaires,
            rapport_non_protege, alcool_substances, connaissance_ist,
            deja_depiste, participation_campagnes, influence_reseaux_sociaux,
            ist_diagnostiquee, vaccin_hpv
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', data)
    conn.commit()
    conn.close()

def charger_participants():
    """Charge tous les participants depuis SQLite vers un DataFrame"""
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM participants", conn)
    conn.close()
    return df

def supprimer_toutes_donnees():
    """Supprime toutes les données de la base"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("DELETE FROM participants")
    conn.commit()
    conn.close()

# ==== Initialisation de la base ====
init_database()

# ==================== SESSION STATE POUR LE MENU ====================
if 'page' not in st.session_state:
    st.session_state.page = "ajouter"

# ==================== MENU PRINCIPAL EN CARRÉS ====================
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("✏️🫂\nAJOUTER\nPARTICIPANT", use_container_width=True, type="primary" if st.session_state.page == "ajouter" else "secondary"):
        st.session_state.page = "ajouter"
        st.rerun()

with col2:
    if st.button("📋👥\nPARTICIPANTS\nENREGISTRÉS", use_container_width=True, type="primary" if st.session_state.page == "participants" else "secondary"):
        st.session_state.page = "participants"
        st.rerun()

with col3:
    if st.button("📈🔬\nANALYSES\nAVANCÉES", use_container_width=True, type="primary" if st.session_state.page == "analyses" else "secondary"):
        st.session_state.page = "analyses"
        st.rerun()

with col4:
    if st.button("📚🩺\nPRÉVENTION\n& RISQUES", use_container_width=True, type="primary" if st.session_state.page == "prevention" else "secondary"):
        st.session_state.page = "prevention"
        st.rerun()

st.markdown("---")

# ==================== CONTENU SELON LA PAGE ====================

# ----- PAGE 1 : AJOUTER UN PARTICIPANT -----
if st.session_state.page == "ajouter":
    st.header("🫂✏️ Ajouter un nouveau participant")
    st.markdown("*Toutes vos réponses sont anonymes et seront conservées dans la base de données.*")
    
    with st.form("collecte", clear_on_submit=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**👤 Votre profil**")
            age = st.slider("Âge", 15, 95, 25)
            sexe = st.radio("Sexe", ["Homme", "Femme", "Autre"])
            pays = st.selectbox("Pays", ["Cameroun", "Sénégal", "Côte d'Ivoire", "Nigéria", "Kenya", "Autre"])
            profession = st.selectbox("Profession", ["Étudiant", "Employé", "Indépendant", "Fonctionnaire", "Sans emploi"])
            niveau_etude = st.selectbox("Niveau d'étude", ["Secondaire", "Universitaire", "Supérieur", "Aucun"])
        
        with col2:
            st.markdown("**💕 Habitudes et comportements**")
            utilisation_preservatifs = st.select_slider("Utilisation des préservatifs", options=["Jamais", "Rarement", "Parfois", "Souvent", "Systématiquement"])
            nb_partenaires = st.selectbox("Nombre de partenaires", ["1", "2-5", "6-10", "11-20", "20+"])
            rapport_non_protege = st.selectbox("Avez-vous eu un rapport non protégé ?", ["Jamais", "Une fois", "Plusieurs fois"])
            
            st.markdown("**🏥 Connaissance et dépistage**")
            connaissance_ist = st.select_slider("Connaissance des IST", options=["Très mauvaise", "Mauvaise", "Moyenne", "Bonne", "Très bonne"])
            deja_depiste = st.radio("Déjà dépisté ?", ["Jamais", "Une fois", "Plusieurs fois", "Régulièrement"])
            participation_campagnes = st.select_slider("Participation aux campagnes", options=["Jamais", "Rarement", "Parfois", "Souvent", "Très souvent"])
            
            st.markdown("**📱 Réseaux sociaux**")
            influence_reseaux = st.select_slider("Influence des réseaux sociaux", options=["Négativement", "Neutre", "Positivement"])
        
        st.markdown("---")
        submitted = st.form_submit_button("✅ Enregistrer le participant", use_container_width=True)
        
        if submitted:
            data = (
                datetime.now().strftime("%Y-%m-%d %H:%M"),
                age, sexe, pays, profession, niveau_etude,
                "Oui", utilisation_preservatifs, nb_partenaires,
                rapport_non_protege, "Parfois", connaissance_ist,
                deja_depiste, participation_campagnes, influence_reseaux,
                "Non renseigné", "Non"
            )
            sauvegarder_participant(data)
            st.success("✅ Participant enregistré avec succès !")
            st.balloons()

# ----- PAGE 2 : PARTICIPANTS ENREGISTRÉS -----
elif st.session_state.page == "participants":
    st.header("📋👥 Participants enregistrés")
    
    df = charger_participants()
    
    if len(df) == 0:
        st.info("📭 Aucun participant pour le moment. Utilisez l'onglet 'AJOUTER PARTICIPANT' pour commencer.")
    else:
        st.markdown(f"**Total : {len(df)} participant(s) dans la base de données**")
        st.dataframe(df, use_container_width=True)
        
        csv = df.to_csv(index=False).encode('utf-8')
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("📥 Exporter les données (CSV)", csv, "participants_ist.csv", "text/csv")
        with col2:
            if st.button("🗑️ Supprimer toutes les données", use_container_width=True):
                supprimer_toutes_donnees()
                st.success("Toutes les données ont été supprimées !")
                st.rerun()

# ----- PAGE 3 : ANALYSES AVANCÉES -----
elif st.session_state.page == "analyses":
    st.header("📈🔬 Analyses avancées des données")
    
    df = charger_participants()
    
    if len(df) < 3:
        st.warning(f"⚠️ Besoin d'au moins 3 participants pour les analyses. Actuellement : {len(df)} participant(s).")
    else:
        # Conversion des données
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
        df['Connaissance_num'] = df['connaissance_ist'].map({'Très mauvaise':1,'Mauvaise':2,'Moyenne':3,'Bonne':4,'Très bonne':5})
        df['Preservatifs_num'] = df['utilisation_preservatifs'].map({'Jamais':1,'Rarement':2,'Parfois':3,'Souvent':4,'Systématiquement':5})
        df['Campagnes_num'] = df['participation_campagnes'].map({'Jamais':1,'Rarement':2,'Parfois':3,'Souvent':4,'Très souvent':5})
        df['Influence_num'] = df['influence_reseaux_sociaux'].map({'Négativement':1,'Neutre':2,'Positivement':3})
        df['Nb_partenaires_num'] = df['nb_partenaires'].map({'1':1,'2-5':2,'6-10':3,'11-20':4,'20+':5})
        
        df_clean = df.dropna(subset=['Age', 'Connaissance_num', 'Preservatifs_num'])
        
        # Score de risque
        df_clean['Score_Risque'] = (
            (6 - df_clean['Preservatifs_num']) * 2 +
            df_clean['Nb_partenaires_num'] * 1.5 +
            (df_clean['rapport_non_protege'].map({'Jamais':0, 'Une fois':1, 'Plusieurs fois':2})) * 2 +
            (df_clean['Connaissance_num'] < 3).astype(int) * 2
        )
        df_clean['Categorie_Risque'] = df_clean['Score_Risque'].apply(lambda x: 'Faible' if x <= 8 else ('Modéré' if x <= 15 else 'Élevé'))
        
        # Sous-onglets
        anal_tab1, anal_tab2, anal_tab3, anal_tab4, anal_tab5 = st.tabs([
            "📈 Régression simple", "🔬 Régression multiple", "🎯 PCA", "🏷️ Classification", "🔄 Clustering"
        ])
        
        # Régression simple
        with anal_tab1:
            if len(df_clean) >= 3:
                X = df_clean[['Age']].values
                y = df_clean['Connaissance_num'].values
                modele = LinearRegression().fit(X, y)
                fig = px.scatter(df_clean, x='Age', y='Connaissance_num', title="Âge vs Connaissance des IST",
                                 color='Categorie_Risque', hover_data=['profession'])
                x_range = np.linspace(df_clean['Age'].min(), df_clean['Age'].max(), 100)
                y_pred = modele.predict(x_range.reshape(-1, 1))
                fig.add_trace(go.Scatter(x=x_range, y=y_pred, mode='lines', name='Tendance', line=dict(color='red')))
                st.plotly_chart(fig, use_container_width=True)
                st.metric("R²", f"{r2_score(y, modele.predict(X)):.3f}")
                st.info("📖 Plus le R² est proche de 1, plus l'âge explique les différences de connaissance.")
        
        # Régression multiple
        with anal_tab2:
            if len(df_clean) >= 5:
                X = df_clean[['Age', 'Preservatifs_num', 'Nb_partenaires_num', 'Campagnes_num']].values
                y = df_clean['Connaissance_num'].values
                modele = LinearRegression().fit(X, y)
                coef_df = pd.DataFrame({'Facteur': ['Âge', 'Préservatifs', 'Nb partenaires', 'Campagnes'], 'Coefficient': modele.coef_})
                st.dataframe(coef_df)
                predictions = modele.predict(X)
                fig = px.scatter(x=y, y=predictions, title="Prédictions vs Réalité")
                fig.add_trace(go.Scatter(x=[1,5], y=[1,5], mode='lines', name='Parfait', line=dict(dash='dash')))
                st.plotly_chart(fig, use_container_width=True)
                st.metric("R²", f"{r2_score(y, predictions):.3f}")
        
        # PCA
        with anal_tab3:
            if len(df_clean) >= 4:
                features = ['Age', 'Connaissance_num', 'Preservatifs_num', 'Nb_partenaires_num', 'Campagnes_num']
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(df_clean[features])
                pca = PCA(n_components=2)
                result = pca.fit_transform(X_scaled)
                df_viz = pd.DataFrame({'PC1': result[:,0], 'PC2': result[:,1], 'Risque': df_clean['Categorie_Risque']})
                fig = px.scatter(df_viz, x='PC1', y='PC2', color='Risque', title="Projection PCA")
                st.plotly_chart(fig, use_container_width=True)
        
        # Classification
        with anal_tab4:
            if len(df_clean) >= 6:
                df_clean['Cible'] = (df_clean['Categorie_Risque'] == 'Élevé').astype(int)
                X = df_clean[['Age', 'Preservatifs_num', 'Nb_partenaires_num', 'Campagnes_num', 'Connaissance_num']].values
                y = df_clean['Cible'].values
                rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=4).fit(X, y)
                
                importance = pd.DataFrame({'Facteur': ['Âge', 'Préservatifs', 'Partenaires', 'Campagnes', 'Connaissance'],
                                          'Importance %': (rf.feature_importances_ * 100).round(1)})
                st.dataframe(importance)
                
                st.subheader("🔮 Évaluez votre risque")
                col1, col2 = st.columns(2)
                with col1:
                    age_t = st.slider("Âge", 18, 65, 25, key="risk_age")
                    preserv_t = st.select_slider("Préservatifs", options=["Systématiquement", "Souvent", "Parfois", "Rarement", "Jamais"], key="risk_preserv")
                with col2:
                    partenaires_t = st.select_slider("Partenaires", options=["1", "2-5", "6-10", "11-20", "20+"], key="risk_partenaires")
                    connais_t = st.select_slider("Connaissance", options=["Très bonne", "Bonne", "Moyenne", "Mauvaise", "Très mauvaise"], key="risk_connais")
                
                if st.button("Estimer mon risque"):
                    p_map = {"Systématiquement":5, "Souvent":4, "Parfois":3, "Rarement":2, "Jamais":1}
                    k_map = {"1":1, "2-5":2, "6-10":3, "11-20":4, "20+":5}
                    c_map = {"Très bonne":5, "Bonne":4, "Moyenne":3, "Mauvaise":2, "Très mauvaise":1}
                    pred = rf.predict([[age_t, p_map[preserv_t], k_map[partenaires_t], 3, c_map[connais_t]]])[0]
                    if pred == 1:
                        st.error("⚠️ Risque ÉLEVÉ")
                    else:
                        st.success("✅ Risque FAIBLE à MODÉRÉ")
        
        # Clustering
        with anal_tab5:
            if len(df_clean) >= 5:
                X = df_clean[['Age', 'Connaissance_num', 'Preservatifs_num', 'Nb_partenaires_num']].values
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                k = st.slider("Nombre de segments", 2, 4, 3)
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(X_scaled)
                fig = px.scatter(df_clean, x='Age', y='Connaissance_num', color=clusters.astype(str),
                                 title=f"Segmentation en {k} groupes")
                st.plotly_chart(fig, use_container_width=True)

# ----- PAGE 4 : PRÉVENTION & RISQUES -----
elif st.session_state.page == "prevention":
    st.header("📚🩺 Prévention, conséquences et risques liés aux IST")
    
    # Conséquences sur la vie quotidienne
    with st.expander("🏠 Conséquences sur la vie quotidienne", expanded=True):
        st.markdown("""
        - **Douleurs chroniques** : Douleurs abdominales, pelviennes ou lors des rapports sexuels
        - **Fatigue persistante** : Épuisement physique et mental récurrent
        - **Impact sur la vie sexuelle** : Baisse de libido, douleurs, peur de transmettre
        - **Stigmatisation sociale** : Isolement, jugement, difficultés relationnelles
        - **Anxiété et dépression** : Stress lié au diagnostic, à la contagiosité, au regard des autres
        - **Arrêts de travail répétés** : Consultations médicales, traitements longs
        """)
    
    # Conséquences à long terme
    with st.expander("⏰ Conséquences à long terme", expanded=True):
        st.markdown("""
        - **Infertilité** (homme et femme) : Lésions des trompes, obstruction des canaux déférents
        - **Grossesses extra-utérines** : Risque vital nécessitant une intervention rapide
        - **Cancers** : HPV responsable de cancers du col de l'utérus, de l'anus, du pénis
        - **Transmission mère-enfant** : Pendant la grossesse ou l'accouchement
        - **Complications neurologiques** : Syphilis tardive pouvant toucher le cerveau
        - **Arthrite réactionnelle** : Inflammation articulaire post-infection
        """)
    
    # Risques liés aux IST non traitées
    with st.expander("⚠️ Risques des IST non traitées", expanded=True):
        st.markdown("""
        - **Aggravation des symptômes** : Douleurs plus intenses, lésions plus étendues
        - **Propagation à d'autres personnes** : Contagiosité prolongée
        - **Développement de résistances** : Aux traitements antibiotiques
        - **Augmentation du risque VIH** : Jusqu'à 10 fois plus élevé
        - **Complications irréversibles** : Stérilité, lésions neurologiques
        - **Propagation à d'autres organes** : Dissémination de l'infection
        """)
    
    st.markdown("---")
    
    # Symptômes
    with st.expander("🚨 Symptômes évocateurs - Consultez rapidement", expanded=True):
        st.markdown("""
        - Écoulements anormaux (urètre, vagin, anus)
        - Douleurs ou brûlures en urinant
        - Lésions, boutons, ulcères ou verrues sur les organes génitaux
        - Démangeaisons intenses ou irritations
        - Ganglions anormalement gonflés dans l'aine
        - Fièvre inexpliquée
        """)
        st.warning("⚠️ **Certaines IST sont asymptomatiques** - Dépistage régulier indispensable")
    
    # Moyens de prévention
    with st.expander("🛡️ Moyens de prévention efficaces", expanded=True):
        st.markdown("""
        - **Préservatifs masculins et féminins** (protection physique)
        - **Dépistage régulier** (au moins 2x par an si vie sexuelle active)
        - **Vaccination** (HPV, Hépatite B)
        - **Communication** : parler de son statut avec son/sa partenaire
        - **Réduction du nombre de partenaires**
        """)
    
    # Lieux de dépistage
    with st.expander("📍 Où se faire dépister ?", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🇨🇲 Cameroun** : Hôpital Général Yaoundé, Hôpital Laquintinie Douala, ASES")
            st.markdown("**🇸🇳 Sénégal** : Hôpital Fann Dakar, ALCS")
        with col2:
            st.markdown("**🇨🇮 Côte d'Ivoire** : INHP Abidjan, Centre Treichville")
            st.markdown("**🌍 Autres pays** : Hôpitaux publics, Croix-Rouge")

# ==================== FOOTER ====================
st.markdown("""
<div class="footer">
    📌 Application développée par MADJOU FORTUNE NESLINE (24G2876)<br>
    Programme INF232 EC2 - Analyse des IST, prévention et protection
</div>
""", unsafe_allow_html=True)