import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from supabase import create_client, Client
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score

# ==================== CONFIGURATION SUPABASE ====================
SUPABASE_URL = "https://joktlxwrewixsrjvhgxu.supabase.co"
SUPABASE_KEY = "sb_publishable_769DD7DqdKSMwmhqKhFshg_R2tNjBGb"

@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase = init_supabase()

# ==================== FONCTIONS BASE DE DONNÉES ====================
def sauvegarder_participant(data):
    """Sauvegarde un participant dans Supabase"""
    try:
        columns = [
            'date', 'age', 'sexe', 'pays', 'profession', 'niveau_etude',
            'partenaires_sexuels', 'utilisation_preservatifs', 'nb_partenaires',
            'rapport_non_protege', 'alcool_substances', 'connaissance_ist',
            'ist_connues', 'connaissance_asymptomatique', 'deja_depiste',
            'savoir_depistage_gratuit', 'frequence_depistage', 'moyens_prevention',
            'participation_campagnes', 'influence_reseaux_sociaux', 'ist_diagnostiquee',
            'vaccin_hpv', 'consultation_medecin'
        ]
        participant_dict = {columns[i]: data[i] for i in range(len(data))}
        response = supabase.table("participants").insert(participant_dict).execute()
        return True
    except Exception as e:
        st.error(f"Erreur lors de la sauvegarde : {e}")
        return False

def charger_participants():
    """Charge tous les participants depuis Supabase"""
    try:
        response = supabase.table("participants").select("*").execute()
        if response.data:
            return pd.DataFrame(response.data)
        else:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Erreur lors du chargement : {e}")
        return pd.DataFrame()

def supprimer_toutes_donnees():
    """Supprime toutes les données de Supabase"""
    try:
        supabase.table("participants").delete().neq("id", 0).execute()
        return True
    except Exception as e:
        st.error(f"Erreur lors de la suppression : {e}")
        return False

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
    
    .stApp, .main, .block-container {
        font-family: 'Manrope', sans-serif !important;
    }
    
    .nexhealth-banner {
        background: linear-gradient(90deg, #1b5e20 0%, #2e7d32 100%);
        border-radius: 24px;
        padding: 40px;
        margin-bottom: 32px;
        color: #ffffff !important;
        box-shadow: 0 10px 30px rgba(27, 94, 32, 0.15);
    }
    .nexhealth-banner h1, .nexhealth-banner p {
        color: #ffffff !important;
    }
    
    .stButton > button {
        background-color: #2e7d32 !important;
        color: white !important;
        border-radius: 40px !important;
        padding: 10px 20px !important;
        font-weight: 600 !important;
        font-family: 'Manrope', sans-serif !important;
    }
    .stButton > button:hover {
        background-color: #1b5e20 !important;
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

# ==================== PAGE 1 : AJOUTER PARTICIPANT ====================
if st.session_state.page == "ajouter":
    st.header("🫂✏️ Ajouter un nouveau participant")
    st.markdown("*Toutes vos réponses sont anonymes et seront conservées.*")
    
    with st.form("collecte", clear_on_submit=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**👤 Votre profil**")
            age = st.slider("Âge", 15, 95, 25)
            sexe = st.radio("Sexe", ["Homme", "Femme", "Autre"])
            pays = st.selectbox("Pays", ["Cameroun", "Sénégal", "Côte d'Ivoire", "Nigéria", "Kenya"])
            profession = st.selectbox("Profession", ["Étudiant", "Employé", "Indépendant", "Fonctionnaire"])
            niveau_etude = st.selectbox("Niveau d'étude", ["Secondaire", "Universitaire", "Supérieur"])
        
        with col2:
            st.markdown("**💕 Habitudes**")
            utilisation_preservatifs = st.select_slider("Utilisation des préservatifs", options=["Jamais", "Rarement", "Parfois", "Souvent", "Systématiquement"])
            nb_partenaires = st.selectbox("Nombre de partenaires", ["1", "2-5", "6-10", "11-20", "20+"])
            rapport_non_protege = st.selectbox("Rapport non protégé", ["Jamais", "Une fois", "Plusieurs fois"])
        
        st.markdown("---")
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("**🏥 Connaissance**")
            connaissance_ist = st.select_slider("Connaissance des IST", options=["Très mauvaise", "Mauvaise", "Moyenne", "Bonne", "Très bonne"])
            deja_depiste = st.radio("Déjà dépisté ?", ["Jamais", "Une fois", "Plusieurs fois", "Régulièrement"])
        
        with col4:
            st.markdown("**📱 Réseaux sociaux**")
            influence_reseaux = st.select_slider("Influence des réseaux sociaux", options=["Négativement", "Neutre", "Positivement"])
        
        submit = st.form_submit_button("✅ Envoyer", use_container_width=True)
        
        if submit:
            data = (
                datetime.now().strftime("%Y-%m-%d %H:%M"),
                age, sexe, pays, profession, niveau_etude,
                "Oui", utilisation_preservatifs, nb_partenaires,
                rapport_non_protege, "Parfois", connaissance_ist,
                "", "", deja_depiste,
                "", "", "",
                "", influence_reseaux, "",
                "", ""
            )
            if sauvegarder_participant(data):
                st.success("✅ Participant enregistré !")
                st.balloons()

# ==================== PAGE 2 : PARTICIPANTS ====================
elif st.session_state.page == "participants":
    st.header("📋👥 Participants enregistrés")
    df = charger_participants()
    if len(df) == 0:
        st.info("📭 Aucun participant. Utilisez 'AJOUTER PARTICIPANT'.")
    else:
        st.metric("Total", len(df))
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode('utf-8')
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("📥 Export CSV", csv, "participants.csv")
        with col2:
            if st.button("🗑️ Supprimer tout"):
                supprimer_toutes_donnees()
                st.rerun()

# ==================== PAGE 3 : ANALYSES ====================
elif st.session_state.page == "analyses":
    st.header("📈🔬 Analyses avancées")
    df = charger_participants()
    
    if len(df) < 3:
        st.warning(f"⚠️ Besoin d'au moins 3 participants. Actuellement : {len(df)}")
    else:
        # Conversion
        df['Age'] = pd.to_numeric(df['age'], errors='coerce')
        df['Connaissance_num'] = df['connaissance_ist'].map({'Très mauvaise':1,'Mauvaise':2,'Moyenne':3,'Bonne':4,'Très bonne':5})
        df['Preservatifs_num'] = df['utilisation_preservatifs'].map({'Jamais':1,'Rarement':2,'Parfois':3,'Souvent':4,'Systématiquement':5})
        df['Partenaires_num'] = df['nb_partenaires'].map({'1':1,'2-5':2,'6-10':3,'11-20':4,'20+':5})
        df_clean = df.dropna(subset=['Age', 'Connaissance_num', 'Preservatifs_num'])
        
        # Score risque
        df_clean['Score_Risque'] = (6 - df_clean['Preservatifs_num']) * 2 + df_clean['Partenaires_num'] * 1.5
        df_clean['Categorie_Risque'] = df_clean['Score_Risque'].apply(lambda x: 'Faible' if x <= 8 else ('Modéré' if x <= 15 else 'Élevé'))
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Régression simple", "🔬 Régression multiple", "🎯 PCA", "🏷️ Classification", "📊 Graphiques"])
        
        # Tab 1
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
                st.markdown("""
                <div class="interpretation-box">
                <b>📖 Interprétation :</b><br>
                - Plus le R² est proche de 1, plus l'âge explique les différences de connaissance.<br>
                - Coefficient positif = plus on est âgé, meilleure est la connaissance.
                </div>
                """, unsafe_allow_html=True)
        
        # Tab 2
        with tab2:
            if len(df_clean) >= 4:
                X = df_clean[['Age', 'Preservatifs_num', 'Partenaires_num']].values
                y = df_clean['Connaissance_num'].values
                modele = LinearRegression().fit(X, y)
                st.dataframe(pd.DataFrame({'Facteur':['Âge','Préservatifs','Partenaires'],'Coefficient':modele.coef_}))
                predictions = modele.predict(X)
                fig = px.scatter(x=y, y=predictions)
                fig.add_trace(go.Scatter(x=[1,5], y=[1,5], mode='lines', name='Parfait', line=dict(dash='dash')))
                st.plotly_chart(fig, use_container_width=True)
                st.metric("R²", f"{r2_score(y, predictions):.3f}")
                st.markdown("""
                <div class="interpretation-box">
                <b>📖 Interprétation :</b><br>
                - Coefficient positif = ce facteur améliore la connaissance.<br>
                - Points proches de la diagonale = bonne prédiction.
                </div>
                """, unsafe_allow_html=True)
        
        # Tab 3
        with tab3:
            if len(df_clean) >= 4:
                features = ['Age', 'Connaissance_num', 'Preservatifs_num']
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(df_clean[features])
                pca = PCA(n_components=2)
                result = pca.fit_transform(X_scaled)
                fig = px.scatter(x=result[:,0], y=result[:,1], color=df_clean['Categorie_Risque'])
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("""
                <div class="interpretation-box">
                <b>📖 Interprétation :</b><br>
                - Les points proches ont des profils similaires.<br>
                - Les couleurs indiquent le niveau de risque.
                </div>
                """, unsafe_allow_html=True)
        
        # Tab 4
        with tab4:
            if len(df_clean) >= 5:
                df_clean['Cible'] = (df_clean['Categorie_Risque'] == 'Élevé').astype(int)
                X = df_clean[['Age', 'Preservatifs_num', 'Partenaires_num']].values
                y = df_clean['Cible'].values
                rf = RandomForestClassifier().fit(X, y)
                
                st.subheader("🔮 Testez votre risque")
                col1, col2 = st.columns(2)
                with col1:
                    age_t = st.slider("Âge", 18, 65, 25, key="age_t")
                    preserv_t = st.select_slider("Préservatifs", options=["Systématiquement","Souvent","Parfois","Rarement","Jamais"], key="preserv_t")
                with col2:
                    part_t = st.select_slider("Partenaires", options=["1","2-5","6-10","11-20","20+"], key="part_t")
                
                if st.button("Estimer mon risque"):
                    p_map = {"Systématiquement":5,"Souvent":4,"Parfois":3,"Rarement":2,"Jamais":1}
                    k_map = {"1":1,"2-5":2,"6-10":3,"11-20":4,"20+":5}
                    pred = rf.predict([[age_t, p_map[preserv_t], k_map[part_t]]])[0]
                    if pred == 1:
                        st.error("⚠️ Risque ÉLEVÉ - Consultez la rubrique Prévention")
                    else:
                        st.success("✅ Risque FAIBLE à MODÉRÉ")
        
        # Tab 5 - Graphiques
        with tab5:
            st.subheader("📊 Graphiques descriptifs")
            col1, col2 = st.columns(2)
            with col1:
                fig_hist = px.histogram(df_clean, x='Age', nbins=15, title="Distribution des âges")
                st.plotly_chart(fig_hist, use_container_width=True)
                fig_bar = px.bar(df_clean['connaissance_ist'].value_counts().reset_index(), x='index', y='connaissance_ist', title="Niveau de connaissance")
                st.plotly_chart(fig_bar, use_container_width=True)
            with col2:
                fig_pie = px.pie(df_clean, names='Categorie_Risque', title="Répartition par risque")
                st.plotly_chart(fig_pie, use_container_width=True)
                fig_preserv = px.bar(df_clean['utilisation_preservatifs'].value_counts().reset_index(), x='index', y='utilisation_preservatifs', title="Utilisation des préservatifs")
                st.plotly_chart(fig_preserv, use_container_width=True)

# ==================== PAGE 4 : PRÉVENTION ====================
elif st.session_state.page == "prevention":
    st.header("📚🩺 Prévention et risques liés aux IST")
    
    with st.expander("📖 Qu'est-ce qu'une IST ?", expanded=True):
        st.markdown("""
        Les **IST (Infections Sexuellement Transmissibles)** sont des infections qui se transmettent lors de rapports sexuels non protégés.
        
        **Points clés :**
        - Certaines sont asymptomatiques
        - Toutes peuvent avoir des conséquences graves
        - La plupart sont évitables ou soignables
        """)
    
    with st.expander("🦠 Principales IST"):
        st.markdown("""
        **VIH :** Destruction immunitaire. Traitement antirétroviral.
        **Syphilis :** Lésions, complications neurologiques. Se guérit.
        **Gonorrhée :** Écoulements, douleurs. Risque d'infertilité.
        **Chlamydia :** Asymptomatique. Peut rendre stérile.
        **HPV :** Verrues, cancers. Vaccination préventive.
        """)
    
    with st.expander("🚨 Symptômes évocateurs"):
        st.markdown("""
        - Écoulements anormaux
        - Douleurs en urinant
        - Lésions ou verrues
        - Démangeaisons
        - Ganglions gonflés
        """)
        st.warning("⚠️ Dépistage régulier indispensable")
    
    with st.expander("🛡️ Moyens de prévention"):
        st.markdown("""
        - Préservatifs (masculins et féminins)
        - Dépistage régulier (2x par an)
        - Vaccination HPV
        - Communication avec le/la partenaire
        """)
    
    with st.expander("📍 Où se faire dépister ?"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Cameroun :** Hôpital Général Yaoundé, Hôpital Laquintinie Douala")
            st.markdown("**Sénégal :** Hôpital Fann Dakar, ALCS")
        with col2:
            st.markdown("**Côte d'Ivoire :** INHP Abidjan")
            st.markdown("**Autres :** Hôpitaux publics, Croix-Rouge")
        st.info("📢 Ces informations ne remplacent pas l'avis d'un médecin")

# ==================== FOOTER ====================
st.markdown("""
<div class="nexhealth-footer">
    📌 MADJOU FORTUNE NESLINE (24G2876) - INF232 EC2
</div>
""", unsafe_allow_html=True)