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
    .banner {
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 50%, #4caf50 100%);
        padding: 25px;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 30px;
        color: white;
    }
    .banner h1 { font-size: 2.2rem; font-weight: bold; font-style: italic; margin: 0; }
    .banner p { font-size: 1rem; font-weight: bold; font-style: italic; margin: 8px 0 0 0; }
    .banner-sub { font-size: 0.9rem; margin: 3px 0; }
    .footer {
        text-align: center;
        padding: 15px;
        margin-top: 40px;
        background: linear-gradient(135deg, #1b5e20, #2e7d32);
        color: white;
        border-radius: 15px;
        font-weight: bold;
        font-size: 0.8rem;
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
            date TEXT, age INTEGER, sexe TEXT, pays TEXT, profession TEXT, niveau_etude TEXT,
            partenaires_sexuels TEXT, utilisation_preservatifs TEXT, nb_partenaires TEXT,
            rapport_non_protege TEXT, alcool_substances TEXT, connaissance_ist TEXT,
            ist_connues TEXT, connaissance_asymptomatique TEXT, deja_depiste TEXT,
            savoir_depistage_gratuit TEXT, frequence_depistage TEXT, moyens_prevention TEXT,
            participation_campagnes TEXT, influence_reseaux_sociaux TEXT,
            ist_diagnostiquee TEXT, vaccin_hpv TEXT, consultation_medecin TEXT
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
    st.markdown("*Toutes vos réponses sont anonymes et seront conservées dans la base de données.*")
    
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
                ["1", "2-5", "6-10", "11-20", "20+", "Préfère ne répondre"])
            rapport_non_protege = st.selectbox("Avez-vous déjà eu un rapport non protégé ?", ["Jamais", "Une fois", "Plusieurs fois", "Préfère ne répondre"])
            alcool_substances = st.select_slider("Consommez-vous de l'alcool ou des substances avant des rapports ?",
                options=["Jamais", "Rarement", "Parfois", "Souvent", "Très souvent"])
        
        st.markdown("---")
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("**🏥 Connaissance, dépistage et prévention**")
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
        
        if submit:
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
            st.success("✅ Merci ! Votre réponse a été enregistrée.")
            st.balloons()

# ==================== PAGE 2 : PARTICIPANTS ENREGISTRÉS ====================
elif st.session_state.page == "participants":
    st.header("📋👥 Participants enregistrés")
    df = charger_participants()
    if len(df) == 0:
        st.info("📭 Aucun participant pour le moment. Utilisez 'AJOUTER PARTICIPANT' pour commencer.")
    else:
        st.metric("Total participants", len(df))
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

# ==================== PAGE 3 : ANALYSES AVANCÉES ====================
elif st.session_state.page == "analyses":
    st.header("📈🔬 Analyses avancées des données")
    df = charger_participants()
    
    if len(df) < 3:
        st.warning(f"⚠️ Besoin d'au moins 3 participants pour les analyses. Actuellement : {len(df)} participant(s).")
    else:
        df['Age'] = pd.to_numeric(df['age'], errors='coerce')
        df['Connaissance_num'] = df['connaissance_ist'].map({'Très mauvaise':1, 'Mauvaise':2, 'Moyenne':3, 'Bonne':4, 'Très bonne':5})
        df['Preservatifs_num'] = df['utilisation_preservatifs'].map({'Jamais':1, 'Rarement':2, 'Parfois':3, 'Souvent':4, 'Systématiquement':5, 'Pas concerné(e)':3})
        df['Campagnes_num'] = df['participation_campagnes'].map({'Jamais':1, 'Rarement':2, 'Parfois':3, 'Souvent':4, 'Très souvent':5})
        df['Partenaires_num'] = df['nb_partenaires'].map({'1':1, '2-5':2, '6-10':3, '11-20':4, '20+':5})
        
        df_clean = df.dropna(subset=['Age', 'Connaissance_num', 'Preservatifs_num'])
        df_clean['Score_Risque'] = (6 - df_clean['Preservatifs_num']) * 2 + df_clean['Partenaires_num'] * 1.5
        df_clean['Categorie_Risque'] = df_clean['Score_Risque'].apply(lambda x: 'Faible' if x <= 8 else ('Modéré' if x <= 15 else 'Élevé'))
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Régression simple", "🔬 Régression multiple", "🎯 PCA", "🏷️ Classification", "🔄 Clustering"
        ])
        
        with tab1:
            if len(df_clean) >= 3:
                X = df_clean[['Age']].values
                y = df_clean['Connaissance_num'].values
                modele = LinearRegression().fit(X, y)
                fig = px.scatter(df_clean, x='Age', y='Connaissance_num', 
                                 title="Âge vs Connaissance des IST",
                                 color='Categorie_Risque', hover_data=['profession'])
                x_range = np.linspace(df_clean['Age'].min(), df_clean['Age'].max(), 100)
                y_pred = modele.predict(x_range.reshape(-1, 1))
                fig.add_trace(go.Scatter(x=x_range, y=y_pred, mode='lines', 
                                        name='Tendance', line=dict(color='#2e7d32', width=3)))
                st.plotly_chart(fig, use_container_width=True)
                st.metric("R²", f"{r2_score(y, modele.predict(X)):.3f}")
        
        with tab2:
            if len(df_clean) >= 5:
                X = df_clean[['Age', 'Preservatifs_num', 'Partenaires_num', 'Campagnes_num']].values
                y = df_clean['Connaissance_num'].values
                modele = LinearRegression().fit(X, y)
                st.dataframe(pd.DataFrame({'Facteur': ['Âge', 'Préservatifs', 'Partenaires', 'Campagnes'],
                                           'Coefficient': modele.coef_}))
                predictions = modele.predict(X)
                fig = px.scatter(x=y, y=predictions, title="Prédictions vs Réalité")
                fig.add_trace(go.Scatter(x=[1,5], y=[1,5], mode='lines', 
                                        name='Parfait', line=dict(dash='dash', color='#2e7d32')))
                st.plotly_chart(fig, use_container_width=True)
                st.metric("R²", f"{r2_score(y, predictions):.3f}")
        
        with tab3:
            if len(df_clean) >= 4:
                features = ['Age', 'Connaissance_num', 'Preservatifs_num', 'Partenaires_num']
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(df_clean[features])
                pca = PCA(n_components=2)
                result = pca.fit_transform(X_scaled)
                df_viz = pd.DataFrame({'PC1': result[:,0], 'PC2': result[:,1], 'Risque': df_clean['Categorie_Risque']})
                fig = px.scatter(df_viz, x='PC1', y='PC2', color='Risque', title="PCA")
                st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            if len(df_clean) >= 6:
                df_clean['Cible'] = (df_clean['Categorie_Risque'] == 'Élevé').astype(int)
                X = df_clean[['Age', 'Preservatifs_num', 'Partenaires_num', 'Campagnes_num']].values
                y = df_clean['Cible'].values
                rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)
                importance = pd.DataFrame({'Facteur': ['Âge', 'Préservatifs', 'Partenaires', 'Campagnes'],
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
                
                if st.button("Estimer mon risque", key="predict_risk"):
                    p_map = {"Systématiquement":5, "Souvent":4, "Parfois":3, "Rarement":2, "Jamais":1}
                    k_map = {"1":1, "2-5":2, "6-10":3, "11-20":4, "20+":5}
                    pred = rf.predict([[age_t, p_map[preserv_t], k_map[partenaires_t], 3]])[0]
                    if pred == 1:
                        st.error("⚠️ Risque ÉLEVÉ - Consultez la rubrique Prévention")
                    else:
                        st.success("✅ Risque FAIBLE à MODÉRÉ")
        
        with tab5:
            if len(df_clean) >= 5:
                X = df_clean[['Age', 'Connaissance_num', 'Preservatifs_num']].values
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                k = st.slider("Nombre de segments", 2, 4, 3)
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(X_scaled)
                fig = px.scatter(df_clean, x='Age', y='Connaissance_num', 
                                 color=clusters.astype(str), title=f"Segmentation en {k} groupes")
                st.plotly_chart(fig, use_container_width=True)

# ==================== PAGE 4 : PRÉVENTION ====================
elif st.session_state.page == "prevention":
    st.header("📚🩺 Prévention et risques liés aux IST")
    
    with st.expander("📖 Qu'est-ce qu'une IST ?", expanded=True):
        st.markdown("""
        Les IST (Infections Sexuellement Transmissibles) sont des infections qui se transmettent lors de rapports sexuels non protégés.
        
        **Points clés :**
        - Certaines sont asymptomatiques
        - Toutes peuvent avoir des conséquences graves
        - La plupart sont évitables ou soignables
        """)
    
    with st.expander("🦠 Principales IST - Symptômes et conséquences"):
        st.markdown("""
        **VIH / Sida :** Destruction du système immunitaire. Traitement antirétroviral.
        
        **Syphilis :** Lésions cutanées, complications neurologiques. Se guérit avec antibiotiques.
        
        **Gonorrhée :** Écoulements, douleurs. Risque d'infertilité.
        
        **Chlamydia :** Souvent asymptomatique. Peut rendre stérile.
        
        **HPV :** Verrues génitales, cancers. Vaccination préventive.
        """)
    
    with st.expander("🚨 Symptômes évocateurs"):
        st.markdown("- Écoulements anormaux\n- Douleurs en urinant\n- Lésions ou verrues\n- Démangeaisons\n- Ganglions gonflés")
        st.warning("⚠️ Dépistage régulier indispensable (certaines IST sont asymptomatiques)")
    
    with st.expander("🛡️ Moyens de prévention"):
        st.markdown("- Préservatifs (masculins et féminins)\n- Dépistage régulier (2x par an)\n- Vaccination HPV\n- Communication avec le/la partenaire")
    
    with st.expander("📍 Où se faire dépister ?"):
        st.markdown("""
        **Cameroun :** Hôpital Général Yaoundé, Hôpital Laquintinie Douala
        **Sénégal :** Hôpital Fann Dakar, ALCS
        **Côte d'Ivoire :** INHP Abidjan
        """)
        st.info("📢 Ces informations ne remplacent pas l'avis d'un médecin")

# ==================== FOOTER ====================
st.markdown("""
<div class="footer">
    📌 MADJOU FORTUNE NESLINE (24G2876) - INF232 EC2 - Tous droits réservés
</div>
""", unsafe_allow_html=True)