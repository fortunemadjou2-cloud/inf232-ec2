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
    .risk-card {
        background: #e8f5e9;
        padding: 15px;
        border-radius: 15px;
        margin-bottom: 15px;
        border-left: 5px solid #2e7d32;
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

# ==================== PAGE 1 : AJOUTER PARTICIPANT (FORMULAIRE COMPLET) ====================
if st.session_state.page == "ajouter":
    st.header("🫂✏️ Ajouter un nouveau participant")
    st.markdown("*Toutes vos réponses sont anonymes et seront conservées dans la base de données.*")
    
    with st.form("collecte", clear_on_submit=True):
        col1, col2 = st.columns(2)
        
        # SECTION A : PROFIL
        with col1:
            st.markdown("**👤 Votre profil**")
            age = st.slider("Âge", 15, 95, 25)
            sexe = st.radio("Sexe", ["Homme", "Femme", "Autre", "Préfère ne pas répondre"])
            pays = st.selectbox("Pays", ["Sénégal", "Côte d'Ivoire", "Nigéria", "Kenya", "Ghana", "Maroc", "Cameroun", "Autre"])
            profession = st.selectbox("Profession", ["Étudiant", "Employé", "Indépendant", "Fonctionnaire", "Sans emploi", "Retraité", "Autre"])
            niveau_etude = st.selectbox("Niveau d'étude", ["Aucun", "Primaire", "Secondaire", "Universitaire", "Supérieur", "Autre"])
        
        # SECTION B : HABITUDES
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
        
        # SECTION C : CONNAISSANCE ET PRÉVENTION
        with col3:
            st.markdown("**🏥 Connaissance, dépistage et prévention**")
            connaissance_ist = st.select_slider("Comment évaluez-vous votre connaissance des IST ?",
                options=["Très mauvaise", "Mauvaise", "Moyenne", "Bonne", "Très bonne"])
            ist_connues = st.multiselect("Quelles IST connaissez-vous ?",
                ["Sida/VIH", "Syphilis", "Gonorrhée", "Chlamydia", "HPV", "Herpès", "Hépatite B"])
            ist_autres = st.text_input("Autres IST que vous connaissez (facultatif)")
            connaissance_asympto = st.radio("Savez-vous que certaines IST peuvent être asymptomatiques (sans symptômes) ?",
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
        
        # SECTION D : DONNÉES MÉDICALES
        with col5:
            st.markdown("**📊 Données médicales (optionnelles)**")
            ist_diagnostiquee = st.radio("Avez-vous déjà eu une IST diagnostiquée ?",
                ["Non", "Oui, guérie", "Oui, en traitement", "Préfère ne répondre"])
            vaccin_hpv = st.radio("Avez-vous été vacciné(e) contre le HPV (Papillomavirus) ?",
                ["Oui", "Non", "Je ne sais pas", "En cours"])
            consultation_medecin = st.select_slider("Consultez-vous un gynécologue/urologue régulièrement ?",
                options=["Jamais", "Rarement", "Parfois", "Régulièrement"])
        
        # Bouton d'envoi
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
        # Conversion des données pour les analyses
        df['Age'] = pd.to_numeric(df['age'], errors='coerce')
        df['Connaissance_num'] = df['connaissance_ist'].map({'Très mauvaise':1, 'Mauvaise':2, 'Moyenne':3, 'Bonne':4, 'Très bonne':5})
        df['Preservatifs_num'] = df['utilisation_preservatifs'].map({'Jamais':1, 'Rarement':2, 'Parfois':3, 'Souvent':4, 'Systématiquement':5, 'Pas concerné(e)':3})
        df['Campagnes_num'] = df['participation_campagnes'].map({'Jamais':1, 'Rarement':2, 'Parfois':3, 'Souvent':4, 'Très souvent':5})
        df['Influence_num'] = df['influence_reseaux_sociaux'].map({'Très négativement':1, 'Négativement':2, 'Neutre':3, 'Positivement':4, 'Très positivement':5})
        df['Partenaires_num'] = df['nb_partenaires'].map({'1':1, '2-5':2, '6-10':3, '11-20':4, '20+':5})
        
        df_clean = df.dropna(subset=['Age', 'Connaissance_num', 'Preservatifs_num', 'Campagnes_num'])
        
        # Score de risque
        df_clean['Score_Risque'] = (6 - df_clean['Preservatifs_num']) * 2 + df_clean['Partenaires_num'] * 1.5
        df_clean['Categorie_Risque'] = df_clean['Score_Risque'].apply(lambda x: 'Faible' if x <= 8 else ('Modéré' if x <= 15 else 'Élevé'))
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Régression simple", "🔬 Régression multiple", "🎯 PCA", "🏷️ Classification", "🔄 Clustering"
        ])
        
        # Régression simple
        with tab1:
            st.subheader("Relation entre l'âge et la connaissance des IST")
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
                st.info("📖 Interprétation : La ligne verte montre la tendance. Plus le R² est proche de 1, plus l'âge est un bon prédicteur.")
        
        # Régression multiple
        with tab2:
            st.subheader("Facteurs influençant la connaissance des IST")
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
                st.info("📖 Un coefficient POSITIF = ce facteur améliore la connaissance. Plus le R² est proche de 1, meilleur est le modèle.")
        
        # PCA
        with tab3:
            st.subheader("Projection des profils (PCA)")
            if len(df_clean) >= 4:
                features = ['Age', 'Connaissance_num', 'Preservatifs_num', 'Partenaires_num', 'Campagnes_num']
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(df_clean[features])
                pca = PCA(n_components=2)
                result = pca.fit_transform(X_scaled)
                df_viz = pd.DataFrame({
                    'PC1': result[:,0], 'PC2': result[:,1],
                    'Risque': df_clean['Categorie_Risque'], 'Âge': df_clean['Age']
                })
                fig = px.scatter(df_viz, x='PC1', y='PC2', color='Risque', size='Âge',
                                 title="Projection PCA - Les points proches se ressemblent",
                                 labels={'PC1': f"Dimension 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
                                        'PC2': f"Dimension 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)"})
                st.plotly_chart(fig, use_container_width=True)
                st.info("📖 Les points proches ont des comportements similaires. Les couleurs indiquent le niveau de risque estimé.")
        
        # Classification
        with tab4:
            st.subheader("Prédiction du risque de contracter une IST")
            if len(df_clean) >= 6:
                df_clean['Cible'] = (df_clean['Categorie_Risque'] == 'Élevé').astype(int)
                X = df_clean[['Age', 'Preservatifs_num', 'Partenaires_num', 'Campagnes_num', 'Connaissance_num']].values
                y = df_clean['Cible'].values
                rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=4).fit(X, y)
                
                importance = pd.DataFrame({
                    'Facteur': ['Âge', 'Préservatifs', 'Partenaires', 'Campagnes', 'Connaissance'],
                    'Importance (%)': (rf.feature_importances_ * 100).round(1)
                }).sort_values('Importance (%)', ascending=False)
                st.dataframe(importance, use_container_width=True)
                
                st.subheader("🔮 Évaluez VOTRE risque personnel")
                col1, col2 = st.columns(2)
                with col1:
                    age_t = st.slider("Âge", 18, 65, 25, key="risk_age")
                    preserv_t = st.select_slider("Utilisation des préservatifs", 
                        options=["Systématiquement", "Souvent", "Parfois", "Rarement", "Jamais"], key="risk_preserv")
                with col2:
                    partenaires_t = st.select_slider("Nombre de partenaires", options=["1", "2-5", "6-10", "11-20", "20+"], key="risk_part")
                    connais_t = st.select_slider("Connaissance des IST", 
                        options=["Très bonne", "Bonne", "Moyenne", "Mauvaise", "Très mauvaise"], key="risk_connais")
                
                if st.button("🔮 Estimer mon risque", key="predict_risk"):
                    p_map = {"Systématiquement":5, "Souvent":4, "Parfois":3, "Rarement":2, "Jamais":1}
                    k_map = {"1":1, "2-5":2, "6-10":3, "11-20":4, "20+":5}
                    c_map = {"Très bonne":5, "Bonne":4, "Moyenne":3, "Mauvaise":2, "Très mauvaise":1}
                    pred = rf.predict([[age_t, p_map[preserv_t], k_map[partenaires_t], 3, c_map[connais_t]]])[0]
                    proba = rf.predict_proba([[age_t, p_map[preserv_t], k_map[partenaires_t], 3, c_map[connais_t]]]).max()
                    if pred == 1:
                        st.error(f"⚠️ Risque ÉLEVÉ (confiance : {proba:.1%})")
                        st.markdown("""
                        **💡 Recommandations :**
                        - Utilisez des préservatifs à chaque rapport
                        - Réduisez le nombre de partenaires
                        - Faites-vous dépister régulièrement (2x/an)
                        """)
                    else:
                        st.success(f"✅ Risque FAIBLE à MODÉRÉ (confiance : {proba:.1%})")
                        st.markdown("**💡 Continuez les bonnes pratiques et restez informé(e) !**")
        
        # Clustering
        with tab5:
            st.subheader("Segmentation automatique des participants")
            if len(df_clean) >= 5:
                X = df_clean[['Age', 'Connaissance_num', 'Preservatifs_num', 'Partenaires_num']].values
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                k = st.slider("Nombre de segments", 2, 4, 3, help="Plus il y a de segments, plus la segmentation est fine")
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(X_scaled)
                fig = px.scatter(df_clean, x='Age', y='Connaissance_num', 
                                 color=clusters.astype(str), size='Preservatifs_num',
                                 title=f"Segmentation des participants en {k} groupes",
                                 hover_data=['profession'])
                st.plotly_chart(fig, use_container_width=True)
                st.info("📖 Chaque couleur représente un groupe aux habitudes similaires. Identifiez à quel groupe vous ressemblez.")
        
        # Graphiques supplémentaires
        st.markdown("---")
        st.subheader("📊 Graphiques descriptifs supplémentaires")
        col1, col2 = st.columns(2)
        with col1:
            fig_hist = px.histogram(df_clean, x='Age', nbins=15, title="Distribution des âges")
            st.plotly_chart(fig_hist, use_container_width=True)
            fig_pie = px.pie(df_clean, names='Categorie_Risque', title="Répartition par niveau de risque")
            st.plotly_chart(fig_pie, use_container_width=True)
        with col2:
            fig_bar = px.bar(df_clean['connaissance_ist'].value_counts().reset_index(), 
                            x='index', y='connaissance_ist', title="Niveau de connaissance des IST")
            st.plotly_chart(fig_bar, use_container_width=True)
            fig_preserv = px.bar(df_clean['utilisation_preservatifs'].value_counts().reset_index(),
                                x='index', y='utilisation_preservatifs', title="Utilisation des préservatifs")
            st.plotly_chart(fig_preserv, use_container_width=True)

# ==================== PAGE 4 : PRÉVENTION & RISQUES ====================
elif st.session_state.page == "prevention":
    st.header("📚🩺 Prévention, conséquences et risques liés aux IST")
    
    # Définition des IST
    with st.expander("📖 Qu'est-ce qu'une IST ? (Définition)", expanded=True):
        st.markdown("""
        **IST** = **Infections Sexuellement Transmissibles** (anciennement appelées MST).
        
        Ce sont des infections qui se transmettent principalement lors de rapports sexuels (vagins, anaux ou oraux) non protégés.
        
        **Caractéristiques importantes :**
        - Certaines IST sont **asymptomatiques** (pas de symptômes visibles)
        - Toutes les IST peuvent avoir des **conséquences graves** si non traitées
        - La plupart sont **évitables** par des moyens de prévention simples
        - La **plupart se soignent** (bactéries) ou se contrôlent (virus)
        """)
    
    # Liste des IST avec symptômes et conséquences (sections déroulantes individuelles)
    st.markdown("### 🦠 Les principales IST : symptômes et conséquences")
    st.markdown("*Cliquez sur chaque IST pour plus d'informations*")
    
    ist_data = {
        "VIH / Sida": {
            "symptomes": "Phase aiguë : fièvre, fatigue, ganglions. Phase chronique : asymptomatique pendant des années. Phase Sida : infections opportunistes, amaigrissement.",
            "consequences": "Destruction du système immunitaire, infections opportunistes, cancers, décès si non traité.",
            "traitement": "Traitement antirétroviral (non guérit mais contrôle la maladie)."
        },
        "Syphilis": {
            "symptomes": "1er stade : chancre (ulcère indolore). 2e stade : éruptions cutanées, fièvre, ganglions. 3e stade : complications neurologiques et cardiovasculaires.",
            "consequences": "Lésions neurologiques, paralysie, cécité, démence, maladies cardiaques, décès.",
            "traitement": "Se guérit avec des antibiotiques (pénicilline)."
        },
        "Gonorrhée (Chaudepisse)": {
            "symptomes": "Écoulements (urètre, vagin, anus), douleurs en urinant, douleurs pelviennes.",
            "consequences": "Infertilité (homme et femme), grossesse extra-utérine, propagation aux articulations.",
            "traitement": "Se guérit avec des antibiotiques (risque de résistance)."
        },
        "Chlamydia": {
            "symptomes": "Souvent asymptomatique ! Écoulements, douleurs en urinant, douleurs pelviennes.",
            "consequences": "Infertilité (femme), grossesse extra-utérine, douleurs pelviennes chroniques.",
            "traitement": "Se guérit avec des antibiotiques."
        },
        "HPV (Papillomavirus)": {
            "symptomes": "Verrues génitales (condylomes), souvent asymptomatique.",
            "consequences": "Cancers : col de l'utérus, vagin, vulve, pénis, anus, gorge.",
            "traitement": "Traitement des lésions, vaccination PREVENTIVE efficace."
        },
        "Herpès génital": {
            "symptomes": "Vésicules douloureuses, ulcères, démangeaisons, brûlures.",
            "consequences": "Récidives fréquentes, transmission au nouveau-né (risque grave).",
            "traitement": "Traitement antiviral (réduit les symptômes, ne guérit pas)."
        },
        "Hépatite B": {
            "symptomes": "Fatigue, jaunisse, nausées, douleurs abdominales, fièvre.",
            "consequences": "Hépatite chronique, cirrhose, cancer du foie, décès.",
            "traitement": "Vaccination PREVENTIVE, traitement antiviral (forme chronique)."
        }
    }
    
    for ist, infos in ist_data.items():
        with st.expander(f"🩺 {ist}"):
            st.markdown(f"""
            **🤒 Symptômes :** {infos['symptomes']}
            
            **⚠️ Conséquences :** {infos['consequences']}
            
            **💊 Traitement / Prévention :** {infos['traitement']}
            """)
    
    st.markdown("---")
    
    # Autres sections de prévention
    with st.expander("🚨 Symptômes évocateurs - Consultez rapidement"):
        st.markdown("""
        - Écoulements anormaux (urètre, vagin, anus)
        - Douleurs ou brûlures en urinant
        - Lésions, boutons, ulcères ou verrues sur les organes génitaux
        - Démangeaisons intenses ou irritations
        - Ganglions anormalement gonflés dans l'aine
        - Fièvre inexpliquée
        """)
        st.warning("⚠️ **IMPORTANT** : Certaines IST sont asymptomatiques (sans symptômes visibles). Le seul moyen d'être sûr(e) est le DÉPISTAGE RÉGULIER.")
        st.info("📢 *Ces informations ne remplacent pas l'avis d'un médecin. Consultez un professionnel de santé pour tout diagnostic.*")
    
    with st.expander("🛡️ Les moyens de prévention efficaces"):
        st.markdown("""
        - **Préservatifs masculins et féminins** (protection contre la plupart des IST)
        - **Dépistage régulier** (au moins 2 fois par an si vie sexuelle active avec partenaires multiples)
        - **Vaccination** (HPV, Hépatite B)
        - **Traitements préventifs** (Prophylaxie Pré-Exposition contre le VIH)
        -