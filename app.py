import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

# Configuration
st.set_page_config(page_title="NEXHEALTH SURVEY NO TABOO", page_icon="🩺", layout="wide")

# Titre
st.title("🩺 NEXHEALTH SURVEY NO TABOO")

# Slogan
st.markdown("""
**✨ Parce qu'en santé, il n'y a pas de tabou. ✨**  
**🔍 Brisons le silence sur les IST, protégeons notre bien-être.**  
**⚖️ La santé sexuelle est un droit, la protection est une responsabilité.**
""")

# Initialisation des données
if 'donnees' not in st.session_state:
    st.session_state.donnees = pd.DataFrame(columns=[
        'Date', 'Age', 'Sexe', 'Pays', 'Profession', 'Niveau_Etude',
        'Partenaires_Sexuels', 'Utilisation_Preservatifs', 'Nb_Partenaires',
        'Rapport_Non_Protege', 'Alcool_Substances',
        'Connaissance_IST', 'Deja_Depiste', 'Participation_Campagnes',
        'Influence_Reseaux_Sociaux', 'IST_Diagnostiquee', 'Vaccin_HPV'
    ])
    
    # 30 données de démonstration
    np.random.seed(42)
    for i in range(30):
        demo = pd.DataFrame([{
            'Date': datetime.now().strftime("%Y-%m-%d %H:%M"),
            'Age': int(np.random.randint(18, 65)),
            'Sexe': np.random.choice(["Homme", "Femme"]),
            'Pays': np.random.choice(["Cameroun", "Sénégal", "Côte d'Ivoire", "Nigéria", "Kenya"]),
            'Profession': np.random.choice(["Étudiant", "Employé", "Indépendant", "Fonctionnaire"]),
            'Niveau_Etude': np.random.choice(["Secondaire", "Universitaire", "Supérieur"]),
            'Partenaires_Sexuels': np.random.choice(["Oui", "Non"], p=[0.85,0.15]),
            'Utilisation_Preservatifs': np.random.choice(["Jamais", "Rarement", "Parfois", "Souvent", "Systématiquement"]),
            'Nb_Partenaires': np.random.choice(["1", "2-5", "6-10", "11-20"]),
            'Rapport_Non_Protege': np.random.choice(["Jamais", "Une fois", "Plusieurs fois"]),
            'Alcool_Substances': np.random.choice(["Jamais", "Rarement", "Parfois", "Souvent"]),
            'Connaissance_IST': np.random.choice(["Très mauvaise", "Mauvaise", "Moyenne", "Bonne", "Très bonne"]),
            'Deja_Depiste': np.random.choice(["Jamais", "Une fois", "Plusieurs fois", "Régulièrement"]),
            'Participation_Campagnes': np.random.choice(["Jamais", "Rarement", "Parfois", "Souvent", "Très souvent"]),
            'Influence_Reseaux_Sociaux': np.random.choice(["Négativement", "Neutre", "Positivement"]),
            'IST_Diagnostiquee': np.random.choice(["Non", "Faible", "Modéré", "Élevé"], p=[0.6,0.2,0.1,0.1]),
            'Vaccin_HPV': np.random.choice(["Non", "Oui", "Je ne sais pas"], p=[0.6,0.2,0.2])
        }])
        st.session_state.donnees = pd.concat([st.session_state.donnees, demo], ignore_index=True)

# Sidebar
with st.sidebar:
    st.markdown("**Réalisé par : MADJOU FORTUNE NESLINE - 24G2876**")
    st.markdown("📚 **Programme INF232 EC2**")
    st.markdown("---")
    
    st.header("📝 Formulaire de collecte")
    st.markdown("*Toutes vos réponses sont anonymes*")
    
    with st.form("collecte"):
        st.markdown("### 👤 Votre profil")
        age = st.slider("Âge", 15, 95, 25)
        sexe = st.radio("Sexe", ["Homme", "Femme", "Autre"])
        pays = st.selectbox("Pays", ["Cameroun", "Sénégal", "Côte d'Ivoire", "Nigéria", "Kenya", "Autre"])
        profession = st.selectbox("Profession", ["Étudiant", "Employé", "Indépendant", "Fonctionnaire", "Sans emploi"])
        
        st.markdown("---")
        st.markdown("### 💕 Habitudes")
        utilisation_preservatifs = st.select_slider("Utilisation des préservatifs", options=["Jamais", "Rarement", "Parfois", "Souvent", "Systématiquement"])
        nb_partenaires = st.selectbox("Nombre de partenaires", ["1", "2-5", "6-10", "11-20", "20+"])
        rapport_non_protege = st.selectbox("Avez-vous eu un rapport non protégé ?", ["Jamais", "Une fois", "Plusieurs fois"])
        
        st.markdown("---")
        st.markdown("### 🏥 Connaissance")
        connaissance_ist = st.select_slider("Connaissance des IST", options=["Très mauvaise", "Mauvaise", "Moyenne", "Bonne", "Très bonne"])
        deja_depiste = st.radio("Déjà dépisté ?", ["Jamais", "Une fois", "Plusieurs fois", "Régulièrement"])
        participation_campagnes = st.select_slider("Participation aux campagnes", options=["Jamais", "Rarement", "Parfois", "Souvent", "Très souvent"])
        
        st.markdown("---")
        st.markdown("### 📱 Réseaux sociaux")
        influence_reseaux = st.select_slider("Influence des réseaux sociaux", options=["Négativement", "Neutre", "Positivement"])
        
        submit = st.form_submit_button("✅ Envoyer")
        
        if submit:
            nouvelle = pd.DataFrame([{
                'Date': datetime.now().strftime("%Y-%m-%d %H:%M"),
                'Age': age, 'Sexe': sexe, 'Pays': pays, 'Profession': profession,
                'Niveau_Etude': "Universitaire", 'Partenaires_Sexuels': "Oui",
                'Utilisation_Preservatifs': utilisation_preservatifs, 'Nb_Partenaires': nb_partenaires,
                'Rapport_Non_Protege': rapport_non_protege, 'Alcool_Substances': "Parfois",
                'Connaissance_IST': connaissance_ist, 'Deja_Depiste': deja_depiste,
                'Participation_Campagnes': participation_campagnes, 'Influence_Reseaux_Sociaux': influence_reseaux,
                'IST_Diagnostiquee': "Non renseigné", 'Vaccin_HPV': "Non"
            }])
            st.session_state.donnees = pd.concat([st.session_state.donnees, nouvelle], ignore_index=True)
            st.success("✅ Merci ! Votre réponse est enregistrée.")
            st.balloons()
    
    st.metric("👥 Participants", len(st.session_state.donnees))

# Préparation des données
df = st.session_state.donnees.copy()
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df['Connaissance_num'] = df['Connaissance_IST'].map({'Très mauvaise':1,'Mauvaise':2,'Moyenne':3,'Bonne':4,'Très bonne':5})
df['Preservatifs_num'] = df['Utilisation_Preservatifs'].map({'Jamais':1,'Rarement':2,'Parfois':3,'Souvent':4,'Systématiquement':5})
df['Campagnes_num'] = df['Participation_Campagnes'].map({'Jamais':1,'Rarement':2,'Parfois':3,'Souvent':4,'Très souvent':5})
df['Influence_num'] = df['Influence_Reseaux_Sociaux'].map({'Négativement':1,'Neutre':2,'Positivement':3})
df['Partenaires_num'] = df['Nb_Partenaires'].map({'1':1,'2-5':2,'6-10':3,'11-20':4,'20+':5})

# Score de risque
df['Score_Risque'] = (
    (6 - df['Preservatifs_num']) * 2 +
    df['Partenaires_num'] * 1.5 +
    df['Rapport_Non_Protege'].map({'Jamais':0, 'Une fois':1, 'Plusieurs fois':2}) * 2 +
    (df['Connaissance_num'] < 3).astype(int) * 2
)
df['Categorie_Risque'] = df['Score_Risque'].apply(lambda x: 'Faible' if x <= 8 else ('Modéré' if x <= 15 else 'Élevé'))

df_clean = df.dropna(subset=['Age', 'Connaissance_num', 'Preservatifs_num', 'Campagnes_num', 'Partenaires_num'])

# Onglets
tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📋 Participants", "📈 Régression simple", "🔬 Régression multiple", 
    "🎯 PCA", "🏷️ Classification (Risque IST)", "🔄 Clustering", "📊 Graphiques", "📚 Prévention"
])

# ============================================================
# TAB 0 : PARTICIPANTS
# ============================================================
with tab0:
    st.header("📋 Participants à l'étude")
    st.dataframe(df_clean, use_container_width=True)
    csv = df_clean.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Télécharger les données (CSV)", csv, "donnees_ist.csv", "text/csv")

# ============================================================
# TAB 1 : RÉGRESSION SIMPLE
# ============================================================
with tab1:
    st.header("📈 Régression simple : Âge → Connaissance des IST")
    st.markdown("**Objectif :** Vérifier si l'âge influence le niveau de connaissance des IST.")
    
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
        **📖 Interprétation :**
        - Chaque point représente un participant
        - La ligne rouge montre la tendance générale
        - Les couleurs indiquent le niveau de risque IST estimé
        - R² proche de 1 = bonne prédiction
        """)
    else:
        st.warning(f"⚠️ Besoin d'au moins 3 participants. Actuellement : {len(df_clean)}")

# ============================================================
# TAB 2 : RÉGRESSION MULTIPLE
# ============================================================
with tab2:
    st.header("🔬 Régression multiple : Facteurs influençant la connaissance des IST")
    st.markdown("**Objectif :** Identifier quels comportements sont liés à une meilleure connaissance des IST.")
    
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
        **📖 Interprétation :**
        - Un coefficient POSITIF = plus le facteur augmente, meilleure est la connaissance
        - Un coefficient NÉGATIF = plus le facteur augmente, moins bonne est la connaissance
        - Les points proches de la ligne rouge indiquent une bonne prédiction
        """)
    else:
        st.warning(f"⚠️ Besoin d'au moins 5 participants. Actuellement : {len(df_clean)}")

# ============================================================
# TAB 3 : PCA
# ============================================================
with tab3:
    st.header("🎯 Analyse en Composantes Principales (PCA)")
    st.markdown("**Objectif :** Visualiser les profils similaires dans un espace réduit à 2 dimensions.")
    
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
        **📖 Interprétation :**
        - Les POINTS PROCHES ont des comportements similaires
        - Les COULEURS indiquent le niveau de risque IST
        - Plus la variance expliquée est élevée, plus la projection est fidèle
        """)
    else:
        st.warning(f"⚠️ Besoin d'au moins 4 participants. Actuellement : {len(df_clean)}")

# ============================================================
# TAB 4 : CLASSIFICATION (PRÉDICTION DU RISQUE)
# ============================================================
with tab4:
    st.header("🏷️ Classification : Prédire son risque de contracter une IST")
    st.markdown("**Objectif :** Le modèle apprend à estimer votre niveau de risque selon vos habitudes.")
    
    if len(df_clean) >= 6:
        df_clean['Cible_Risque'] = (df_clean['Categorie_Risque'] == 'Élevé').astype(int)
        
        X = df_clean[['Age', 'Preservatifs_num', 'Partenaires_num', 'Campagnes_num', 'Connaissance_num']].values
        y = df_clean['Cible_Risque'].values
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=4)
        rf.fit(X, y)
        
        # Importance des facteurs
        importance_df = pd.DataFrame({
            'Facteur': ['Âge', 'Préservatifs', 'Nombre de partenaires', 'Participation campagnes', 'Connaissance IST'],
            'Importance (%)': (rf.feature_importances_ * 100).round(1)
        }).sort_values('Importance (%)', ascending=False)
        st.dataframe(importance_df, use_container_width=True)
        
        fig_imp = px.bar(importance_df, x='Importance (%)', y='Facteur', orientation='h',
                         title="Facteurs influençant le risque IST")
        st.plotly_chart(fig_imp, use_container_width=True)
        
        # Section test interactif
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
            # Conversion
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
        
        # Précision du modèle
        try:
            scores = cross_val_score(rf, X, y, cv=min(3, len(np.unique(y))))
            st.caption(f"📊 Précision du modèle : {scores.mean():.1%} (basé sur les données existantes)")
        except:
            pass
    else:
        st.warning(f"⚠️ Besoin d'au moins 6 participants. Actuellement : {len(df_clean)}")

# ============================================================
# TAB 5 : CLUSTERING
# ============================================================
with tab5:
    st.header("🔄 Clustering : Segmentation automatique des profils")
    st.markdown("**Objectif :** Regrouper automatiquement les personnes aux habitudes similaires.")
    
    if len(df_clean) >= 5:
        features_cluster = ['Age', 'Connaissance_num', 'Preservatifs_num', 'Partenaires_num']
        scaler = StandardScaler()
        X_cluster = scaler.fit_transform(df_clean[features_cluster])
        
        k = st.slider("Nombre de segments (clusters)", 2, 4, 3, 
                      help="Plus il y a de segments, plus la segmentation est fine")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_cluster)
        df_clean['Segment'] = clusters
        
        fig = px.scatter(df_clean, x='Age', y='Connaissance_num', 
                         color=clusters.astype(str), size='Preservatifs_num',
                         title=f"Segmentation des participants en {k} groupes",
                         hover_data=['Profession', 'Categorie_Risque'])
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("📊 Profil type de chaque segment")
        profil = df_clean.groupby('Segment')[['Age', 'Connaissance_num', 'Preservatifs_num', 'Partenaires_num']].mean().round(1)
        profil.columns = ['Âge moyen', 'Connaissance (1-5)', 'Préservatifs (1-5)', 'Nombre de partenaires']
        st.dataframe(profil)
        
        st.info("""
        **📖 Interprétation :**
        - Chaque couleur représente un groupe aux habitudes similaires
        - Le tableau montre le profil type de chaque groupe
        - Identifiez à quel groupe vous ressemblez pour mieux cibler vos besoins
        """)
    else:
        st.warning(f"⚠️ Besoin d'au moins 5 participants. Actuellement : {len(df_clean)}")

# ============================================================
# TAB 6 : GRAPHIQUES STATISTIQUES
# ============================================================
with tab6:
    st.header("📊 Graphiques statistiques descriptifs")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 1. Distribution des âges (Histogramme)")
        fig_hist = px.histogram(df_clean, x='Age', nbins=15, title="Âges des participants")
        st.plotly_chart(fig_hist, use_container_width=True)
        
        st.subheader("📊 2. Niveau de connaissance des IST")
        connais_counts = df_clean['Connaissance_IST'].value_counts().reset_index()
        connais_counts.columns = ['Niveau', 'Nombre']
        fig_bar = px.bar(connais_counts, x='Niveau', y='Nombre', title="Auto-évaluation")
        st.plotly_chart(fig_bar, use_container_width=True)
        
        st.subheader("📊 3. Répartition par sexe")
        fig_pie = px.pie(df_clean, names='Sexe', title="Hommes / Femmes")
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("📊 4. Répartition par profession")
        prof_counts = df_clean['Profession'].value_counts().reset_index()
        prof_counts.columns = ['Profession', 'Nombre']
        fig_barh = px.bar(prof_counts, x='Nombre', y='Profession', orientation='h', title="Participants par profession")
        st.plotly_chart(fig_barh, use_container_width=True)
        
        st.subheader("📊 5. Utilisation des préservatifs")
        preserv_counts = df_clean['Utilisation_Preservatifs'].value_counts().reset_index()
        preserv_counts.columns = ['Fréquence', 'Nombre']
        fig_preserv = px.bar(preserv_counts, x='Fréquence', y='Nombre', title="Fréquence d'utilisation")
        st.plotly_chart(fig_preserv, use_container_width=True)
        
        st.subheader("📊 6. Répartition par niveau de risque")
        fig_risk = px.pie(df_clean, names='Categorie_Risque', title="Risque IST estimé")
        st.plotly_chart(fig_risk, use_container_width=True)

# ============================================================
# TAB 7 : PRÉVENTION
# ============================================================
with tab7:
    st.header("📚 Espace Prévention & Éducation")
    
    st.markdown("### 🚨 Symptômes évocateurs - Consultez immédiatement")
    st.markdown("""
    - Écoulements anormaux (urètre, vagin, anus)
    - Douleurs ou brûlures en urinant
    - Lésions, boutons, ulcères ou verrues sur les organes génitaux
    - Démangeaisons intenses ou irritations persistantes
    - Ganglions anormalement gonflés dans l'aine
    - Fièvre inexpliquée accompagnée d'autres symptômes
    """)
    
    st.warning("⚠️ **IMPORTANT** : Certaines IST sont asymptomatiques (sans symptômes visibles). Le seul moyen d'être sûr(e) de son statut est le **DÉPISTAGE RÉGULIER**.")
    
    st.info("📢 *Ces informations ne remplacent pas l'avis d'un médecin. Consultez un professionnel de santé pour tout diagnostic.*")
    
    st.markdown("### 🛡️ Les moyens de prévention efficaces")
    st.markdown("""
    - **Préservatifs masculins et féminins** : Protection contre la plupart des IST
    - **Dépistage régulier** : Au moins 2 fois par an si vie sexuelle active
    - **Vaccination** : Contre le HPV (Papillomavirus) et l'Hépatite B
    - **Communication** : Parler de son statut avec le/la partenaire
    """)
    
    st.markdown("### 📍 Où se faire dépister ?")
    
    with st.expander("🇨🇲 CAMEROUN"):
        st.markdown("""
        - **Hôpitaux publics** : Hôpital Général de Yaoundé, Hôpital Laquintinie (Douala)
        - **Centres de santé communautaires** : Dans tous les arrondissements
        - **Associations** : ASES (Association Santé Équitable), IRESCO
        """)
    
    with st.expander("🇸🇳 SÉNÉGAL"):
        st.markdown("""
        - **Hôpital de Fann** (Dakar) - Service des maladies infectieuses
        - **ALCS (Association de Lutte contre le Sida)** - Dépistage gratuit
        - **Centres de santé régionaux** : Thiès, Saint-Louis, Ziguinchor
        """)
    
    with st.expander("🇨🇮 CÔTE D'IVOIRE"):
        st.markdown("""
        - **INHP (Institut National d'Hygiène Publique)** - Abidjan
        - **Centre de santé de Treichville**
        - **Association Ivoirienne pour le Bien-être Familial (AIBEF)**
        """)
    
    with st.expander("🌍 AUTRES PAYS"):
        st.markdown("""
        - Hôpitaux publics et centres de santé de district
        - Organisations internationales : Croix-Rouge, Médecins du Monde
        - Lignes d'écoute nationales (numéros verts)
        """)

# Footer
st.markdown("---")
st.markdown("📌 **Application développée par MADJOU FORTUNE NESLINE (24G2876) - Programme INF232 EC2**")