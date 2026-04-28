import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

# Configuration
st.set_page_config(page_title="NEXHEALTH SURVEY NO TABOO", page_icon="🩺", layout="wide")

# Style CSS
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #f5f7fa 0%, #e8f5e9 100%); }
    .slogan {
        background: linear-gradient(90deg, #1b5e20, #2e7d32);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 25px;
    }
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background: linear-gradient(90deg, #1b5e20, #2e7d32);
        color: white;
        text-align: center;
        padding: 8px;
        font-size: 0.8rem;
        z-index: 999;
    }
    .main-title { text-align: center; margin-bottom: 20px; }
    .main-title h1 { color: #1b5e20; font-weight: bold; }
    .prevention-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border-left: 5px solid #2e7d32;
    }
    .interpretation {
        background: #e3f2fd;
        padding: 15px;
        border-radius: 10px;
        margin: 15px 0;
        border-left: 5px solid #2196f3;
    }
</style>
""", unsafe_allow_html=True)

# Titre et slogan
st.markdown("<h1 style='text-align:center; color:#1b5e20;'>🩺 NEXHEALTH SURVEY NO TABOO</h1>", unsafe_allow_html=True)
st.markdown("""
<div class="slogan">
    <p><i><strong>✨ Parce qu'en santé, il n'y a pas de tabou. ✨</strong></i></p>
    <p><i><strong>🔍 Brisons le silence sur les IST, protégeons notre bien-être.</strong></i></p>
    <p><i><strong>⚖️ La santé sexuelle est un droit, la protection est une responsabilité.</strong></i></p>
</div>
""", unsafe_allow_html=True)

# Initialisation des données
if 'donnees' not in st.session_state:
    st.session_state.donnees = pd.DataFrame(columns=[
        'Date', 'Age', 'Sexe', 'Pays', 'Profession', 'Niveau_Etude',
        'Partenaires_Sexuels', 'Utilisation_Preservatifs', 'Nb_Partenaires',
        'Rapport_Non_Protege', 'Alcool_Substances',
        'Connaissance_IST', 'Deja_Depiste', 'Participation_Campagnes',
        'Influence_Reseaux_Sociaux', 'IST_Diagnostiquee', 'Vaccin_HPV'
    ])
    
    # 30 données démo
    np.random.seed(42)
    for i in range(30):
        risque = np.random.choice(["Faible", "Modéré", "Élevé"], p=[0.5, 0.3, 0.2])
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
            'IST_Diagnostiquee': risque,
            'Vaccin_HPV': np.random.choice(["Non", "Oui", "Je ne sais pas"], p=[0.6,0.2,0.2])
        }])
        st.session_state.donnees = pd.concat([st.session_state.donnees, demo], ignore_index=True)

# Sidebar
with st.sidebar:
    st.markdown("### 👩‍💻 Réalisé par")
    st.markdown("**MADJOU FORTUNE NESLINE - 24G2876**")
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
        niveau_etude = st.selectbox("Niveau d'étude", ["Secondaire", "Universitaire", "Supérieur", "Aucun"])
        
        st.markdown("---")
        st.markdown("### 💕 Habitudes")
        utilisation_preservatifs = st.select_slider("Utilisation des préservatifs", options=["Jamais", "Rarement", "Parfois", "Souvent", "Systématiquement"])
        nb_partenaires = st.selectbox("Nombre de partenaires", ["1", "2-5", "6-10", "11-20", "20+"])
        rapport_non_protege = st.selectbox("Rapport non protégé", ["Jamais", "Une fois", "Plusieurs fois"])
        
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
                'Niveau_Etude': niveau_etude, 'Partenaires_Sexuels': "Oui",
                'Utilisation_Preservatifs': utilisation_preservatifs, 'Nb_Partenaires': nb_partenaires,
                'Rapport_Non_Protege': rapport_non_protege, 'Alcool_Substances': "Parfois",
                'Connaissance_IST': connaissance_ist, 'Deja_Depiste': deja_depiste,
                'Participation_Campagnes': participation_campagnes, 'Influence_Reseaux_Sociaux': influence_reseaux,
                'IST_Diagnostiquee': "Non renseigné", 'Vaccin_HPV': "Non"
            }])
            st.session_state.donnees = pd.concat([st.session_state.donnees, nouvelle], ignore_index=True)
            st.success("✅ Enregistré !")
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
df['Risque_IST'] = df['IST_Diagnostiquee'].map({'Non':0, 'Faible':0, 'Modéré':1, 'Élevé':2, 'Oui, guérie':1, 'Non renseigné':0})

df_clean = df.dropna(subset=['Age', 'Connaissance_num', 'Preservatifs_num', 'Campagnes_num', 'Partenaires_num'])

# Calcul d'un score de risque pour la prédiction
df_clean['Score_Risque'] = (
    (6 - df_clean['Preservatifs_num']) * 2 +  # Moins de préservatifs = risque +
    (df_clean['Partenaires_num']) * 1.5 +      # Plus de partenaires = risque +
    (df_clean['Rapport_Non_Protege'].map({'Jamais':0, 'Une fois':1, 'Plusieurs fois':2})) * 2 +
    (df_clean['Connaissance_num'] < 3).astype(int) * 2
)
df_clean['Categorie_Risque'] = pd.cut(df_clean['Score_Risque'], bins=[0, 8, 15, 100], labels=['Faible', 'Modéré', 'Élevé'])

# Onglets
tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📋 Participants", "📈 Régression simple", "🔬 Régression multiple", 
    "🎯 PCA", "🏷️ Classification", "🔄 Clustering", "📊 Graphiques", "📚 Prévention"
])

# ========== TAB 0 ==========
with tab0:
    st.header("📋 Participants")
    st.dataframe(df_clean, use_container_width=True)
    csv = df_clean.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Télécharger CSV", csv, "donnees_ist.csv")

# ========== TAB 1 : RÉGRESSION SIMPLE ==========
with tab1:
    st.header("📈 Âge → Connaissance des IST")
    
    if len(df_clean) >= 3:
        X = df_clean[['Age']].values
        y = df_clean['Connaissance_num'].values
        modele = LinearRegression().fit(X, y)
        
        fig = px.scatter(df_clean, x='Age', y='Connaissance_num', 
                         title="Relation entre l'âge et la connaissance des IST",
                         color=df_clean['Categorie_Risque'],
                         hover_data=['Profession'])
        x_range = np.linspace(df_clean['Age'].min(), df_clean['Age'].max(), 100)
        y_pred = modele.predict(x_range.reshape(-1, 1))
        fig.add_trace(go.Scatter(x=x_range, y=y_pred, mode='lines', 
                                name='Tendance', line=dict(color='red', width=3)))
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📐 Coefficient", f"{modele.coef_[0]:.3f}")
            if modele.coef_[0] > 0:
                st.caption("✅ Plus on est âgé, meilleure est la connaissance des IST")
            else:
                st.caption("⚠️ Les jeunes ont une meilleure connaissance des IST")
        with col2:
            r2 = r2_score(y, modele.predict(X))
            st.metric("🎯 R²", f"{r2:.3f}")
            if r2 > 0.5:
                st.caption("✅ Bon pouvoir prédictif de l'âge")
            else:
                st.caption("⚠️ L'âge explique peu les différences de connaissance")
        
        st.markdown("""
        <div class="interpretation">
        <b>📖 Interprétation :</b><br>
        - Chaque point représente un participant<br>
        - La ligne rouge montre la tendance générale<br>
        - Plus le R² est proche de 1, plus l'âge est un bon prédicteur<br>
        - Les couleurs indiquent le niveau de risque IST estimé
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Ajoutez au moins 3 participants")

# ========== TAB 2 : RÉGRESSION MULTIPLE ==========
with tab2:
    st.header("🔬 Facteurs influençant la connaissance des IST")
    
    if len(df_clean) >= 5:
        features = ['Age', 'Preservatifs_num', 'Partenaires_num', 'Campagnes_num']
        X = df_clean[features].values
        y = df_clean['Connaissance_num'].values
        
        modele = LinearRegression().fit(X, y)
        
        coef_df = pd.DataFrame({
            'Facteur': ['Âge', 'Utilisation préservatifs', 'Nombre de partenaires', 'Participation campagnes'],
            'Impact': modele.coef_,
            'Interprétation': [
                f"{'⬆️ Améliore' if c>0 else '⬇️ Diminue'} la connaissance" for c in modele.coef_
            ]
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
        
        st.markdown("""
        <div class="interpretation">
        <b>📖 Interprétation :</b><br>
        - Un impact POSITIF signifie que plus le facteur augmente, meilleure est la connaissance<br>
        - Un impact NÉGATIF signifie l'inverse<br>
        - Les points proches de la ligne rouge indiquent une bonne prédiction<br>
        - <b>Conseil :</b> Pour améliorer la connaissance des IST, concentrez-vous sur les facteurs avec impact positif
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Ajoutez au moins 5 participants")

# ========== TAB 3 : PCA ==========
with tab3:
    st.header("🎯 Visualisation des profils (PCA)")
    
    if len(df_clean) >= 4:
        features = ['Age', 'Connaissance_num', 'Preservatifs_num', 'Partenaires_num', 'Campagnes_num']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_clean[features])
        pca = PCA(n_components=2)
        result = pca.fit_transform(X_scaled)
        
        df_viz = pd.DataFrame({'PC1': result[:,0], 'PC2': result[:,1], 
                               'Risque': df_clean['Categorie_Risque'],
                               'Âge': df_clean['Age']})
        
        fig = px.scatter(df_viz, x='PC1', y='PC2', color='Risque', size='Âge',
                         title="Projection des profils (plus ils sont proches, plus ils se ressemblent)",
                         labels={'PC1': f'Dimension 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                                'PC2': f'Dimension 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)'})
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div class="interpretation">
        <b>📖 Interprétation :</b><br>
        - Les POINTS PROCHES ont des comportements similaires face aux IST<br>
        - Les COULEURS indiquent le niveau de risque estimé<br>
        - Plus la variance expliquée est élevée, plus la projection est fidèle à la réalité<br>
        - <b>Observation :</b> Les points de même couleur ont tendance à se regrouper
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Ajoutez au moins 4 participants")

# ========== TAB 4 : CLASSIFICATION - PRÉDICTION DU RISQUE ==========
with tab4:
    st.header("🏷️ Prédire son risque de contracter une IST")
    st.markdown("**Ce modèle analyse vos habitudes pour estimer votre niveau de risque IST**")
    
    if len(df_clean) >= 6:
        # Création de la cible (Élevé vs Faible/Modéré)
        df_clean['Cible_Risque'] = (df_clean['Categorie_Risque'] == 'Élevé').astype(int)
        
        X = df_clean[['Age', 'Preservatifs_num', 'Partenaires_num', 
                      'Campagnes_num', 'Connaissance_num']].values
        y = df_clean['Cible_Risque'].values
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=4)
        rf.fit(X, y)
        
        # Importance des facteurs
        importance_df = pd.DataFrame({
            'Facteur': ['Âge', 'Utilisation préservatifs', 'Nombre de partenaires', 
                       'Participation campagnes', 'Connaissance IST'],
            'Importance': rf.feature_importances_,
            'Poids': (rf.feature_importances_ * 100).round(1)
        }).sort_values('Importance', ascending=False)
        
        fig_imp = px.bar(importance_df, x='Importance', y='Facteur', orientation='h',
                         title="Quels facteurs influencent le plus le risque IST ?",
                         text='Poids', color='Importance')
        fig_imp.update_traces(texttemplate='%{text}%', textposition='outside')
        st.plotly_chart(fig_imp, use_container_width=True)
        
        # Test interactif
        st.subheader("🔮 Évaluez VOTRE risque personnel")
        st.markdown("Renseignez vos habitudes ci-dessous pour obtenir une estimation personnalisée.")
        
        col1, col2 = st.columns(2)
        with col1:
            age_test = st.slider("Votre âge", 18, 65, 25, key="risk_age")
            preserv_test = st.select_slider("Utilisation des préservatifs", 
                                           options=["Systématiquement", "Souvent", "Parfois", "Rarement", "Jamais"],
                                           key="risk_preserv")
            nb_partenaires_test = st.select_slider("Nombre de partenaires dans l'année",
                                                   options=["1", "2-5", "6-10", "11-20", "20+"],
                                                   key="risk_partenaires")
        with col2:
            campagnes_test = st.select_slider("Participation aux campagnes de dépistage",
                                             options=["Très souvent", "Souvent", "Parfois", "Rarement", "Jamais"],
                                             key="risk_campagnes")
            connais_test = st.select_slider("Connaissance des IST",
                                           options=["Très bonne", "Bonne", "Moyenne", "Mauvaise", "Très mauvaise"],
                                           key="risk_connais")
        
        # Conversion pour la prédiction
        preserv_map = {"Systématiquement":5, "Souvent":4, "Parfois":3, "Rarement":2, "Jamais":1}
        campagnes_map = {"Très souvent":5, "Souvent":4, "Parfois":3, "Rarement":2, "Jamais":1}
        connais_map = {"Très bonne":5, "Bonne":4, "Moyenne":3, "Mauvaise":2, "Très mauvaise":1}
        partenaires_map = {"1":1, "2-5":2, "6-10":3, "11-20":4, "20+":5}
        
        if st.button("🔮 Estimer mon risque de contracter une IST", key="predict_risk"):
            preserv_val = preserv_map[preserv_test]
            camp_val = campagnes_map[campagnes_test]
            connais_val = connais_map[connais_test]
            partenaires_val = partenaires_map[nb_partenaires_test]
            
            pred = rf.predict([[age_test, preserv_val, partenaires_val, camp_val, connais_val]])[0]
            proba = rf.predict_proba([[age_test, preserv_val, partenaires_val, camp_val, connais_val]]).max()
            
            if pred == 1:
                st.error(f"⚠️ **Risque ÉLEVÉ de contracter une IST** (confiance : {proba:.1%})")
                st.markdown("""
                **💡 Recommandations :**
                - ✅ Augmentez l'utilisation des préservatifs
                - ✅ Réduisez le nombre de partenaires
                - ✅ Faites-vous dépister régulièrement (2x/an)
                - ✅ Participez aux campagnes de sensibilisation
                - ✅ Améliorez vos connaissances sur les IST
                """)
            else:
                st.success(f"✅ **Risque FAIBLE à MODÉRÉ** (confiance : {proba:.1%})")
                st.markdown("""
                **💡 Pour rester protégé(e) :**
                - ✅ Continuez les bonnes pratiques
                - ✅ Maintenez un dépistage régulier
                - ✅ Sensibilisez votre entourage
                """)
            
            # Afficher les facteurs de risque
            st.markdown("---")
            st.markdown("**📊 Analyse personnalisée :**")
            if preserv_val <= 2:
                st.warning("⚠️ L'utilisation irrégulière des préservatifs augmente votre risque")
            if partenaires_val >= 4:
                st.warning("⚠️ Le nombre élevé de partenaires augmente votre risque")
            if connais_val <= 2:
                st.warning("⚠️ Améliorer votre connaissance des IST réduirait votre risque")
            if camp_val <= 2:
                st.info("ℹ️ Participer aux campagnes de dépistage vous aiderait à mieux vous protéger")
        
        # Précision du modèle
        try:
            scores = cross_val_score(rf, X, y, cv=min(3, len(np.unique(y))))
            st.caption(f"📊 Précision du modèle : {scores.mean():.1%} (basé sur les données existantes)")
        except:
            pass
    else:
        st.warning(f"Ajoutez au moins 6 participants ({len(df_clean)} actuellement)")

# ========== TAB 5 : CLUSTERING ==========
with tab5:
    st.header("🔄 Segmentation automatique des profils")
    
    if len(df_clean) >= 5:
        X = df_clean[['Age', 'Connaissance_num', 'Preservatifs_num', 'Partenaires_num']].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        k = st.slider("Nombre de segments", 2, 4, 3, help="Plus il y a de segments, plus la segmentation est fine")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        df_clean_loc = df_clean.copy()
        df_clean_loc['Segment'] = clusters
        
        fig = px.scatter(df_clean_loc, x='Age', y='Connaissance_num', 
                         color=clusters.astype(str), size='Preservatifs_num',
                         title=f"Groupes de personnes aux comportements similaires ({k} segments)",
                         hover_data=['Profession', 'Partenaires_num'])
        st.plotly_chart(fig, use_container_width=True)
        
        profil = df_clean_loc.groupby('Segment')[['Age', 'Connaissance_num', 'Preservatifs_num', 'Partenaires_num']].mean()
        profil.columns = ['Âge moyen', 'Connaissance (1-5)', 'Préservatifs (1-5)', 'Partenaires']
        st.dataframe(profil.style.background_gradient(cmap='Blues'))
        
        st.markdown("""
        <div class="interpretation">
        <b>📖 Interprétation :</b><br>
        - Chaque couleur représente un groupe de personnes aux habitudes similaires<br>
        - Le tableau montre le profil type de chaque groupe<br>
        - Identifiez à quel groupe vous ressemblez pour mieux cibler vos besoins de prévention
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Ajoutez au moins 5 participants")

# ========== TAB 6 : GRAPHIQUES ==========
with tab6:
    st.header("📊 Analyses descriptives")
    col1, col2 = st.columns(2)
    with col1:
        fig_hist = px.histogram(df_clean, x='Age', nbins=15, title="Distribution des âges")
        st.plotly_chart(fig_hist, use_container_width=True)
        
        connais_counts = df_clean['Connaissance_IST'].value_counts().reset_index()
        connais_counts.columns = ['Niveau', 'Nombre']
        fig_bar = px.bar(connais_counts, x='Niveau', y='Nombre', title="Niveau de connaissance")
        st.plotly_chart(fig_bar, use_container_width=True)
        
        fig_pie = px.pie(df_clean, names='Categorie_Risque', title="Répartition par risque IST")
        st.plotly_chart(fig_pie, use_container_width=True)
    with col2:
        prof_counts = df_clean['Profession'].value_counts().reset_index()
        prof_counts.columns = ['Profession', 'Nombre']
        fig_barh = px.bar(prof_counts, x='Nombre', y='Profession', orientation='h', title="Par profession")
        st.plotly_chart(fig_barh, use_container_width=True)
        
        preserv_counts = df_clean['Utilisation_Preservatifs'].value_counts().reset_index()
        preserv_counts.columns = ['Fréquence', 'Nombre']
        fig_preserv = px.bar(preserv_counts, x='Fréquence', y='Nombre', title="Utilisation préservatifs")
        st.plotly_chart(fig_preserv, use_container_width=True)

# ========== TAB 7 : PRÉVENTION ==========
with tab7:
    st.header("📚 Prévention & Éducation")
    st.markdown("""
    <div class="prevention-card">
        <h3>🚨 Signes d'alerte</h3>
        <ul><li>Écoulements anormaux</li><li>Douleurs en urinant</li><li>Lésions/verrues</li></ul>
        <b>⚠️ IST souvent asymptomatiques → Dépistage régulier indispensable</b>
    </div>
    <div class="prevention-card">
        <h3>🛡️ Protection</h3>
        <ul><li>Préservatifs (masculins/féminins)</li><li>Dépistage régulier (2x/an)</li><li>Vaccination HPV/Hépatite B</li></ul>
    </div>
    <div class="prevention-card">
        <h3>📍 Dépistage</h3>
        <b>Cameroun :</b> Hôpital Général Yaoundé, Hôpital Laquintinie Douala, ASES<br>
        <b>Sénégal :</b> Hôpital Fann Dakar, ALCS<br>
        <b>Côte d'Ivoire :</b> INHP Abidjan, Centre Treichville
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p>📌 MADJOU FORTUNE NESLINE (24G2876) - INF232 EC2 - NEXHEALTH SURVEY NO TABOO</p>
</div>
""", unsafe_allow_html=True)