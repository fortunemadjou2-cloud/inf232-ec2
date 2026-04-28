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

# Configuration de la page
st.set_page_config(page_title="INF232 EC2 - Analyse Numérique Afrique", layout="wide")

st.title("🌍 Analyse des Habitudes Numériques en Afrique")
st.markdown("---")

# Initialisation des données
if 'donnees' not in st.session_state:
    st.session_state.donnees = pd.DataFrame(columns=[
        'Date', 'Age', 'Pays', 'Profession', 'Heures_Internet', 
        'Nb_Reseaux_Sociaux', 'Achats_En_Ligne', 'Streaming', 'Budget_Data'
    ])

# ==================== FORMULAIRE DE COLLECTE ====================
with st.sidebar:
    st.header("📝 Formulaire de collecte")
    st.markdown("---")
    
    with st.form("collecte"):
        age = st.slider("Âge", 15, 80, 25)
        pays = st.selectbox("Pays", ["Sénégal", "Côte d'Ivoire", "Nigéria", "Kenya", "Ghana", "Maroc"])
        profession = st.selectbox("Profession", ["Étudiant", "Employé", "Indépendant", "Sans emploi"])
        heures = st.slider("Heures par jour sur Internet", 0, 16, 4)
        nb_reseaux = st.slider("Nombre de réseaux sociaux utilisés", 0, 7, 3)
        achats = st.select_slider("Fréquence des achats en ligne", options=["Jamais", "Rarement", "Parfois", "Souvent"])
        streaming = st.select_slider("Utilisation du streaming", options=["Jamais", "Rarement", "Parfois", "Souvent"])
        budget = st.number_input("Budget data mensuel (FCFA)", 0, 50000, 5000)
        
        submit = st.form_submit_button("✅ Envoyer ma participation")
        
        if submit:
            nouvelle_ligne = pd.DataFrame([{
                'Date': datetime.now().strftime("%Y-%m-%d %H:%M"),
                'Age': age,
                'Pays': pays,
                'Profession': profession,
                'Heures_Internet': heures,
                'Nb_Reseaux_Sociaux': nb_reseaux,
                'Achats_En_Ligne': ["Jamais", "Rarement", "Parfois", "Souvent"].index(achats),
                'Streaming': ["Jamais", "Rarement", "Parfois", "Souvent"].index(streaming),
                'Budget_Data': budget
            }])
            st.session_state.donnees = pd.concat([st.session_state.donnees, nouvelle_ligne], ignore_index=True)
            st.success("✅ Enregistré !")
            st.balloons()
    
    st.metric("👥 Participants", len(st.session_state.donnees))

# Vérifier qu'il y a assez de données
if len(st.session_state.donnees) < 3:
    st.warning("⚠️ Besoin d'au moins 3 participants pour les analyses. Remplissez le formulaire à gauche !")
    st.stop()

df = st.session_state.donnees.copy()

# ==================== ONGLETS D'ANALYSE ====================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Régression simple", "🔬 Régression multiple", "🎯 PCA (réduction dim.)",
    "🏷️ Classification supervisée", "🔄 Classification non-supervisée", "📊 Analyse descriptive"
])

# ----- TAB 1 : RÉGRESSION LINÉAIRE SIMPLE -----
with tab1:
    st.header("📈 Régression simple : Âge → Heures internet")
    
    X = df[['Age']].values
    y = df['Heures_Internet'].values
    
    modele = LinearRegression()
    modele.fit(X, y)
    
    fig = px.scatter(df, x='Age', y='Heures_Internet', title="Âge vs Heures Internet")
    
    x_range = np.linspace(df['Age'].min(), df['Age'].max(), 100)
    y_pred = modele.predict(x_range.reshape(-1, 1))
    fig.add_trace(go.Scatter(x=x_range, y=y_pred, mode='lines', name='Régression', line=dict(color='red')))
    
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Coefficient", f"{modele.coef_[0]:.2f}")
        st.metric("Ordonnée origine", f"{modele.intercept_:.2f}")
    with col2:
        r2 = r2_score(y, modele.predict(X))
        st.metric("Qualité R²", f"{r2:.3f}")
        st.caption("R² proche de 1 = bonne prédiction")

# ----- TAB 2 : RÉGRESSION MULTIPLE -----
with tab2:
    st.header("🔬 Régression multiple : Prédiction du budget data")
    
    features = ['Age', 'Heures_Internet', 'Nb_Reseaux_Sociaux', 'Streaming']
    X_multi = df[features].values
    y_multi = df['Budget_Data'].values
    
    modele_multi = LinearRegression()
    modele_multi.fit(X_multi, y_multi)
    
    coef_df = pd.DataFrame({'Variable': features, 'Coefficient': modele_multi.coef_})
    st.dataframe(coef_df)
    
    predictions = modele_multi.predict(X_multi)
    fig = px.scatter(x=y_multi, y=predictions, title="Prédictions vs Réalité")
    fig.add_trace(go.Scatter(x=[y_multi.min(), y_multi.max()], y=[y_multi.min(), y_multi.max()], 
                            mode='lines', name='Parfait', line=dict(dash='dash')))
    st.plotly_chart(fig, use_container_width=True)
    
    st.metric("R² du modèle", f"{r2_score(y_multi, predictions):.3f}")

# ----- TAB 3 : PCA (Réduction de dimensionnalité) -----
with tab3:
    st.header("🎯 PCA : Visualisation 2D des profils")
    
    features_pca = ['Age', 'Heures_Internet', 'Nb_Reseaux_Sociaux', 'Streaming', 'Budget_Data']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features_pca])
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    
    df_pca = pd.DataFrame({'PC1': pca_result[:, 0], 'PC2': pca_result[:, 1], 'Pays': df['Pays']})
    fig = px.scatter(df_pca, x='PC1', y='PC2', color='Pays', title="Projection PCA")
    
    fig.update_layout(
        xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
        yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("PCA réduit les dimensions de 5 variables à 2 pour visualiser les similarités entre profils.")

# ----- TAB 4 : CLASSIFICATION SUPERVISÉE -----
with tab4:
    st.header("🏷️ Random Forest : Prédire la profession")
    
    X_class = df[['Age', 'Heures_Internet', 'Budget_Data']].values
    y_class = df['Profession'].values
    
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X_class, y_class)
    
    # Vérifier qu'il y a assez de classes pour la validation croisée
    n_classes = len(df['Profession'].unique())
    if n_classes >= 3:
        cv_value = min(3, n_classes)
        scores = cross_val_score(rf, X_class, y_class, cv=cv_value)
        st.metric("Précision du modèle", f"{scores.mean():.1%}")
    else:
        st.info(f"ℹ️ Précision du modèle : {n_classes} profession(s) détectée(s). Ajoutez plus de professions variées.")
    
    st.subheader("🔮 Testez le modèle")
    age_test = st.slider("Âge", 15, 80, 30, key="test_age")
    heures_test = st.slider("Heures internet", 0, 16, 5, key="test_heures")
    budget_test = st.slider("Budget data", 0, 50000, 10000, key="test_budget")
    
    if st.button("Prédire la profession"):
        prediction = rf.predict([[age_test, heures_test, budget_test]])[0]
        # Calculer la confiance
        proba = rf.predict_proba([[age_test, heures_test, budget_test]]).max()
        st.success(f"🏆 Profession prédite : **{prediction}** (confiance : {proba:.1%})")

# ----- TAB 5 : CLASSIFICATION NON-SUPERVISÉE (K-Means) -----
with tab5:
    st.header("🔄 K-Means : Segmentation des utilisateurs")
    
    features_cluster = ['Age', 'Heures_Internet', 'Budget_Data']
    scaler_cluster = StandardScaler()
    X_cluster = scaler_cluster.fit_transform(df[features_cluster])
    
    k = st.slider("Nombre de segments (clusters)", 2, 4, 3)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_cluster)
    
    df['Segment'] = clusters
    
    # S'assurer que Age est numérique
    age_numeric = pd.to_numeric(df['Age'], errors='coerce').fillna(0)
    
    fig = px.scatter(df, x='Heures_Internet', y='Budget_Data', 
                     color=clusters.astype(str),
                     title=f"Segmentation en {k} groupes",
                     size=age_numeric.values,
                     size_max=20)
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Profil des segments")
    profil = df.groupby('Segment')[['Age', 'Heures_Internet', 'Budget_Data']].mean()
    st.dataframe(profil.style.background_gradient(cmap='Blues'))

# ----- TAB 6 : ANALYSE DESCRIPTIVE -----
with tab6:
    st.header("📊 Statistiques descriptives")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribution des âges")
        fig = px.histogram(df, x='Age', nbins=20)
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Répartition par pays")
        fig = px.pie(df, names='Pays')
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Matrice de corrélation")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Pas assez de données numériques pour la matrice de corrélation")
    
    st.subheader("Données brutes")
    st.dataframe(df)

st.markdown("---")
st.caption("INF232 EC2 - Application de collecte et analyse de données | Toutes les techniques du cours sont implémentées")