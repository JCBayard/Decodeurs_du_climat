import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
import plotly.express as px
import joblib
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from matplotlib.colors import LinearSegmentedColormap
import scipy.stats as stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_regression
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score


######################################################### LAY-OUT #############################################################################
############################################### FOND D'ECRAN PRINCIPAL ########################################################################

st.set_page_config(
    page_title="Largeur personnalisée Streamlit",
    layout="wide",
)
# CSS pour personnaliser la largeur
custom_width_style = """
<style>
.main {
    max-width: 70%; /* Ajustez ce pourcentage pour définir la largeur */
    margin: 0 auto; /* Centrer le contenu */
}
</style>
"""
# Injection du CSS
st.markdown(custom_width_style, unsafe_allow_html=True)

# Construire le chemin relatif vers l'image
chemin_image = os.path.join(os.getcwd(), "Logo Les Décodeurs du Climat.jpg")
# Charger l'image avec Streamlit
st.image(chemin_image, width=150, caption="", use_column_width=True)

from PIL import Image

import base64
# Fonction pour convertir l'image en Base64
def get_base64_of_image(file_path):
    with open(file_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Convertir l'image en Base64
# Construire le chemin relatif vers l'image
image_path = os.path.join(os.getcwd(), "thermometre_glb_transp.jpg")
# Charger l'image avec Streamlit

image_base64 = get_base64_of_image(image_path)
# CSS pour appliquer l'image en arrière-plan de la sidebar
sidebar_style = f"""
<style>
[data-testid="stSidebar"] {{
    background-image: url("data:image/jpeg;base64,{image_base64}");
    background-size: cover;
    background-repeat: no-repeat;
    background-position: center;
}}
</style>
"""
# Injection du CSS dans Streamlit
st.markdown(sidebar_style, unsafe_allow_html=True)

# Construire le chemin relatif vers l'image
image2_path = os.path.join(os.getcwd(), "DTS-new-logo.png")

# CSS pour centrer l'image
centered_image_style = """
<style>
[data-testid="stSidebar"] > div:first-child {
    display: flex;
    justify-content: center;  /* Centrer horizontalement */
    align-items: center;     /* Centrer verticalement si nécessaire */
    height: 100%;            /* Optionnel pour ajuster la hauteur */
}
</style>
"""
# Injection du style pour centrer l'image
st.markdown(centered_image_style, unsafe_allow_html=True)

# Afficher l'image dans la sidebar
st.sidebar.image(image2_path, caption="DataScientest", width=160)  # Ajuster la largeur si nécessaire

st.sidebar.write("## Soutenance Projet")
st.sidebar.write("### TEMPERATURE TERRESTRE")

st.sidebar.write("## Haut Sommet'R 📈") 
pages = [
    "Objectifs",
    "Exploration des données",
    "Visualisation des données",
    "Modélisation Machine Learning",
    "Prédictions",
    "Pertinence & Conclusion",
]
page = st.sidebar.radio("", pages)

st.sidebar.write("")
st.sidebar.write("#### Notre Mentor:")
text000 = """
    <span style="color: brown;">
    <div style="text-indent: 20px;">
    Alain FERLAC
    </span>
    """
st.sidebar.markdown(text000, unsafe_allow_html=True)

st.sidebar.write("#### Les décodeurs:")
text001 = """
    <span style="color: brown;">
    <div style="text-indent: 20px;">
    Assma ALIOUI
    <br> <div style="text-indent: 20px;">
    Félicia Sandra FOTSO 
    <br><div style="text-indent: 20px;">
    Emeric SCHAAL
    <br><div style="text-indent: 20px;">
    Jean-Christophe BAYARD
    </span>
    """
st.sidebar.markdown(text001, unsafe_allow_html=True)

st.sidebar.write("#### Soutenance:")
text002 = """
    <span style="color: brown;">
    <div style="text-indent: 20px;">
    17 mars 2025<br>
    <br>  
    </span>
    """
st.sidebar.markdown(text002, unsafe_allow_html=True)

# Définition d'une fonction de replacement en haut de la page lors de la sélection de la page
def scroll_to_top():
    st.markdown(
        """
        <script>
            window.scrollTo(0, 0);
        </script>
        """,
        unsafe_allow_html=True
    )

if page == pages[0] : 
    scroll_to_top() # appel de la fonction pour remonter en haut de la page
    st.write("# Objectifs")
    text00 = """
    <span style="color: blue;font-size:10px">
    <br>
    <span style="color: blue;font-size:30px">
    <div style="text-indent: 20px;">
    Objectifs initiaux du projet de groupe : 
    
    <br>
     
    <span style="color: black;font-size:25px">
    
    <div style="text-indent: 40px;">
        1. Constater la situation climatique, le réchauffement et le dérèglement  

    <span style="color: black;font-size:20px">
    <div style="text-indent: 70px;">
    - A l’échelle globale de la planète sur les derniers siècles et dernières décennies  
    <div style="text-indent: 70px;">
    - Analyse des données sous différents axes
    <div style="text-indent: 70px;">
    - Comparer la situation actuelle à des phases de température antérieures
    <br>

    <span style="color: red;font-size:30px">
    <br>
    <div style="text-indent: 20px;">
    Objectifs complémentaires entrepris par l'équipe :    
    
    <br>
    <span style="color: black;font-size:25px">
    <div style="text-indent: 40px;">
        2. Rechercher et entrainer un modèle de prédictions 
    <br>
     <div style="text-indent: 40px;">
        3. Prédire l'évolution de la situation
    <br>
     <div style="text-indent: 40px;">
        4. Déterminer la pertinence de nos prédictions
    <span>
    """
    st.markdown(text00, unsafe_allow_html=True)

if page == pages[1] : 
    scroll_to_top() # appel de la fonction pour remonter en haut de la page   
    st.write("## Exploration des données 'NASA Globe' et 'OWID' (Our World In Data)")

    # Construire le chemin relatif vers le fichier CSV
    chemin_csv = os.path.join(os.getcwd(), "GLB.Ts+dSST.csv")
    # Charger le fichier CSV
    T_GLB = pd.read_csv(chemin_csv, skiprows=1)

    st.markdown(
    """
    <style>
        .block-container {
            padding-left: 20px; /* Ajuste ici l'indentation globale si nécessaire */
        }
    </style>
    """,
        unsafe_allow_html=True)

    if st.checkbox("**Dataframe Global de la NASA**", key="df_nasa"):
        
        st.write("**Ecarts de température** mensuels, trimestriels (saison) et annuels par rapport à la moyenne de la <u>période de référence 1951-1980</u>", unsafe_allow_html=True)
        st.write("Aperçu du dataframe")
        T_GLB_2 = T_GLB
        T_GLB_2['Year'] = T_GLB['Year'].astype(str)
        st.dataframe(T_GLB_2.head())
        st.write(T_GLB.shape)

        T_GLB["D-N"] = T_GLB["D-N"].replace("***", None).astype(float)  # Remplacer les astérisques et convertir en float
        T_GLB["DJF"] = T_GLB["DJF"].replace("***", None).astype(float)

        col1, col2 = st.columns([0.05, 0.95])
        with col2:
            if st.checkbox("Valeurs statistiques du dataframe", key="stats_nasa"):
                st.dataframe(T_GLB.describe())
            if st.checkbox("Valeurs manquantes", key="missing_nasa"):
                st.write(T_GLB.isna().sum())
    
    st.markdown(
    """
    <style>
        .block-container {
            padding-left: 20px; /* Ajuste ici l'indentation globale si nécessaire */
        }
    </style>
    """,
        unsafe_allow_html=True)
    
    if st.checkbox("**Dataframe OWID**", key="df_owid"):
        source1 = os.path.join(os.getcwd(), "owid-co2-data.csv")
        df = pd.read_csv(source1)
        df_2 = df
        df_2['year']=df_2['year'].astype(int).astype(str)
        # Analyse du dataframe:
        col1, col2 = st.columns([0.05, 0.95])
        with col2:
            if st.checkbox("Aperçu du dataframe", key="apercu_owid"):
                st.dataframe(df_2.head())
                st.write(df.shape)
        
            if st.checkbox("Valeurs statistiques du dataframe", key="stats_owid"):
                st.dataframe(df.describe())
        
            if st.checkbox("Proportion de valeurs manquantes", key="missing_owid"):
                st.write(df.isna().sum() / len(df['year']) * 100)

                df_world_var_select_visu_path = os.path.join(os.getcwd(), "df_world_var_select_Visu.csv")
                df_world_var_select_visu = pd.read_csv(df_world_var_select_visu_path)
                st.write("### Histogramme des valeurs manquantes (NaN)")
                st.write("##### ")
                st.write("##### Ce graphique montre le nombre de <u> **valeurs annuelles manquantes**</u> (NaN) par <u>**décennie**</u> et par variable.", unsafe_allow_html=True)
                st.write("Certaines variables n'ont que peu de données (continues/annuelles) disponibles, et parfois depuis peu de temps.", unsafe_allow_html=True)
                # Calcul du nombre de valeurs manquantes par tranche temporelle
                decade_nan_count = df_world_var_select_visu.groupby('year_range').agg(lambda x: x.isna().sum()).reset_index()
                decade_nan_count = decade_nan_count.sort_values('year_range', ascending=False)

                # Transformation pour heatmap
                decade_nan_count_melted = decade_nan_count.melt(id_vars='year_range', var_name='Variables', value_name='nan_count')
                pivot_table_nan = decade_nan_count_melted.pivot(index='year_range', columns='Variables', values='nan_count').sort_index(ascending=False)

                # Définition de la palette de couleurs (Vert → Bleu → Rouge)
                colors = [(0, 1, 0), (0, 0, 1), (1, 0, 0)]  # Vert -> Bleu -> Rouge
                n_bins = 11  # Nombre de niveaux
                cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=n_bins)

                # Création du graphique
                fig, ax = plt.subplots(figsize=(6, 3))
                
                # Paramètres personnalisés pour les polices
                font_title_size = 6  # Taille de la police pour le titre
                font_label_x_size = 6  # Taille de la police pour le label de l'axe X
                font_label_y_size = 6  # Taille de la police pour le label de l'axe Y
                font_ticks_size = 4  # Taille de la police pour les ticks
                font_annotation_size = 6  # Taille de la police pour les annotations dans la heatmap
                font_colorbar_size = 5  # Taille de la police pour la légende de couleur (colorbar)

                sns.heatmap(
                    pivot_table_nan, cmap=cmap, vmin=0, vmax=n_bins - 1, annot=True, fmt='d',
                    annot_kws={"size": font_annotation_size},  # Taille de la police des annotations
                    cbar_kws={'label': 'Nbr valeurs NaN'}, linecolor='white', linewidths=0.5
                )

                ax.tick_params(axis='x', labelsize=font_ticks_size)  # Taille des ticks pour l'axe X
                ax.tick_params(axis='y', labelsize=font_ticks_size)  # Taille des ticks pour l'axe Y
                
                # Ajustement des éléments du graphique
                ax.set_xlabel("Variables", fontsize=font_label_x_size)  # Taille du label de l'axe X
                ax.set_ylabel("Décennies", fontsize=font_label_y_size)  # Taille du label de l'axe Y
                ax.set_title(" ", fontsize=font_title_size, loc="right")  # Taille du titre

                # Affichage dans Streamlit
                st.pyplot(fig, use_container_width=False)
    st.write("#####     ")
    if st.checkbox("**Modifications du dataframe OWID**", key="modif_owid"):
        st.write("Nous avons réduit le dataframe OWID sur la <u>**variable 'Country'**</u> pour afficher uniquement les <u>**valeurs mondiales**</u> ainsi que pour coller aux années du dataframe de la **NASA**", unsafe_allow_html=True)

        st.markdown(
    """
    <style>
        .block-container {
            padding-left: 20px; /* Ajuste ici l'indentation globale si nécessaire */
        }
    </style>
    """,
        unsafe_allow_html=True)
        
        col1, col2 = st.columns([0.05, 0.95])
        with col2:
            if st.checkbox("Préparation du dataframe pour le calcul de la population", key="prep_df"):
                
                df_world2_path = os.path.join(os.getcwd(), "df_world2.csv")
                df_world2 = pd.read_csv(df_world2_path)

                col3, col4 = st.columns([0.05, 0.95])
                with col4:
                    #if st.checkbox("Aperçu du dataframe", key="apercu_world2"):
                    st.dataframe(df_world2.head())
                    st.write("Nous avons ici que des valeurs **tous les 10 ans juqu'en 1950**, nous devons calculer les valeurs manquantes grâce à une **formule mathématique**.")
                    model = np.poly1d(np.polyfit(df_world2["year"], df_world2["population"], 4))
                    
                subcol1, subcol2, subcol3 = st.columns([0.05,0.4,0.50])
                with subcol2:
                    # Affichage de la courbe du polynôme
                    myline = np.linspace(1880, 2020, 10)
                    fig, ax = plt.subplots(figsize=(4, 3))  # Taille ajustée de la figure

                    # Points de données avec taille personnalisée
                    ax.scatter(df_world2["year"], df_world2["population"], label="Données réelles", color="red", s=10)  # Taille des points : s=50

                    # Courbe du modèle avec épaisseur personnalisée
                    ax.plot(myline, model(myline), label="Modèle polynomial", color="blue", linewidth=1)  # Épaisseur de la ligne : linewidth=2

                    # Personnalisation des labels et du titre
                    ax.set_xlabel("Années", fontsize=6)
                    ax.set_ylabel("Population (Milliards)", fontsize=6)
                    ax.set_title("Évolution de la population au fil du temps", fontsize=8)

                    # Personnalisation des ticks des axes
                    ax.tick_params(axis='x', labelsize=5)
                    ax.tick_params(axis='y', labelsize=5)

                    # Ajouter une légende avec taille de police personnalisée
                    ax.legend(fontsize=6)

                    # Grille et tracé
                    ax.grid(True)

                    # Afficher le graphe dans Streamlit
                    st.pyplot(fig, use_container_width=False)

                st.write("Voici la formule utilisée pour calculer cette courbe :")
                st.write(model)

                df_world_path = os.path.join(os.getcwd(), "df_world.csv")
                df_world = pd.read_csv(df_world_path)
                df_world_2 = df_world
                df_world_2['year']=df_world_2['year'].astype(int).astype(str)
                st.write("Nous avons intégré les valeurs calculées par le **polynôme** dans le dataframe d'origine.")
                st.dataframe(df_world_2.head())

                st.write("Nous avons concaténé les dataframes OWID et NASA.")
                merged_df_path = os.path.join(os.getcwd(), "df_world+nasa.csv")
                merged_df = pd.read_csv(merged_df_path)
                merged_df_2 = merged_df
                merged_df_2['year']=merged_df_2['year'].astype(int).astype(str)
                st.dataframe(merged_df_2.head())
            
if page == pages[2] : 
    scroll_to_top() # appel de la fonction pour remonter en haut de la page
    st.write("## Visualisation des données OWID")

    st.markdown(
    """
    <style>
        .block-container {
            padding-left: 20px; /* Ajuste ici l'indentation globale si nécessaire */
        }
    </style>
    """,
        unsafe_allow_html=True)

    col1, col2 = st.columns([2,5])
    with col1:
        option = st.selectbox(
        "Sélectionnez la variable à afficher :",
        ["CO2", "Méthane", "Population", "Différents gaz à effet de serre"])
    
        merged_df_path = os.path.join(os.getcwd(), "df_world+nasa.csv")
        merged_df = pd.read_csv(merged_df_path)
    if option == "CO2":

        col1, col2,col3= st.columns([2.5,3,3])
        with col1:
            st.write("#### Analyse de la relation CO2 - Température")
            st.write("Ce graphique explore la relation entre les émissions de CO2 (including LUC) et la température moyenne annuelle (J-D).")
            
            # Création du graphique
            fig, ax = plt.subplots()
            sns.scatterplot(data=merged_df, x="co2_including_luc", y="J-D", color="blue", label="Données", ax=ax)
            sns.regplot(data=merged_df, x="co2_including_luc", y="J-D", scatter=False, color="red", label="Tendance", ci=None, ax=ax)

            ax.set_xlabel("CO2_including_luc")
            ax.set_ylabel("Température (J-D, °C)")
            ax.set_title("Relation entre CO2_including_luc et Température (J-D)")
            ax.legend()
            ax.grid(True)

            st.pyplot(fig, use_container_width=True)

            # Calcul du coefficient de corrélation et de la P-value
            correlation, p_value = stats.pearsonr(merged_df["co2_including_luc"], merged_df["J-D"])
            subcol1, subcol2= st.columns([1,3])
            with subcol2:
                st.write(f"**Coefficient de corrélation :** {correlation:.4f}")
                st.write(f"**P-value :** {p_value:.4e}")

        with col2:
            st.write("#### Évolution des émissions de CO2 dans le temps")
            st.write("Ce graphique montre l'évolution des émissions de CO2 au fil des années.")

        # Création du graphique
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(merged_df['year'], merged_df['co2_including_luc'], marker='o', linestyle='-', color='r', label="Émissions de CO2")

            ax.set_xlabel("Année")
            ax.set_ylabel("Émissions de CO2")
            ax.set_title("Émission de CO2 dans le temps")
            ax.legend()
            ax.grid(True)

        # Affichage dans Streamlit
            st.pyplot(fig, use_container_width=True)

        with col3:
            st.write("#### Évolution des émissions de CO2 cumulées au fil des années.")
            st.write("Ce graphique montre l'évolution des émissions de CO2 <u>**cumulées**</u> au fil des années.", unsafe_allow_html=True)

            # Création du graphique
            fig, ax = plt.subplots(figsize=(10, 5.9))
            ax.plot(merged_df['year'], merged_df['cumulative_co2_including_luc'], marker='o', linestyle='-', color='r', label="Émissions de CO2 cumulées")

            ax.set_xlabel("Année")
            ax.set_ylabel("Émissions de CO2 cumulées")
            ax.set_title("Émissions de CO2 cumulées dans le temps")
            ax.legend()
            ax.grid(True)

        # Affichage dans Streamlit
            st.pyplot(fig, use_container_width=True)

    if option == "Méthane":
        col1, col2, col3= st.columns([2.5,0.5,3])
        with col1:
            st.write("### Analyse de la relation Méthane - Température")
            st.write("Ce graphique explore la relation entre les émissions de Méthane et la température moyenne annuelle (J-D).")
            
            # Création de la figure
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(data=merged_df, x="methane", y="J-D", color="blue", label="Données", ax=ax)
            sns.regplot(data=merged_df, x="methane", y="J-D", scatter=False, color="red", label="Tendance", ci=None, ax=ax)

            ax.set_xlabel("Méthane")
            ax.set_ylabel("Température (J-D, °C)")
            ax.set_title("Relation entre Méthane et Température (J-D)")
            ax.legend()
            ax.grid(True)

            st.pyplot(fig)

            subcol1, subcol2= st.columns([1,3])
            with subcol2:
                # Calcul du coefficient de corrélation et de la P-value
                correlation, p_value = stats.pearsonr(merged_df["methane"], merged_df["J-D"])
                
                st.write(f"**Coefficient de corrélation :** {correlation:.4f}")
                st.write(f"**P-value :** {p_value:.4e}")

        
        with col3:
            st.write("### Évolution des émissions de Méthane")

            st.write("Ce graphique montre l'évolution des émissions de méthane au fil des années.")

        # Création du graphique
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(merged_df['year'], merged_df['methane'], marker='o', linestyle='-', color='r', label="Émissions de Méthane")

            ax.set_xlabel("Année")
            ax.set_ylabel("Émissions de Méthane")
            ax.set_title("Émissions de Méthane dans le temps")
            ax.legend()
            ax.grid(True)

        # Affichage dans Streamlit
            st.pyplot(fig,use_container_width=True)

    if option == "Population":
        col1, col2, col3= st.columns([3,0.1,3.2])
        with col1:
            st.write("### Analyse de la relation Population - Température")

            st.write("Ce graphique explore la relation entre la population et la température moyenne annuelle (J-D).")
            
            # Création de la figure
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(data=merged_df, x="population", y="J-D", color="blue", label="Données", ax=ax)
            sns.regplot(data=merged_df, x="population", y="J-D", scatter=False, color="red", label="Tendance", ci=None, ax=ax)

            ax.set_xlabel("Population")
            ax.set_ylabel("Température (J-D, °C)")
            ax.set_title("Relation entre Population et Température (J-D)")
            ax.legend()
            ax.grid(True)

            st.pyplot(fig,use_container_width=True)

            subcol1, subcol2= st.columns([1,3])
            with subcol2:
                # Calcul du coefficient de corrélation et de la P-value
                correlation, p_value = stats.pearsonr(merged_df["population"], merged_df["J-D"])
                st.write(f"**Coefficient de corrélation :** {correlation:.4f}")
                st.write(f"**P-value :** {p_value:.4e}")
        
        with col3:
            st.write("### Analyse de la relation Population - CO2")

            st.write("Ce graphique explore la relation entre la population mondiale et les émissions de CO2 (including LUC).")
            
            # Création de la figure
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(merged_df['population'], merged_df['co2_including_luc'], marker='o', linestyle='-', color='r', label='CO2 vs Population')

            ax.set_xlabel("Population mondiale")
            ax.set_ylabel("Émissions de CO2")
            ax.set_title("Relation entre la population mondiale et les émissions de CO2")
            ax.legend()
            ax.grid(True)

            st.pyplot(fig,use_container_width=True)

    if option == "Différents gaz à effet de serre":
                
        col1, col2,col3= st.columns([2.6,0.1,3])
        
        with col1:
            ######### GRAPHE 1 #########
            st.write("### Évolution de la différence de température terrestre (1880-2023)")
            st.write("Ce graphique montre l'évolution de l'impact de 2 gaz à effet de serre (**Dioxyde de Carbone (CO2)** et **Oxyde nitreux (NO2)**) et l'**ensemble des GES** (ou GHG) sur la température mondiale.")
            
            OWID1_CO2_SHARE_path = os.path.join(os.getcwd(),  "OWID1_CO2_SHARE.csv")
            OWID1_CO2_SHARE = pd.read_csv(OWID1_CO2_SHARE_path)

            # Création du graphique
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(OWID1_CO2_SHARE["year"], OWID1_CO2_SHARE["temperature_change_from_ghg"], "r--", label="GHG")
            ax.plot(OWID1_CO2_SHARE["year"], OWID1_CO2_SHARE["temperature_change_from_co2"], "y--", label="CO2")
            ax.plot(OWID1_CO2_SHARE["year"], OWID1_CO2_SHARE["temperature_change_from_n2o"], "b--", label="N2O")

            ax.set_title("Évolution de la différence de température terrestre \n au niveau mondial (1880-2023)")
            ax.set_ylabel("Température (°C)")
            ax.set_xlabel("Année")
            ax.legend()
            ax.grid(True)

            # Affichage dans Streamlit
            st.pyplot(fig,use_container_width=True)
        
        with col3:
            ######### GRAPHE 2 #########
            OWID1_world_ghg_path = os.path.join(os.getcwd(),  "OWID1_world_ghg.csv")
            OWID1_world_ghg = pd.read_csv(OWID1_world_ghg_path)

            T_GLB_path = os.path.join(os.getcwd(),  "T_GLB.csv")
            T_GLB = pd.read_csv(T_GLB_path)
            T_GLB = T_GLB.iloc[:-1]

            st.write("### Évolution des émissions de gaz à effet de serre vs. température (1880-2023)")

            st.write("Ce graphique montre la relation entre l'évolution des émissions CO2**e** des **3 plus grands gaz à effet de serre** et les variations de température mondiale.")

            # Création du graphique
            fig, ax1 = plt.subplots(figsize=(10, 6))

            # Courbes des gaz à effet de serre
            ax1.plot('year', 'total_ghg', data=OWID1_world_ghg, label="Total GES", color="yellow")
            ax1.fill_between(OWID1_world_ghg["year"], OWID1_world_ghg["total_ghg"], color="yellow", alpha=0.3)

            ax1.plot('year', 'co2_including_luc', data=OWID1_world_ghg, label="CO₂", color="darkorange")
            ax1.fill_between(OWID1_world_ghg["year"], OWID1_world_ghg["co2_including_luc"], color="darksalmon", alpha=0.3)

            ax1.plot('year', 'methane', data=OWID1_world_ghg, label="CH₄ (Méthane)", color="brown")
            ax1.fill_between(OWID1_world_ghg["year"], OWID1_world_ghg["methane"], color="brown", alpha=0.3)

            ax1.plot('year', 'nitrous_oxide', data=OWID1_world_ghg, label="N₂O (Protoxyde d’azote)", color="saddlebrown")
            ax1.fill_between(OWID1_world_ghg["year"], OWID1_world_ghg["nitrous_oxide"], color="saddlebrown", alpha=0.3)

            ax1.set_xlabel("Année")
            ax1.set_ylabel("Émissions de GES (MTCO₂ eq)", color="grey")
            ax1.tick_params(axis="y", labelcolor="grey")

            # Courbe de la température en axe secondaire
            ax2 = ax1.twinx()
            ax2.plot("Year", 'J-D', data=T_GLB, label="ΔTempérature", color="teal", marker="^", linestyle="-", alpha=0.7)
            ax2.set_ylabel("Δ Température (°C)", color="teal")
            ax2.tick_params(axis="y", labelcolor="teal")

            # Ajout des légendes
            ax1.legend(loc="upper left")
            ax2.legend(loc="upper right")

            plt.title("Évolution des émissions de gaz à effet de serre vs. température mondiale (1880-2023)", color="grey")

            # Affichage dans Streamlit
            st.pyplot(fig)

        col1, col2,col3= st.columns([1,5,1])
        with col2:
            st.write("### ")
            #"######### GRAPHE 3 #########
            st.write("### Évolution des émissions cumulées de CO₂ par source (1900/1950/2000/2023)")
            st.write("Ces graphiques illustrent l'évolution des émissions cumulées de CO₂ <u>**par source**</u> (déforestation, charbon, gaz, pétrole, etc.) en <u>**4 années**</u> de la période analysée.",unsafe_allow_html=True)

            # Définition des labels des catégories d’émissions
            labels = ['cumulative_luc_co2', 'cumulative_coal_co2', 'cumulative_gas_co2', 'cumulative_oil_co2', 'cumulative_other_co2']

            # Extraction des données pour les années 1900, 1950, 2000 et 2023
            X_1900 = OWID1_CO2_SHARE.loc[OWID1_CO2_SHARE['year'] == 1900, labels].iloc[0]
            X_1950 = OWID1_CO2_SHARE.loc[OWID1_CO2_SHARE['year'] == 1950, labels].iloc[0]
            X_2000 = OWID1_CO2_SHARE.loc[OWID1_CO2_SHARE['year'] == 2000, labels].iloc[0]
            X_2023 = OWID1_CO2_SHARE.loc[OWID1_CO2_SHARE['year'] == 2023, labels].iloc[0]

            # Fonction pour l'affichage des pourcentages
            def autopct_func(pct):
                return ('%1.1f%%' % pct) if pct > 0.7 else ''

            # Création du graphique en camemberts
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            fig.suptitle("", fontsize=14, color="green")

            # Définition des couleurs et des étiquettes
            colors = ['#ccebc5', '#a8ddb5', '#7bccc4', '#43a2ca', '#0868ac']
            labels_fr = ['LUC\n(Déforestation)', 'Charbon', 'Gaz', 'Pétrole', 'Autre']

            # Ajout des sous-graphiques
            axes[0, 0].pie(X_1900, labels=labels_fr, wedgeprops={'edgecolor': 'white'}, textprops={'fontsize': 7}, autopct=autopct_func, colors=colors)
            axes[0, 0].set_title("Année 1900", fontsize=10, color="grey")

            axes[0, 1].pie(X_1950, labels=labels_fr, wedgeprops={'edgecolor': 'white'}, textprops={'fontsize': 7}, autopct=autopct_func, colors=colors)
            axes[0, 1].set_title("Année 1950", fontsize=10, color="grey")

            axes[1, 0].pie(X_2000, labels=labels_fr, wedgeprops={'edgecolor': 'white'}, textprops={'fontsize': 7}, autopct=autopct_func, colors=colors)
            axes[1, 0].set_title("Année 2000", fontsize=10, color="grey")

            axes[1, 1].pie(X_2023, labels=labels_fr, wedgeprops={'edgecolor': 'white'}, textprops={'fontsize': 7}, autopct=autopct_func, colors=colors)
            axes[1, 1].set_title("Année 2023", fontsize=10, color="grey")

            # Affichage de la légende
            plt.legend(labels_fr, bbox_to_anchor=(1.5, 1))

            # Affichage dans Streamlit
            st.pyplot(fig)


    ################################## VISUALISATION DONNEES NASA ZONALES ######################################################
    st.write("## ")
    st.write("### Visualisation des données de la NASA par zones latitudinales")

    ### GRAPHE "Mappemonde des zones latitudinales"
    st.write("##### Mappemonde des zones latitudinales",unsafe_allow_html=True)
    # Définir les zones avec les intervalles de latitude
    zones = [
        {"zone": "24N-90N", "lat_start": 24, "lat_end": 90},
        {"zone": "24S-24N", "lat_start": -24, "lat_end": 24},
        {"zone": "90S-24S", "lat_start": -90, "lat_end": -24},
        {"zone": "64N-90N", "lat_start": 64, "lat_end": 90},
        {"zone": "44N-64N", "lat_start": 44, "lat_end": 64},
        {"zone": "24N-44N", "lat_start": 24, "lat_end": 44},
        {"zone": "EQU-24N", "lat_start": 0, "lat_end": 24},
        {"zone": "24S-EQU", "lat_start": -24, "lat_end": 0},
        {"zone": "44S-24S", "lat_start": -44, "lat_end": -24},
        {"zone": "64S-44S", "lat_start": -64, "lat_end": -44},
        {"zone": "90S-64S", "lat_start": -90, "lat_end": -64},
    ]
    # Créer un DataFrame des zones
    df_ZA = pd.DataFrame(zones)
    # Création d'une mappemonde interactive
    fig = px.scatter_geo(
        df_ZA, 
        lat="lat_start", 
        lon=[0]*len(df_ZA), 
        hover_name="zone", 
        size_max=10, 
        projection="natural earth"
    )
    # Ajouter les lignes représentant les intervalles de latitude
    for i, row in df_ZA.iterrows():
        fig.add_trace(
            go.Scattergeo(
                lat=[row["lat_start"], row["lat_end"]],
                lon=[0, 0],
                mode="lines",
                line=dict(width=5),
                name=row["zone"]
            )
        )
    # Personnaliser l'apparence de la mappemonde
    fig.update_geos(
        showcoastlines=True, 
        coastlinecolor="Black", 
        showland=True, 
        landcolor="lightgrey"
    )
    # Afficher le graphique dans Streamlit
    st.plotly_chart(fig, use_container_width=True)


    ### GRAPHE "Evolution des écarts de températures par zones latitudinales"
    st.write("##### Evolution des écarts de températures par zones latitudinales",unsafe_allow_html=True)
   
    # Charger les données
    data_path = os.path.join(os.getcwd(),  "ZonAnn.Ts+dSST.csv")
    data = pd.read_csv(data_path)
    # Sélectionner les colonnes nécessaires
    cols = ['Year', '64N-90N', '44N-64N', '24N-44N', 'EQU-24N', '24S-EQU', '44S-24S', '64S-44S', '90S-64S']
    data = data[cols]
    # Gérer les valeurs manquantes (interpolation au lieu de 0)
    data = data.interpolate()
    # Transformer les données pour avoir années (X) et latitudes (Y)
    data_long = data.melt(id_vars=['Year'], var_name='Latitude', value_name='Température')

    # Mise à jour des modalités pour assurer le classement des latitudes
    data_long['Latitude'] = data_long['Latitude'].replace({
        '90S-64S': 'A: 90S/64S (Pôle Sud-Cercle Antarctique)',
        '64S-44S': 'B: 64S/44S (Cercle Antarctique-ARG/NZL)',
        '44S-24S': 'C: 44S/24S (ARG/NZL-Tropique Capricorne)',
        '24S-EQU': 'D: 24S/EQU (Tropique Capricorne -Equateur)',
        'EQU-24N': 'E: EQU/24N (Equateur-Tropique Cancer)',
        '24N-44N': 'F: 24N/44N (Tropique Cancer-FRA/USA/JPN)',
        '44N-64N': 'G: 44N/64N (FRA/USA/JPN-Cercle Arctique)',
        '64N-90N': 'H: 64N/90N (Cercle Arctique-Pôle Nord)'
    })
    # Créer un heatmap interactif avec Plotly
    fig = px.imshow(
        data_long.pivot(index='Latitude', columns='Year', values='Température').sort_index(ascending=False),
        color_continuous_scale="RdBu_r",
        #title="Evolution des écarts de températures par latitude",
        labels={"color": "Ecart de température (°C)"}
    )
    # Ajustements de layout
    fig.update_layout(
        xaxis_title="Années",
        yaxis_title="Zones latitudinales",
        width=1400,  # Ajustez si besoin
        height=800,  # Ajustez si besoin
        xaxis=dict(
          title_font=dict(size=18),  # Taille du titre de l'axe X
          tickfont=dict(size=14)  # Taille des ticks sur l'axe X
        ),
        yaxis=dict(
          title_font=dict(size=18),  # Taille du titre de l'axe Y
          tickfont=dict(size=14)  # Taille des ticks sur l'axe Y
        )
    )
    # Afficher le graphique dans Streamlit
    st.plotly_chart(fig, use_container_width=False)

if page == pages[3] : 
    scroll_to_top() # appel de la fonction pour remonter en haut de la page
    st.write("### Modélisation Machine Learning")
    # 📌 Charger les datasets
    
    # Construire les chemins relatifs pour chaque fichier
    chemin_df_ML1 = os.path.join(os.getcwd(), "df_ML1.xls")
    chemin_df_ML2 = os.path.join(os.getcwd(), "df_ML2.xls")
    chemin_df_ML3 = os.path.join(os.getcwd(), "df_ML3.xls")
    chemin_df_ML4 = os.path.join(os.getcwd(), "df_ML4.xls")
    chemin_df_ML5 = os.path.join(os.getcwd(), "df_ML5_RFR.xls")
    chemin_ML6_best = os.path.join(os.getcwd(), "ML6_best.xls")

    # Charger les fichiers avec pandas
    df_ML1 = pd.read_csv(chemin_df_ML1, dtype={"year": str})  
    df_ML2 = pd.read_csv(chemin_df_ML2, dtype={"year": str})
    df_ML3 = pd.read_csv(chemin_df_ML3, dtype={"year": str})
    df_ML4 = pd.read_csv(chemin_df_ML4, dtype={"year": str})
    df_ML5 = pd.read_csv(chemin_df_ML5, dtype={"year": str})
    ML6_best = pd.read_csv(chemin_ML6_best, dtype={"year": str})

    df_ML1 = df_ML1.set_index('year', drop=False)
    df_ML2 = df_ML2.set_index('year', drop=False)
    df_ML3 = df_ML3.set_index('year', drop=False)
    df_ML5 = df_ML5.set_index('year', drop=False)
    ML6_best = ML6_best.set_index('year', drop=False)

    # Créer une mise en page avec deux colonnes
    col1, col2, col3, col4 = st.columns([2, 0.1, 0.9, 1.5   ])  # Ajuste les proportions si nécessaire

    # Placer le sélecteur de modèle dans la colonne de gauche
    with col1:
        model_type = st.selectbox("📌 Choisissez le type de modèle :", 
                                ["Régression linéaire","Arbre de régression",  "Forêt aléatoire"])

        # 📌 Sélection du modèle en fonction du type choisi
        if model_type == "Arbre de régression":
            model_choice = st.selectbox("📌 Choisissez le nombre de variables :", ["Toutes les variables.", "3 Variables", "4 Variables.","5 Variables", "6 Variables"])
            if model_choice == "Toutes les variables.":
                df = df_ML1
                model_path = os.path.join(os.getcwd(), "Modèle_dtr.pkl")
            elif model_choice == "4 Variables.":
                df = df_ML2
                model_path = os.path.join(os.getcwd(), "Modèle_dtr2.pkl")
            
        elif model_type == "Régression linéaire":
            model_choice = st.selectbox("📌 Choisissez le nombre de variables:", ["Toutes les variables.", "3 Variables", "4 Variables","5 Variables", "6 Variables"])
            if model_choice == "Toutes les variables.":
                df = df_ML3
                model_path = os.path.join(os.getcwd(), "Modèle_LR1.pkl")

            elif model_choice == "5 Variables":
                df = df_ML4
                model_path = os.path.join(os.getcwd(), "Modèle_LR2.pkl")

        elif model_type == "Forêt aléatoire":
            model_choice = st.selectbox("📌 Choisissez le nombre de variables", ["Toutes les variables.", "3 Variables", "4 Variables","5 Variables", "6 Variables."])
            # 📌 Case à cocher pour faire varier le split uniquement pour Forêt aléatoire
            #vary_split = st.checkbox("Faire varier le split du modèle Forêt aléatoire")
        # 📌 Si la case est cochée, afficher le slider
            #if vary_split:
            #model_choice = st.slider("Choisissez un ratio de split :", 0.2, 0.25, 0.5,  step=0.05)
            def adjust_value(value):
                if value <= 0.225:
                    return 0.20
                elif value <= 0.475:    
                    return 0.25
                else:
                    return 0.5
            
            # Utilisation du slider
            raw_choice = st.slider("Choisissez un ratio de split :", 0.2, 0.5, step=0.05)
            model_choice_split = adjust_value(raw_choice)
            if model_choice == "Toutes les variables.":
                df = df_ML5
                
                if model_choice_split == 0.2:
                    model_path = os.path.join(os.getcwd(), "Modèle_RFR1.pkl")
                elif model_choice_split == 0.25:
                    model_path = os.path.join(os.getcwd(), "Modèle_RFR_best.pkl")
                elif model_choice_split == 0.5:
                    model_path = os.path.join(os.getcwd(), "Modèle_RFR2.pkl")
            
            elif model_choice == "6 Variables.":
                df = ML6_best
                model_choice_split == 0.25
                model_path = os.path.join(os.getcwd(), "Modèle_RFR_6.pkl")


        # 📌 Séparer les variables
        X = df.drop("J-D", axis=1)
        y = df["J-D"]

        # 📌 Split des données
        split_ratio = 0.20
        shuffle_on_off = True
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, shuffle=shuffle_on_off, random_state=42)

        # 📌 Charger le modèle sélectionné
        model = joblib.load(model_path)

        # 📌 Aligner les colonnes du DataFrame avec celles du modèle
        X = X[model.feature_names_in_]

        # 📌 Prédictions
        y_pred = model.predict(X_test)
        y_pred_train = model.predict(X_train)

        # 📌 Calcul des métriques
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2_test = r2_score(y_test, y_pred)

        # 📌 Affichage des résultats

        df = pd.DataFrame(df)

        # Fonction d'affichage conditionnel du dataframe
        def afficher_dataframe(dataframe, afficher=True):
            if afficher:
                st.dataframe(dataframe, use_container_width=True)
            else:
                st.write(" ")

        # Ajouter une case à cocher pour afficher/masquer le tableau
        afficher = st.checkbox("Afficher les résultats")

        # Afficher le dataframe en fonction de la case à cocher
        afficher_dataframe(df, afficher)

    with col3:
        st.markdown("##### 📊 **Tableau des Métriques**")

    # Créer un dataframe avec les résultats
        metrics = {
        "Métrique": ["MAE", "MSE ", "Score R² Train", "Score R² Test"],
        "Valeur": [mae, mse, model.score(X_train, y_train), model.score(X_test, y_test)]}

        df_metrics = pd.DataFrame(metrics)

        # Style du tableau
        styled_df = df_metrics.style.format({"Valeur": "{:.4f}"}).applymap(lambda x: "background-color: #FFEEA8" if x > 0.7 else "background-color: #D1E7DD", subset=["Valeur"])

        st.write(styled_df)

    with col4:
        st.write("##### 📊 Importance des variables")
       
        if not isinstance(model, LinearRegression):  # Only for non-linear models like decision trees
            # Feature importances for models like DecisionTreeRegressor
            importances = model.feature_importances_

            # Sort importances in descending order
            indices = np.argsort(importances)[::-1]

            # Create the plot for tree-based models with smaller size
            fig, ax = plt.subplots(figsize=(4, 2))  # Adjusted size (smaller)
            sns.barplot(x=X.columns[indices], y=importances[indices], palette="viridis", ax=ax)
            
            # Ajuster la taille de la police des labels et des titres
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=6)  # Taille du texte
            ax.set_xlabel(" ", fontsize=0.1)  # Taille du texte du label de l'axe x
            ax.tick_params(axis='y', labelsize=7)
            ax.grid(True)
            
            # Display the plot
            st.pyplot(fig, use_container_width=True)

        else:
            # Feature coefficients for Linear Regression
            importances = pd.Series(model.coef_, index=X.columns)

            # Sort by absolute importance (for better visualization of effect size)
            importances_sorted = importances.abs().sort_values(ascending=False)

            # Create the plot for linear regression with smaller size
            fig, ax = plt.subplots(figsize=(4, 2))  # Adjusted size (smaller)
            sns.barplot(x=importances_sorted.index, y=importances_sorted.values, palette="viridis", ax=ax)
            
            # Ajuster la taille de la police des labels et des titres
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=6)  # Taille du texte
            ax.set_xlabel(" ", fontsize=7)  # Taille du texte du label de l'axe x
            ax.tick_params(axis='y', labelsize=0.1)
            ax.grid(True)

            # 📌 Afficher le graphique dans Streamlit
            st.pyplot(fig, use_container_width=True)

# Créer une mise en page avec deux colonnes
    col1, col2, col3, col4, col5 = st.columns([0.05, 1.5, 0.05, 1.5, 0.15])  # Ajuste les proportions si nécessaire

    # Placer le sélecteur de modèle dans la colonne de gauche
    with col2:
    # 📌 Affichage du deuxième graphique (évolution des prédictions)
        
        # Vérifier si l'index est bien en années
        if not isinstance(y.index, pd.DatetimeIndex):
            y.index = pd.to_datetime(y.index, format='%Y')
            y_test.index = pd.to_datetime(y_test.index, format='%Y')
            y_train.index = pd.to_datetime(y_train.index, format='%Y')
        fig2 = plt.figure(figsize=(9, 7))  # Créer une nouvelle figure pour le deuxième graphique
        ax2 = fig2.add_subplot(111)
        ax2.plot(y.index, y, label='Valeurs réelles', color='grey', linestyle='--')
        ax2.plot(y_test.index, y_pred, label='Prédictions sur jeu de test', marker='o', linestyle='none')
        ax2.plot(y_train.index, y_pred_train, label="Prédictions sur jeu d'entraînement", marker='o', linestyle='none', alpha=0.6)
        ax2.plot(y.index, [0] * len(y.index), label='Moyenne 1951-1980', color='black', linewidth=0.5, linestyle='--')
        ax2.set_title(f"Prédictions de l'écart de température par rapport à la moyenne de référence (1951-1980)\n modèle {model_choice}", fontsize=12)
        ax2.set_xticks(np.arange(1880, 2024, 10))  # Placer un tick tous les 10 ans
        ax2.set_xticklabels(np.arange(1880, 2024, 10), rotation=25)  # Afficher les années correctement
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Afficher les années au format correct
        ax2.xaxis.set_major_locator(mdates.YearLocator(10))  # Placer un tick tous les 10 ans

        ax2.set_xlabel('Années')
        ax2.set_ylabel("Écart de température (°C)")
        ax2.legend()
        st.pyplot(fig2,use_container_width=True)

    with col4:   
        # 📌 Affichage du premier graphique (corrélation entre prédictions et valeurs réelles)
        fig1 = plt.figure(figsize=(9, 7))  # Créer une nouvelle figure pour le premier graphique
        ax1 = fig1.add_subplot(111)
        sns.scatterplot(x=y_test, y=y_pred, color="blue", label="Prédictions sur jeu de test", ax=ax1)
        sns.scatterplot(x=y_train, y=y_pred_train, color="orange", label="Prédictions sur jeu d'entraînement", ax=ax1)
        ax1.plot(y_test, y_test, color="red", label="Idéal (y_test = y_pred)")
        ax1.set_xlabel("Valeurs Réelles (J-D)")
        ax1.set_ylabel("Prédictions (J-D)")
        ax1.set_title(f"Corrélation des valeurs réelles et prédites\n modèle {model_choice}", fontsize=12)
        ax1.legend()
        # 📌 Afficher le premier graphique dans Streamlit
        st.pyplot(fig1,use_container_width=True)
 
if page == pages[4] : 
    scroll_to_top() # appel de la fonction pour remonter en haut de la page
    st.write("# Prédictions")
    
    st.write("## Prédictions des variables explicatives")
    col1, col2 = st.columns([4, 5])  # La première colonne est plus large pour le texte
    with col1:
        st.write("### Modèle prédictif : stationnarité et saisonnalité")
    with col2:
        box_ticked = st.checkbox("", label_visibility="hidden")

    # Action conditionnelle si la checkbox est cochée
    if box_ticked:
        ## Appel du df_ML0_pred à charger depuis 
        # Construire le chemin relatif vers le fichier CSV
        chemin_df_ML0 = os.path.join(os.getcwd(), "df_ML0_pred.csv")
        # Charger le fichier CSV avec pandas
        df_ML0 = pd.read_csv(chemin_df_ML0)

        df_ML0['year']=df_ML0.year.astype(str)
        #df_ML0['population']=df_ML0.population.astype(int)
        
        ## Liste des variables à  prédire
        liste_variable_to_pred = ["population", "cumulative_co2", "methane","nitrous_oxide","J-D","oil_co2", "gas_co2", "coal_co2", "land_use_change_co2", "others_co2","year"] 
        
        ### Mise en parallèle des informations
        col1, col2 = st.columns([5,4])  # Ajustez les proportions si nécessaire
            # Afficher la première image dans la première colonne
        with col1:
            st.write("##### Données annuelles historiques 1880-2023")
        
            # Champs pour entrer manuellement les valeurs de X et Y
            subcol1, subcol2 = st.columns([1,1])
            with subcol1:
                X = st.number_input("Nombre d'entrées les plus anciennes :", min_value=1, max_value=len(df_ML0), value=5, step=1)
            with subcol2:
                Y = st.number_input("Nombre d'entrées les plus récentes :", min_value=1, max_value=len(df_ML0), value=5, step=1)

            # Extraction des X premières et Y dernières entrées
            df_top = df_ML0.head(X)
            df_bottom = df_ML0.tail(Y)
            df_combined = pd.concat([df_top, df_bottom])

            # Afficher le tableau combiné dans Streamlit
            st.write(f"Affichage des {X} entrées les plus anciennes et {Y} plus récentes")
            st.dataframe(df_combined)
        
        with col2:
            # Ajouter une liste déroulante pour sélectionner une variable
            selected_variable = st.selectbox(
                "##### Sélectionnez une variable explicative", 
                liste_variable_to_pred )
            
            if selected_variable:
                #st.subheader(f"Graphique : {selected_variable} en fonction des années")

                # Création du graphique avec Matplotlib
                fig, ax = plt.subplots(figsize=(3, 2))  # Taille adaptée
                ax.plot(df_ML0['year'], df_ML0[selected_variable], linestyle='-', linewidth=1.0)
                
                # Personnalisation du graphique
                ax.set_title(f"{selected_variable} (données initiales)", fontsize=6)
                ax.set_xticks(np.arange(0, len(df_ML0['year']), 10))
                ax.tick_params(axis='x', labelrotation=25, labelsize=5)
                ax.set_xlabel("Années", fontsize=5)
                ax.tick_params(axis='y', labelsize=5)
                ax.set_ylabel(selected_variable, fontsize=5)
                ax.grid()
                # Afficher le graphique dans Streamlit
                st.pyplot(fig, use_container_width=False)
        
        # affichage des modèles de prédictions, sélection et animation du bon choix
        # Boutons radio pour choisir un modèle (un seul groupe)
        selected_model = st.radio(
            "Quel modèle choisir alors ?", 
            ["ARMA", "ARIMA", "SARIMA"], 
            index=0,  # Option par défaut (ARMA est pré-sélectionné)
            horizontal=True  # Affiche les boutons radio côte à côte
        )

        # Afficher le modèle sélectionné
        #st.write(f"Le modèle prédictif à envisager est donc... **{selected_model}**")

        # Si ARIMA est sélectionné, afficher une mini animation
        if selected_model == "ARIMA":
            st.success("Bravo ! ARIMA est le bon choix dans ce cas ! 🎉👍🎉")
            st.write("# ")
            st.write("## Prédictions ARIMA variables explicatives")
            st.write("### Meilleure combinaison des hyperparamètres")

            col1, col2 = st.columns([1,1])  # Ajustez les proportions si nécessaire
            with col1:
                text50 = """
                <span style="font-size:20px">
                <div style="text-indent: 20px;">
                Valeurs des paramètres (p, d, q) à  tester : <br><br>
                
                <span style="color: black;font-size:16px">
                
                <div style="text-indent: 70px;">
                p_values = [0, 1, 2, 3, 4, 5, 6] <br>
                
                <div style="text-indent: 70px;">
                d_values = [0, 1, 2, 3] <br>
                
                <div style="text-indent: 70px;">
                q_values = [0, 1, 2, 3, 4, 5, 6]<br><br><br>
                <div style="text-indent: 40px;">
                                
                <span style="color: red;font-size:20px">
                <div style="text-indent: 20px;">
                
                <span>
                """
                st.markdown(text50, unsafe_allow_html=True)
                if st.button("Tic Tac..."):
                    st.session_state.button_clicked = False
                    st.session_state.tableau_others_co2_checked = False
                # Vérification et initialisation des variables d'état
                if "button_clicked" not in st.session_state:
                    st.session_state.button_clicked = False
                if "tableau_others_co2_checked" not in st.session_state:
                    st.session_state.tableau_others_co2_checked = False

                # Bouton "Tic, Tac, Tic, Tac"
                if st.button("Driiiiing !"):
                    st.session_state.button_clicked = True  # Mémorise que le bouton a été cliqué

                # Si le bouton a été cliqué, afficher le contenu
                if st.session_state.button_clicked:
                    subcol1, subcol2 = st.columns([1.5, 2.5])
                    with subcol1:
                        st.write("12 minutes plus tard !")
                        st.markdown("<h2 style='text-align: center;'>⏰</h2>", unsafe_allow_html=True)
                    with subcol2:
                        st.markdown("<p style='color:red;'>🚨 La variable 'others_co2' a un RSME en erreur >>>>>>></p>", unsafe_allow_html=True)

                    with col2:
                        # Afficher le tableau complet
                        st.write("Tableau de la meilleure combinaison des hyperparamètres **par variable**", unsafe_allow_html=True)
                        # Construire le chemin relatif vers le fichier CSV
                        chemin_df_best_param = os.path.join(os.getcwd(), "df_best_param_var_incl_othco2_NA.csv")
                        # Charger le fichier CSV avec pandas
                        df_best_param_var_incl_othco2_NA = pd.read_csv(chemin_df_best_param)
                        
                        st.dataframe(df_best_param_var_incl_othco2_NA)

                        # Checkbox pour le tableau spécifique
                        tableau_others_co2 = st.checkbox(
                            "tableau others_co2",
                            label_visibility="hidden",
                            value=st.session_state.tableau_others_co2_checked
                        )

                        # Mémoriser l'état du checkbox
                        st.session_state.tableau_others_co2_checked = tableau_others_co2

                        # Afficher le tableau "others_co2" si le checkbox est coché
                        if st.session_state.tableau_others_co2_checked:
                            st.write(
                                "Tableau de la meilleure combinaison des hyperparamètres pour <u>**'others_co2'**</u>",
                                unsafe_allow_html=True,
                            )
                            # Construire le chemin relatif vers le fichier CSV
                            chemin_df_best_param = os.path.join(os.getcwd(), "df_best_param_var_expl_others_co2.csv")
                            # Charger le fichier CSV avec pandas
                            df_best_param_var_expl_others_co2 = pd.read_csv(chemin_df_best_param)

                            st.dataframe(df_best_param_var_expl_others_co2)
                          
                            others_co2_ticked = st.checkbox("Analyse 'others_co2'", label_visibility="visible") 
            
                            if others_co2_ticked:
                                st.write("### Analyse de la variable 'others_co2'")
                                # informations spécifiques à l'analyse de "others_co2"
                                # ## liste des années avec valeurs négatives
                                annees_others_co2_negatif = df_ML0.loc[df_ML0['others_co2'] < 0, 'year'].tolist()
                                annees_others_co2_negatif_lineaire = ", ".join(annees_others_co2_negatif)
                                st.write("(1) Années où <u>**'others_co2' < 0**</u> :", annees_others_co2_negatif_lineaire, unsafe_allow_html=True)

                                ## liste des années avec valeurs < 1
                                annees_others_co2_inf_1 = df_ML0.loc[df_ML0['others_co2'] < 1, 'year'].tolist()
                                annees_others_co2_inf_1_lineaire = ", ".join(annees_others_co2_inf_1)
                                st.write("(2) Années où <u>**'others_co2' < 1**</u> :", annees_others_co2_inf_1_lineaire, unsafe_allow_html=True)

                                st.write(f"\n(3) <u>**Période à  retirer**</u> pour retester le modèle de prédiction sur 'others_co2': **de {min(annees_others_co2_inf_1)} à  {max(annees_others_co2_inf_1)}**.", unsafe_allow_html=True)

            col1,col2 = st.columns([0.1,5])
            with col2:
                box_ticked_2 = st.checkbox("ouverture section suivante", label_visibility="hidden")

    # Action conditionnelle si la checkbox est cochée
            if box_ticked_2:

                st.write("# ")
                st.write("............................................................................................................................................................................................................................................................................................................")
                
                ######################## PREDICTION ARIMA VAR EXPLICATIVES ########################################

                st.markdown("# ")
                
                st.write("### Prédictions ARIMA(p,d,q) des variables prédictives")
                st.write("#### Tableau des valeurs prédites par variable explicatives")

                ## Liste des variables à  prédire
                liste_variable_to_pred_2 = ["population", "cumulative_co2","methane","nitrous_oxide","oil_co2", "gas_co2", "coal_co2", "land_use_change_co2", "others_co2"] 
                
                # Ajouter une liste déroulante pour sélectionner une variable
                col1, col2 = st.columns([1,3]) 
                with col1:
                    selected_variable_2 = st.selectbox(
                    "###### Choisissez une variable explicative", 
                    liste_variable_to_pred_2 ) 

                    st.write("Prédictions: 2024-2028 & ...& 2046-2050")
                    # Construire le chemin relatif vers le fichier CSV
                    chemin_df_future_clean = os.path.join(os.getcwd(), "df_future_clean.csv")
                    # Charger le fichier CSV avec pandas
                    df_future_clean = pd.read_csv(chemin_df_future_clean)
                    
                    df_future_clean2=df_future_clean
                    df_future_clean2.year=df_future_clean2.year.astype(str)
                    df_future_clean2=df_future_clean2.set_index('year', drop=False)
                    
                    df_future_clean2_top = df_future_clean2.head(5)
                    df_future_clean2_bottom = df_future_clean2.tail(5)
                    df_future_clean2_combined = pd.concat([df_future_clean2_top, df_future_clean2_bottom])
                    st.dataframe(df_future_clean2_combined[selected_variable_2])
                with col2:
                    # Construire les chemins relatifs pour chaque fichier
                    chemin_merged_df = os.path.join(os.getcwd(), "merged_df_pred.csv")
                    chemin_df_param_excl_othco2 = os.path.join(os.getcwd(), "df_best_param_var_expl_excl_othco2.csv")
                    chemin_df_param_others_co2 = os.path.join(os.getcwd(), "df_best_param_var_expl_others_co2.csv")

                    # Charger les fichiers CSV avec pandas
                    merged_df = pd.read_csv(chemin_merged_df)
                    df_best_param_var_expl_excl_othco2 = pd.read_csv(chemin_df_param_excl_othco2)
                    df_best_param_var_expl_others_co2 = pd.read_csv(chemin_df_param_others_co2)

                    df_best_param_var_expl = pd.concat([df_best_param_var_expl_excl_othco2, df_best_param_var_expl_others_co2])

                    if selected_variable_2:
                        # Extraction des paramètres pour la variable sélectionnée
                        params_row = df_best_param_var_expl[df_best_param_var_expl['Variable'] == selected_variable_2]
                        if not params_row.empty:
                            p = int(params_row['p'].values[0])
                            d = int(params_row['d'].values[0])
                            q = int(params_row['q'].values[0])

                            # Série correspondante
                            series = merged_df[selected_variable_2]

                            # Création du graphique
                            fig, ax = plt.subplots(figsize=(6, 3))
                            
                            # Données réelles
                            ax.plot(df_ML0['year'].astype(str), series, label='Données réelles', color='blue', linewidth=1)

                            # Ligne de séparation (données réelles vs prédictions)
                            ax.axvline(x=max(df_ML0['year']), color='green', linewidth=0.5, linestyle='--')

                            # Prédictions sur les 20% dernières années
                            train_size = int(len(series) * 0.8)
                            train, test = series[:train_size], series[train_size:]
                            model = ARIMA(np.log(train), order=(p, d, q))
                            model_fit = model.fit()
                            forecast_log = model_fit.forecast(steps=len(test))
                            forecast_exp = np.exp(forecast_log)
                            ax.plot(test.index, forecast_exp, label='Prédictions (20% dernières années)', color='red', marker='o', markersize=1.0, linestyle='none')

                            # Prédictions futures (2024-2050)
                            ax.plot(df_future_clean2.index.astype(str), df_future_clean2[selected_variable_2], label='Prédictions futures 2024-2050', color='green', linewidth=1)

                            # Titres et légendes
                            ax.set_title(f"Données réelles, Prédictions test et Prédictions futures pour '{selected_variable_2}'\n"
                                        f"Modèle ARIMA({p},{d},{q})", fontsize=7)
                            xticks = np.arange(0, 171, 10)
                            ax.set_xticks(xticks)
                            ax.set_xlabel("Années", fontsize=6)
                            ax.set_ylabel(selected_variable_2, fontsize=6)
                            ax.legend(fontsize=6)
                            ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
                            ax.tick_params(axis='x', rotation=25, labelsize=5)
                            ax.tick_params(axis='y', labelsize=5)

                            # Affichage dans Streamlit
                            st.pyplot(fig)
                        else:
                            st.error(f"Les paramètres pour la variable '{selected_variable_2}' n'ont pas été trouvés dans df_best_param_var_expl.")
            
                col1,col2 = st.columns([0.1,5])
                with col2:
                    box_ticked = st.checkbox("prédictions var cible", label_visibility="hidden")
                st.write(".............................................................................................................................................................................................................................................................................................................................................")                        
                if box_ticked:
                    
                    st.write("# ")
                    st.write("### Prédictions en MACHINE LEARNING de la variable cible")
                    ################################# PREPA INFO POUR LE MODELE ############################################
                    # Chargement des df
                    # Construire les chemins relatifs pour chaque fichier
                    chemin_df_ML1 = os.path.join(os.getcwd(), "df_ML1_pred.csv")
                    chemin_Var_expl_futures = os.path.join(os.getcwd(), "Var_expl_futures_pred.csv")
                    chemin_Var_expl_futures_norm = os.path.join(os.getcwd(), "Var_expl_futures_norm_pred.csv")

                    # Charger et traiter df_ML1
                    df_ML1 = pd.read_csv(chemin_df_ML1)
                    df_ML2 = df_ML1.drop(["land_use_change_co2", "nitrous_oxide", "others_co2", "cumulative_co2"], axis=1)
                    df_ML2 = df_ML2.drop("year.1", axis=1)
                    df_ML2['year'] = df_ML2['year'].astype(str)
                    df_ML2 = df_ML2.set_index('year', drop=False)

                    # Charger et traiter Var_expl_futures
                    Var_expl_futures = pd.read_csv(chemin_Var_expl_futures)
                    Var_expl_futures['year'] = Var_expl_futures['year'].astype(str)
                    Var_expl_futures = Var_expl_futures.set_index('year', drop=False)

                    # Charger et traiter Var_expl_futures_norm
                    Var_expl_futures_norm = pd.read_csv(chemin_Var_expl_futures_norm)
                    Var_expl_futures_norm['year'] = Var_expl_futures_norm['year'].astype(str)
                    Var_expl_futures_norm = Var_expl_futures_norm.set_index('year', drop=False)
                    Var_expl_futures_norm = Var_expl_futures_norm.drop('year.1', axis=1)
                    
                    ## Définition de l'horizon de prédiction
                    horizon_prediction = 2050 #année max
                    ## Calcul du nombre d'années à  prédire (steps)
                    prediction_steps = horizon_prediction - max(df_ML1["year"])
                    
                    # Etablissement du dataframe avec les variables explicatives retenues (suppression des autres)
                    
                    X = df_ML2.drop("J-D", axis = 1) # On conserve l'année comme une variable explicative
                    y = df_ML2["J-D"] # la variable cible est donc l'écart de température annuel par rapport à  la moyenne de référence de la période 1951-1980
                    
                    split_ratio = 0.25
                    shuffle_on_off = True

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = split_ratio, shuffle=shuffle_on_off, random_state=42)
                    
                    #y_pred = loaded_RFR_best_model.predict(X_test)  ## Prédire avec le modèle sur le jeu de données TEST
                    #y_all_pred = loaded_RFR_best_model.predict(X)   ## Prédire avec le modèle sur le jeu de données (1880-2023)      
                    
                    ################################# SELECTION INFO MODELE ############################################
                    subcol1, subcol2 = st.columns([1, 1])
                    with subcol1:
                        st.markdown("<p style='color:black;'><b><u><i>Variables explicatives retenues dans le modèle réduit de Random Forest Regression:</b></u></i></p>",unsafe_allow_html=True)
                        liste_variable_exp_RFR_best = ["population", "year","oil_co2", "gas_co2", "coal_co2","methane"]
                        # Mise en forme pour affichage des var effectives
                        liste_variable_exp_RFR_best_lineaire = " ,  ".join(liste_variable_exp_RFR_best)
                        st.write("<u>**6 variables**</u> : ",liste_variable_exp_RFR_best_lineaire,unsafe_allow_html=True)

                        liste_options_normalisation =[" ","A - Period 1880-2050 sans normalisation", "B - Period 2024-2050 sur Scaler MinMax(1880-2023)", "C - Period 1880-2050 sur Scaler MinMax(1880-2050)", "(D=B+C) - Double normalisation"]
                        
                        modele_normalisation = st.selectbox(
                        "Modèle de normalisation des données prédites 2024-2050 des variables explicatives", 
                        liste_options_normalisation)
                    
                    with subcol2:
                        st.markdown("<p style='color:black;'><b><u><i>Paramètres d'entrainement du modèle:</b></u></i></p>",unsafe_allow_html=True)
                        st.write("Split Ratio :",split_ratio)
                        st.write("Shuffle =",shuffle_on_off)
                        st.write("Steps de prédiction :",prediction_steps)
                    
                    ################################# EXECUTION DU MODELE SUIVANT CHOIX DE NORMALISATION ############################################
                    # ne rien faire si option " " sélectionnée
                    if modele_normalisation != " ":
                        # Autrement...
                        # Exécution des actions spécifiques selon l'option choisie
                        if modele_normalisation == "A - Period 1880-2050 sans normalisation":
                            
                            # Entraîneemnt du le modèle
                            #--------------------------
                            RFR_best = RandomForestRegressor()
                            RFR_best.fit(X_train, y_train)
                            ## Prédire avec le modèle sur le jeu de données TEST
                            y_pred = RFR_best.predict(X_test)
                            ## Prédire avec le modèle sur l'ENSEMBLE du jeu de données
                            y_all_pred = RFR_best.predict(X)   
                            
                                                    ## Etablissement d'un dataframe complet (1880-2050) avant normalisation
                            df_ML1.year = df_ML1.year.astype('str')
                            X_1880_2023 = df_ML1[['year', 'population', 'oil_co2', 'gas_co2', 'coal_co2','methane']]
                            X_1880_2023['year'] = X_1880_2023['year'].astype(str)
                            X_1880_2023 = X_1880_2023.set_index('year', drop=False)

                            X_1880_2050 = pd.concat([X_1880_2023, Var_expl_futures_norm])
                            X_1880_2050.year = X_1880_2050.year.astype('str')

                            ## Prédire avec le modèle sur l'ENSEMBLE du jeu de données NORMALISEES
                            y_1880_2050 = RFR_best.predict(X_1880_2050) # RFR_best est le modèle
                            
                            # GRAPHE
                            #-------
                            col1, col2, col3 = st.columns([0.5, 6, 2.5])
                            with col2:
                                plt.figure(figsize=(8, 6))

                                # courbe des valeurs réelles passées
                                plt.plot(y.index, y, label='Valeurs réelles 1880-2023', linewidth=1.5) #reprendre l'ensemble des valeurs cibles du dataset initial

                                # courbe de prédictions sur jeu de test
                                plt.plot(y_test.index, y_pred, label='Prédict modèle y_test split 0.25',color='black', marker='o',markersize=4.5, linestyle='none') 

                                # courbe de prédictions sur la totalité du dataset
                                plt.plot(y.index, y_all_pred, label='Prédict modèle y_all 1880-2023',color='red', linewidth=1.5, linestyle='--') 

                                # ligne horizontale = moyenne période 1951-1980
                                plt.plot(y.index, [0] * len(y.index),  label='Moyenne 1951-1980',color='orange', linewidth=1.5, linestyle='--') 
                                plt.plot(Var_expl_futures_norm.index, [0] * len(Var_expl_futures_norm.index),color='orange', linewidth=1.5, linestyle='--') 

                                # délimiteur passé/futur
                                plt.axvline(x= max(y.index), color='green',linewidth = 1.0,linestyle='--')


                                # courbes prédictions 1880-2050
                                plt.plot(X_1880_2050.index[-prediction_steps:], y_1880_2050[-prediction_steps:], label='Prédictions 2024-2050',color='blue', linewidth=1.5, linestyle='--') 

                                # paramètres du graphe
                                plt.title(f"Evolution réelle passée (1880-2023) et prédictions (1880-2023)\n Ecarts de température vs. moyenne périodique (1951-1980)\n Modèle RFR_best (split_ratio de {split_ratio} / shuffle {shuffle_on_off})", fontsize=9)
                                xticks = np.arange(0, 171, 10)
                                plt.xticks(xticks, rotation=25, fontsize = 6)
                                plt.yticks(fontsize = 6)
                                plt.xlabel('Années', fontsize = 8)
                                plt.ylabel("Ecart de température °C", fontsize = 8)
                                plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
                                plt.legend()
                                # Affichage du graphe dans Streamlit
                                st.pyplot(plt, use_container_width=True)

                        elif modele_normalisation == "B - Period 2024-2050 sur Scaler MinMax(1880-2023)":
                            
                            # Entraîneemnt du le modèle
                            #--------------------------
                            RFR_best = RandomForestRegressor()
                            RFR_best.fit(X_train, y_train)
                            ## Prédire avec le modèle sur le jeu de données TEST
                            y_pred = RFR_best.predict(X_test)
                            ## Prédire avec le modèle sur l'ENSEMBLE du jeu de données
                            y_all_pred = RFR_best.predict(X)   
                            #st.write(y_all_pred)
                            ## Etablissement d'un dataframe complet (1880-2050) avec VAR-EXPL_FUTURES_NORM (sur base de la normalisation 1880-2023 =>>> val > 1)
                            df_ML1.year = df_ML1.year.astype('str')
                            X_1880_2023 = df_ML1[['year', 'population', 'oil_co2', 'gas_co2', 'coal_co2','methane']].set_index('year', drop=False)
                            X_1880_2050 = pd.concat([X_1880_2023, Var_expl_futures_norm]) 
                            X_1880_2050.year = X_1880_2050.year.astype('str')
                            #st.write("X_1880_2050 start", X_1880_2050.head()) ################################
                            #st.write("X_1880_2050 fin", X_1880_2050.tail()) ################################

                            y_1880_2050 = RFR_best.predict(X_1880_2050)
                                                    
                            # GRAPHE
                            #-------
                            col1, col2, col3 = st.columns([0.5, 6, 2.5])
                            with col2:
                                plt.figure(figsize=(8, 6))
                            
                                # Courbe des valeurs réelles passées
                                plt.plot(y.index, y, label='Valeurs réelles 1880-2023', linewidth=1.5)

                                # Courbe de prédictions sur jeu de test
                                plt.plot(y_test.index, y_pred, label='Prédict modèle y_test split 0.25', color='black', marker='o', markersize=4.5, linestyle='none')

                                # Courbe de prédictions sur la totalité du dataset
                                plt.plot(y.index, y_all_pred, label='Prédict modèle y_all 1880-2023', color='red', linewidth=1.5, linestyle='--')

                                # Ligne horizontale = moyenne période 1951-1980
                                plt.plot(y.index, [0] * len(y.index), label='Moyenne 1951-1980', color='orange', linewidth=1.5, linestyle='--')
                                plt.plot(Var_expl_futures_norm.index, [0] * len(Var_expl_futures_norm.index), color='orange', linewidth=1.5, linestyle='--')

                                # Délimiteur passé/futur
                                plt.axvline(x=max(y.index), color='green', linewidth=1.0, linestyle='--')

                                # Courbes prédictions 1880-2050
                                plt.plot(X_1880_2050.index[-prediction_steps:], y_1880_2050[-prediction_steps:], label='Prédictions 2024-2050', color='blue', linewidth=1.5, linestyle='--')

                                # Paramètres du graphe
                                plt.title(f"Evolution réelle passée (1880-2023) et prédictions (1880-2023)\n Ecarts de température vs. moyenne périodique (1951-1980)\n Modèle RFR_best (split_ratio de {split_ratio} / shuffle {shuffle_on_off})", fontsize=9)
                                xticks = np.arange(0, 171, 10)
                                plt.xticks(xticks, rotation=25, fontsize=8)
                                plt.yticks(fontsize=8)
                                plt.xlabel('Années', fontsize=10)
                                plt.ylabel("Ecart de température °C", fontsize=10)
                                plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
                                plt.legend()

                                # Affichage du graphe dans Streamlit
                                st.pyplot(plt, use_container_width=True)

                        elif modele_normalisation == "C - Period 1880-2050 sur Scaler MinMax(1880-2050)":
                            
                            # Entraîneemnt du le modèle
                            #--------------------------
                            RFR_best = RandomForestRegressor()
                            RFR_best.fit(X_train, y_train)
                            ## Prédire avec le modèle sur le jeu de données TEST
                            y_pred = RFR_best.predict(X_test)
                            ## Prédire avec le modèle sur l'ENSEMBLE du jeu de données
                            y_all_pred = RFR_best.predict(X)   

                            ## Etablissement d'un dataframe complet (1880-2050) avec VAR_EXPL_FUTURES (valeurs NON NORMALISEES) 
                            merged_df.year = merged_df.year.astype('str')
                            X_1880_2023 = merged_df[['year', 'population', 'oil_co2', 'gas_co2', 'coal_co2','methane']].set_index('year', drop=False)
                            X_1880_2050 = pd.concat([X_1880_2023, Var_expl_futures])

                            ### Liste des variables à borner/normaliser EN UNE FOIS SUR PERIODE 1880-2050
                            bornage_1880_2050 = ["population", "oil_co2", "gas_co2", "coal_co2","methane"]
                            import pandas as pd

                            from sklearn.preprocessing import MinMaxScaler
                            scaler = MinMaxScaler()
                            for var in bornage_1880_2050:
                                X_1880_2050[var] = scaler.fit_transform(X_1880_2050[[var]])

                            y_1880_2050 = RFR_best.predict(X_1880_2050) # RFR_best est le modèle
                            
                            # GRAPHE
                            #-------
                            col1, col2, col3 = st.columns([0.5, 6, 2.5])
                            with col2:
                                plt.figure(figsize=(8, 6))

                                # Courbe des valeurs réelles passées
                                plt.plot(y.index, y, label='Valeurs réelles 1880-2023', linewidth=1.5)

                                # Courbe de prédictions sur jeu de test
                                plt.plot(y_test.index, y_pred, label='Prédict modèle y_test split 0.25', color='black', marker='o', markersize=4.5, linestyle='none')

                                # Courbe de prédictions sur la totalité du dataset
                                plt.plot(y.index, y_all_pred, label='Prédict modèle y_all 1880-2023', color='red', linewidth=1.5, linestyle='--')

                                # Ligne horizontale = moyenne période 1951-1980
                                plt.plot(X_1880_2050.index, [0] * len(X_1880_2050.index), label='Moyenne 1951-1980', color='orange', linewidth=1.5, linestyle='--')

                                # Délimiteur passé/futur
                                plt.axvline(x=max(y.index), color='green', linewidth=1.0, linestyle='--')

                                # Découplement des prédictions
                                y_1880_2050_1 = y_1880_2050[:-prediction_steps]
                                y_1880_2050_2 = y_1880_2050[-prediction_steps:]

                                plt.plot(X_1880_2050.index[:-prediction_steps], y_1880_2050_1, label='Prédictions 1880-2023', color='blue', linewidth=1.0, linestyle='-')
                                plt.plot(X_1880_2050.index[-prediction_steps:], y_1880_2050_2, label='Prédictions 2024-2050', color='blue', linewidth=1.5, linestyle='--')

                                # Paramètres du graphe
                                plt.title(f"Evolution réelle passée (1880-2023) et prédictions (1880-2023)\n Ecarts de température vs. moyenne périodique (1951-1980)\n Modèle RFR_best (split_ratio de {split_ratio} / shuffle {shuffle_on_off})", fontsize=9)
                                xticks = np.arange(0, 171, 10)
                                plt.xticks(xticks, rotation=25, fontsize=8)
                                plt.yticks(fontsize=8)
                                plt.xlabel('Années', fontsize=10)
                                plt.ylabel("Ecart de température °C", fontsize=10)
                                plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
                                plt.legend()

                                # Streamlit display
                                st.pyplot(plt,use_container_width=True)

                        elif modele_normalisation == "(D=B+C) - Double normalisation":
                            # Entraîneemnt du le modèle
                            #--------------------------
                            RFR_best = RandomForestRegressor()
                            RFR_best.fit(X_train, y_train)
                            ## Prédire avec le modèle sur le jeu de données TEST
                            y_pred = RFR_best.predict(X_test)
                            ## Prédire avec le modèle sur l'ENSEMBLE du jeu de données
                            y_all_pred = RFR_best.predict(X) 
                            
                            ## Prédire avec le modèle sur le jeu de données TEST
                            y_pred = RFR_best.predict(X_test)

                            ## Prédire avec le modèle sur l'ENSEMBLE du jeu de données
                            y_all_pred = RFR_best.predict(X)
                            ## Prédire avec le modèle sur le jeu de données (1880-2023)
                            ## Etablissement d'un dataframe complet (1880-2050) par association des jeux normalisés sur base 1880-2023
                            df_ML1.year = df_ML1.year.astype('str')
                            X_1880_2023 = df_ML1[['year', 'population', 'oil_co2', 'gas_co2', 'coal_co2','methane']].set_index('year', drop=False)
                            X_1880_2050 = pd.concat([X_1880_2023, Var_expl_futures_norm])
                            X_1880_2050['year'] = X_1880_2050['year'].astype('str')
                            # Renormalisation de l'ensemble du jeu 1880-2050
                            bornage_1880_2050 = ["population", "oil_co2", "gas_co2", "coal_co2","methane"]
                            import pandas as pd

                            from sklearn.preprocessing import MinMaxScaler
                            scaler = MinMaxScaler()
                            for var in bornage_1880_2050:
                                X_1880_2050[var] = scaler.fit_transform(X_1880_2050[[var]])
                            ## Prédire avec le modèle RE-CHARGE sur l'ENSEMBLE du jeu de données DOUBLEMENT NORMALISEES
                            y_1880_2050_norm = RFR_best.predict(X_1880_2050) # RFR_best est le modèle

                            # GRAPHE
                            #-------
                            col1, col2, col3 = st.columns([0.5, 6, 2.5])
                            with col2:
                                plt.figure(figsize=(8, 6))

                                # Courbe des valeurs réelles passées
                                plt.plot(y.index, y, label='Valeurs réelles 1880-2023', linewidth=1.5)

                                # Courbe de prédictions sur jeu de test
                                plt.plot(y_test.index, y_pred, label='Prédict modèle y_test split 0.25', color='black', marker='o', markersize=4.5, linestyle='none')

                                # Courbe de prédictions sur la totalité du dataset
                                plt.plot(y.index, y_all_pred, label='Prédict modèle y_all 1880-2023', color='red', linewidth=1.5, linestyle='--')

                                # Ligne horizontale = moyenne période 1951-1980
                                plt.plot(y.index, [0] * len(y.index), label='Moyenne 1951-1980', color='orange', linewidth=1.5, linestyle='--')
                                plt.plot(Var_expl_futures_norm.index, [0] * len(Var_expl_futures_norm.index), color='orange', linewidth=1.5, linestyle='--')

                                # Délimiteur passé/futur
                                plt.axvline(x=max(y.index), color='green', linewidth=1.0, linestyle='--')

                                # Courbes prédictions 1880-2050 (double normalisation)
                                plt.plot(X_1880_2050.index, y_1880_2050_norm, label='Prédictions 2024-2050', color='blue', linewidth=1.0, linestyle='--')

                                # Paramètres du graphe
                                plt.title(f"Evolution réelle passée (1880-2023) et prédictions (1880-2023)\n Ecarts de température vs. moyenne périodique (1951-1980)\n Modèle RFR_best (split_ratio de {split_ratio} / shuffle {shuffle_on_off})", fontsize=9)
                                xticks = np.arange(0, 171, 10)
                                plt.xticks(xticks, rotation=25, fontsize=8)
                                plt.yticks(fontsize=8)
                                plt.xlabel('Années', fontsize=10)
                                plt.ylabel("Ecart de température °C", fontsize=10)
                                plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
                                plt.legend()

                                # Affichage dans Streamlit
                                st.pyplot(plt,use_container_width=True)

                        elif modele_normalisation == "":
                            st.write("Modèle sélectionné :",modele_normalisation)
                    st.write("# ")
                    st.write("# ")
                    st.write("# ")
                    st.write("# ")
                    
                    ####################### PREDICTION ARIMA VAR CIBLE ########################

                    col1,col2 = st.columns([0.1,5])
                    with col2:
                        box_ticked = st.checkbox("prédictions ARIMA", label_visibility="hidden")
                    st.write(".............................................................................................................................................................................................................................................................................................................................................")                        
                    if box_ticked:
                            st.write("### Prédictions ARIMA de la variable cible")
                            
                            # Construire le chemin relatif vers le fichier CSV
                            chemin_df_future_y_temp = os.path.join(os.getcwd(), "df_future_y_temp.csv")
                            # Charger le fichier CSV avec pandas
                            df_future_y_temp = pd.read_csv(chemin_df_future_y_temp)

                            col1, col2, col3 = st.columns([1,1,0.4])  # Ajustez les proportions si nécessaire
                            with col1:
                                text50 = """
                                <span style="font-size:20px">
                                <div style="text-indent: 20px;">
                                <b>Valeurs des paramètres (p, d, q) à tester :</b> <br>
                                </div>
                                <div style="color: black;font-size:16px"; text-align:justify; padding-left:0px; text-indent:100px;">
                                <div style="text-indent: 70px;">
                                p_values = [0, 1, 2, 3, 4, 5, 6]<br>
                                <div style="text-indent: 70px;">
                                d_values = [0, 1, 2, 3, 4, 5, 6]<br>
                                <div style="text-indent: 70px;">
                                q_values = [0, 1, 2, 3, 4, 5, 6]<br>
                                </div>
                                <span>
                                """
                                st.markdown(text50, unsafe_allow_html=True)
                            with col2:
                                st.write("Tableau de la meilleure combinaison des hyperparamètres pour la variable cible où 'J-D' = Ecarts de température moyens annuels de Janvier à Décembre")

                                # Construire le chemin relatif vers le fichier CSV
                                df_best_parametres_y_temp = os.path.join(os.getcwd(), "df_best_parametres_y_temp.csv")
                                # Charger le fichier CSV avec pandas
                                df_best_parametres_y_temp = pd.read_csv(df_best_parametres_y_temp)

                                st.dataframe(df_best_parametres_y_temp)
                            
                            # GRAPHE ARIMA
                            #-------------
                            # Chargement des données
                            # Construire les chemins relatifs pour chaque fichier
                            chemin_df_future_y_temp = os.path.join(os.getcwd(), "df_future_y_temp.csv")
                            chemin_merged_df = os.path.join(os.getcwd(), "merged_df_pred.csv")
                            chemin_df_ML0 = os.path.join(os.getcwd(), "df_ML0_pred.csv")

                            # Charger les fichiers CSV avec pandas
                            df_future_y_temp = pd.read_csv(chemin_df_future_y_temp)
                            merged_df = pd.read_csv(chemin_merged_df)
                            df_ML0 = pd.read_csv(chemin_df_ML0)  # Charge également df_ML0

                            # Extraction des séries
                            series = merged_df['J-D']

                            # Dimensions et polices ajustables
                            font_title_size = 18
                            font_axis_title_size = 12
                            font_tick_size = 15
                            border_width = 1.5  # Épaisseur de la bordure
                            # Création du graphique avec Plotly
                            fig_y_Arima = go.Figure()

                            # Courbe des données réelles
                            fig_y_Arima.add_trace(go.Scatter(
                                x=df_ML0['year'].astype(str),
                                y=series,
                                mode='lines',
                                name='Données réelles',
                                line=dict(color='blue', width=2)
                            ))
                            # Ligne de séparation (données réelles vs prédictions)
                            fig_y_Arima.add_vline(
                                x=2023,
                                line=dict(color='red', dash='dash', width=2),
                            )
                            fig_y_Arima.add_annotation(
                                x=143,
                                y=1.3,  # Position sur l'axe y (modifiez si nécessaire)
                                text="Fin des données réelles",
                                showarrow=False,
                                font=dict(color="blue"),
                                align="right",
                                xanchor="right",  # Aligner le texte à droite
                                yanchor="top"  # Positionner en bas
                            )
                            fig_y_Arima.add_annotation(
                                x=143,
                                y=1.0,  # Position sur l'axe y (modifiez si nécessaire)
                                text="Prédictions",
                                showarrow=False,
                                font=dict(color="green" ),
                                align="right",
                                xanchor="left",  # Aligner le texte à droite
                                yanchor="bottom"  # Positionner en bas
                            )
                            # Courbe des prédictions futures [2024 à 2050]
                            years_pred_range = np.arange(2024, 2050)
                            fig_y_Arima.add_trace(go.Scatter(
                                x=years_pred_range,
                                y=df_future_y_temp['J-D'],
                                mode='lines',
                                name='Prédictions futures 2024-2050',
                                line=dict(color='green', width=2)
                            ))
                            # Ligne horizontale = moyenne période 1951-1980
                            years_range = np.arange(1880, 2051)
                            fig_y_Arima.add_trace(go.Scatter(
                                x=years_range,
                                y=[0] * len(years_range),
                                mode='lines',
                                name='Moyenne 1951-1980',
                                line=dict(color='orange', width=2, dash='dash')
                            ))
                            # Ajout des titres, légendes et paramètres
                            fig_y_Arima.update_layout(
                                title={
                                    'text': "Données réelles, Prédictions test et Prédictions futures pour les écarts de température<br>Par rapport à la moyenne périodique 1951-1980<br>Modèle ARIMA",
                                    'x': 0.5,
                                    'xanchor': 'center',
                                    'yanchor': 'top'
                                },
                                title_font_size=16,
                                xaxis=dict(
                                    title='Années',
                                    tickmode='array',
                                    tickvals=np.arange(0, 171, 10),
                                    tickangle=25
                                ),
                                yaxis=dict(
                                    title="Écart de température (°C)"
                                ),
                                legend=dict(
                                    font=dict(size=14),
                                    x=0.02,  # Position horizontale sur le graphe (0 = à gauche, 1 = à droite)
                                    y=0.98,  # Position verticale sur le graphe (0 = en bas, 1 = en haut)
                                    bgcolor="rgba(255,255,255,0.7)",  # Fond semi-transparent pour mieux voir les données
                                    bordercolor="black",  # Couleur de la bordure de la légende
                                    borderwidth=1  # Épaisseur de la bordure
                                ),
                                width=1100,  # Largeur du graphique (modifiable ici)
                                height=700,  # Hauteur du graphique (modifiable ici)
                                template="plotly_white"
                            )
                            # Grille
                            fig_y_Arima.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='gray')
                            fig_y_Arima.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='gray')
                            
                            # Ajout de la fine bordure (personnalisation via CSS-like trick)
                            fig_y_Arima.update_layout(
                                paper_bgcolor="white",  # Couleur de fond de la page
                                plot_bgcolor="white",   # Couleur de fond du graphique
                                shapes=[
                                    dict(
                                        type="rect",
                                        xref="paper",
                                        yref="paper",
                                        x0=0,
                                        y0=0,
                                        x1=1,
                                        y1=1,
                                        line=dict(color="black", width=border_width, dash="solid")  # Fine bordure noire
                                    )
                                ]
                            )
                            # Affichage dans Streamlit
                            st.plotly_chart(fig_y_Arima)
                            st.session_state["fig_y_Arima_graph"] = fig_y_Arima
                        
if page == pages[5] : 
    scroll_to_top() # appel de la fonction pour remonter en haut de la page
    if "checkbox_state" not in st.session_state:
        st.session_state.checkbox_state = False

        checkbox = st.checkbox("conclusion", value=st.session_state.checkbox_state)
        st.session_state.checkbox_state = checkbox

    st.markdown("# Pertinence & Conclusion")
    st.markdown("## Pertinence : Sources alternatives vs. Nos Prédictions")

    text50 = """
    <span style="font-size:20px">
    <div style="text-indent: 20px;">
    <b>1 - Evolution de la population jusqu'en 2050: ONU vs. ARIMA

    <span style="color: green;font-size:16px">
    <div style="color: green;font-size:16px"; text-align:justify; padding-left:0px; text-indent:70px;">
    <br></b>Nos prédictions sont <b><u>alignées</b></u> sur la courbe basse (bleue) des scénarii de l'ONU, mais avec un inflection moindre à l'approche de 2050.
    Au-delà de l'horizon projeté (2050), on peut raisonnablement envisager que notre prédiction continue à croître alors que le scénario onusien le plus proche indique une population en baisse.
    </div>
    <br>
    <span>
    """
    st.markdown(text50, unsafe_allow_html=True)
    col1, col2 = st.columns([4,4])  # Ajustez les proportions si nécessaire
        # Afficher la première image dans la première colonne
    with col1:
        text51 = """
        <span style="font-size:14px">
        <div style="text-indent: 60px;">
        <br><div style="text-indent: 60px;"><b>Evolution population ONU: réalité 1950-2023 et prédictions 2024-2100
        
        <span>
        """
        st.markdown(text51, unsafe_allow_html=True)
        # Construire le chemin relatif vers l'image
        chemin_image = os.path.join(os.getcwd(), "Projections d'évolution de la population mondiale (ONU) 2024-2100.jpg")
        # Afficher l'image avec Streamlit
        st.image(chemin_image, caption="", use_column_width=True)
        
    with col2:
        #st.write("Evolution population: réalité 1880-2023 et prédictions ARIMA 2024-2050")

        # Construire les chemins relatifs pour chaque fichier
        chemin_df_ML0 = os.path.join(os.getcwd(), "df_ML0_pred.csv")
        chemin_merged_df = os.path.join(os.getcwd(), "merged_df_pred.csv")
        chemin_df_future_clean = os.path.join(os.getcwd(), "df_future_clean.csv")
        chemin_df_param_excl_othco2 = os.path.join(os.getcwd(), "df_best_param_var_expl_excl_othco2.csv")
        chemin_df_param_others_co2 = os.path.join(os.getcwd(), "df_best_param_var_expl_others_co2.csv")

        # Charger et traiter df_ML0
        df_ML0 = pd.read_csv(chemin_df_ML0)
        df_ML0['year'] = df_ML0.year.astype(int)

        # Charger et traiter merged_df
        merged_df = pd.read_csv(chemin_merged_df)
        merged_df['year'] = merged_df.year.astype(str)

        # Charger et traiter df_future_clean
        df_future_clean = pd.read_csv(chemin_df_future_clean)
        df_future_clean2 = df_future_clean.copy()
        df_future_clean2['year'] = df_future_clean2['year'].astype(str)
        df_future_clean2 = df_future_clean2.set_index('year', drop=False)

        # Charger et concaténer les fichiers des paramètres
        df_best_param_var_expl_excl_othco2 = pd.read_csv(chemin_df_param_excl_othco2)
        df_best_param_var_expl_others_co2 = pd.read_csv(chemin_df_param_others_co2)
        df_best_param_var_expl = pd.concat([df_best_param_var_expl_excl_othco2, df_best_param_var_expl_others_co2])
        
        # Extraction des paramètres pour la variable 'population'
        params_row = df_best_param_var_expl[:1]
        if not params_row.empty:
            p = int(params_row['p'].values[0])
            d = int(params_row['d'].values[0])
            q = int(params_row['q'].values[0])

            # Série de la population
            series = merged_df['population']

            # Prédictions sur les 20% dernières années
            train_size = int(len(series) * 0.8)
            train, test = series[:train_size], series[train_size:]

            # Modèle ARIMA
            model = ARIMA(np.log(train), order=(p, d, q))
            model_fit = model.fit()
            forecast_log = model_fit.forecast(steps=len(test))
            forecast_exp = np.exp(forecast_log)

            # Création du graphique avec Plotly
            fig = go.Figure()

            # Données réelles
            fig.add_trace(go.Scatter(
                x=df_ML0['year'].astype(str),
                y=series,
                mode='lines',
                name='Données réelles',
                line=dict(color='blue', width=2)
            ))

            # Ligne de séparation (données réelles vs prédictions)
            fig.add_vline(
                x=143,
                line=dict(color='green', dash='dash', width=1),
                annotation_text="Fin des données réelles",
                annotation_position="top left"
            )
            fig.add_annotation(
                x=143,
                y=0,  # Position sur l'axe y (modifiez si nécessaire)
                text="Prédictions",
                showarrow=False,
                font=dict(color="green"),
                align="right",
                xanchor="left",  # Aligner le texte à droite
                yanchor="bottom"  # Positionner en bas
)
            # Prédictions futures (2024-2050)
            fig.add_trace(go.Scatter(
                x=df_future_clean2.index.astype(str),
                y=df_future_clean2['population'],
                mode='lines',
                name='Prédictions futures 2024-2050',
                line=dict(color='green', width=2)
            ))

            # Mise en forme du graphique
            fig.update_layout(
                title=f"Données réelles, Prédictions test et Prédictions futures pour la variable 'population'<br>"
                    f"Modèle ARIMA({p},{d},{q})",
                title_font_size=12,
                xaxis=dict(
                    title='Années',
                    tickmode='array',
                    tickvals=np.arange(0, 171, 10),  # Année tous les 10 ans
                    tickangle=45
                ),
                yaxis=dict(
                    title='Population',
                    range=[0, 10e9],
                    tickmode='array',  # Quadrillage basé sur des valeurs fixes
                    tickvals=np.arange(0, 10e9 + 1, 1e9),  # Quadrillage tous les 1 milliard
                    #tickformat=".0e"  # Format pour afficher les nombres en notation scientifique si besoin 
                ),
                legend=dict(font=dict(size=8)),
                width=800,
                height=450,
                template="plotly_white"
            )
            st.plotly_chart(fig)

    text52 = """
    <span style="font-size:20px">
    <div style="text-indent: 20px;">
    <b>2 - Evolution des écarts de température jusqu'en 2050: GIEC vs. ARIMA

    <span style="color: green;font-size:16px">
    <div style="color: green;font-size:16px"; text-align:justify; padding-left:0px; text-indent:70px;">
    <br></b>Même si les écarts que nous utilisons se font par rapport à une période différente ("1951-1980" dans notre cas, "1890-1900" dans la cas du GIEC), les tendances sont globalement <b><u>similaires</b></u>. Nous collons aux courbes GIEC les plus pessimistes (SSP3-7.0 et SSP5-8.5).
    <br>
    </div>
    <span>
    """
    st.markdown(text52, unsafe_allow_html=True)
    col1, col2 = st.columns([1,1]) 
    with col1:
        text53 = """
        <span style="font-size:14px">
        <div style="text-indent: 60px;">
        <br><div style="text-indent: 60px;"><b>Prédictions du GIEC - Evolution température °C vs. période préindustrielle (1890-1900)
        
        <span>
        """
        st.markdown(text53, unsafe_allow_html=True)
        # Construire le chemin relatif vers l'image
        chemin_image = os.path.join(os.getcwd(), "Prédictions du GIEC de l'évolution de la température par rapport à la période préindustrielle (1890-1900).jpg")

        # Afficher l'image avec Streamlit
        st.image(
            chemin_image,
            caption="",
            use_column_width=True
        )
    with col2:
        # Rappeler le graphe sauvegardé à l'étape précédente.
        # Vérifiez si le graphique existe dans `st.session_state`
        if "fig_y_Arima_graph" in st.session_state:
            fig_y_Arima = st.session_state["fig_y_Arima_graph"]
            st.plotly_chart(fig_y_Arima, use_container_width=False)
        else:
            st.write("Aucun graphique trouvé dans la session !")
    
    col1, col2 = st.columns([1, 6])  # La première colonne est plus large pour le texte
    with col1:
        st.markdown("## Conclusion")
    with col2:
        box_ticked = st.checkbox("conclusion", label_visibility="hidden")
            # Action conditionnelle si la checkbox est cochée
    if box_ticked:
        text55 = """
        <div style="color: black; font-size:20px; text-align:justify; padding-left:30px; text-indent:0px;">
        1. Les <b>3 modèles de Machine Learning</b> entrainés ont donné de <b>(très) bonnes performances</b>, même en réduisant le nombre de variables.<br><br>
        2. Les <b>prédictions ARIMA</b> sur les <span style="color: green;"><b>variables explicatives</b></span> sont fort <b>satisfaisantes</b> et <b>pertinentes</b> eu égard à d'autres sources.<br><br>
        3. Toutefois, les prédictions de la <span style="color: red;"><b>variable CIBLE</b></span> (écart de température) n'ont pu être proprement réalisées avec le modèle ML retenu. Les données prédites des variables explicatives étant hors du cadre d'entrainement.<br><br>
        <div style="color: black; font-size:18px; text-align:justify; padding-left:30px; text-indent:0px;">
        -> Nos prédictions ne sont basées que sur des données historiques. Elles sont donc exemptes de facteurs extérieurs tels que les actions en cours et à venir pour mitiger la croissance des variables explicatives et donc impacter l'évolution de la variable cible.<br><br>
        -> L'idée de tester notre modèle de régression linéaire à l'envers, à savoir partir des données prédites d'écarts de température pour prédire les variables explicatives, n'a pu se faire, faute de données cibles disponibles et de formule mathétmatique inversée.<br><br>
        </div>
        4. Si les prédictions <b>ARIMA</b> de la <span style="color: red;"><b>variable CIBLE</b></span> montrent une tendance cohérence à une source externe, elles n'en restent pas moins lissées. La <span style="color: red;">perte de précision</span> par rapport au modèle de ML est très conséquente et dommageable. Il doit y avoir d'autres modèles pour ce type de prédictions sur série temporelle à valeurs croissantes, que d'autres exploreront après nous.


        
        </div>
        """
        st.markdown(text55, unsafe_allow_html=True)
    
         
        col1, col2,col3 = st.columns([0.6, 0.6,1.4])  # La première colonne est plus large pour le texte
        with col2:
            box_ticked_end = st.checkbox("end", label_visibility="hidden")
            if box_ticked_end: 
                st.write("### Questions & Réponses")
                st.image("https://gifdb.com/images/high/question-mark-animation-googly-eyes-8i2uhyawe8shl4pw.gif", caption="")
