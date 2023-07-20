import streamlit as st
import plotly.express as px
import pandas as pd
def load_data_and_description(train_data,test_data):

    # Entête de l'application
    st.title("Analyse de sentiments de tweets")

    # Informations sur le projet et le créateur
    st.markdown(
    """
    <p style='text-align: center; font-size: 24px; font-weight: bold;'>Projet réalisé par <a href='https://www.linkedin.com/in/youssoupha-marega/' style='text-decoration: none; color: #0366d6;'>Youssoupha Marega</a></p>
    <p style='text-align: center; font-size: 18px;'>Contact : <a href='mailto:youssouphamarega@gmail.com' style='text-decoration: none; color: #0366d6;'>youssouphamarega@gmail.com</a></p>
    """,
    unsafe_allow_html=True
    )
    st.markdown("L'analyse des sentiments des tweets est devenue un domaine d'étude essentiel pour comprendre les opinions et les réactions des utilisateurs sur les médias sociaux. Dans le cadre de ce projet, grâce à une analyse approfondie, j'ai découvert que les tweets que nous allons étudier sont directement liés au festival SXSW qui s'est tenu à Austin en 2011. Cette information a permis de donner un contexte précis à notre étude. Les données utilisées dans cette analyse proviennent d'un ensemble de données disponible sur Kaggle, comprenant deux fichiers : Train.csv pour l'entraînement et Test.csv pour les tests.")
    
    st.markdown("L'ensemble de données que nous avons choisi d'analyser offre une perspective unique sur les sentiments et les opinions exprimés par les participants à ce festival. Les tweets recueillis lors de cet événement sont un reflet direct des réactions des utilisateurs et nous permettront de mieux comprendre comment les participants ont réagi à différents aspects du festival, notamment aux produits qui y ont été présentés.")
    
    st.markdown("Une analyse minutieuse de ces tweets a révélé qu'une proportion importante d'entre eux comportaient des moqueries et des critiques négatives concernant le lancement du nouvel iPad 2. Ces critiques étaient principalement axées sur le design de l'iPad, suscitant des réactions mitigées parmi les utilisateurs. Cette découverte souligne l'importance de notre projet, qui vise à analyser et prédire les sentiments associés aux tweets afin d'obtenir des informations précieuses sur la perception des utilisateurs à l'égard de nouveaux produits.")
    
    st.markdown("Pour atteindre cet objectif, nous avons développé un modèle d'intelligence artificielle basé sur des techniques avancées de traitement du langage naturel (NLP). Notre modèle a été entraîné sur l'ensemble de données d'entraînement et a démontré une capacité accrue à détecter rapidement les tweets négatifs par rapport à une simple lecture manuelle. Cela est particulièrement pertinent étant donné que les tweets négatifs sont relativement rares dans l'ensemble de données d'entraînement.")
    
    st.markdown("En développant ce modèle d'intelligence artificielle, nous espérons pouvoir prédire avec précision les sentiments associés aux tweets, ce qui pourrait avoir un impact significatif dans des domaines tels que le marketing, la veille des médias sociaux et la compréhension des opinions des utilisateurs.")
    
    st.markdown("Dans la suite de ce projet, nous présenterons en détail l'analyse des sentiments des tweets par produit en utilisant des outils de visualisation. Ensuite, nous aborderons le développement et le déploiement du modèle d'intelligence artificielle à travers une application dédiée.")

    # Définition des descriptions des variables
    variable_descriptions = {
        'Text_ID': 'Identifiant unique',
        'Product_Description': 'Tweet sur un type de produit par un utilisateur',
        'Product_Type': 'Type de produit associé au tweet (10 produits uniques numérotés de 0 à 9)',
        'Sentiment': 'Sentiment associé au tweet (0 - Ne peut pas dire, 1 - Négatif, 2 - Positif, 3 - Pas de sentiment)'
    }

    # Création du DataFrame avec les noms des colonnes et les descriptions
    column_data = []
    for column in train_data.columns:
        description = variable_descriptions.get(column, 'Description non disponible')
        column_data.append({'Nom': column, 'Description': description})
    df_columns = pd.DataFrame(column_data)

    # Afficher le DataFrame dans Streamlit
    st.markdown("**Nom et description des variables:**")
    st.dataframe(df_columns)

    # Afficher les 10 premières observations des données d'entraînement
    if st.checkbox("Afficher les 10 premières observations des données d'entraînement"):
        train_data_sample = train_data.head(10)
        #st.markdown("**Les 10 premières observations de l'échantillon d'entraînement:**")
        st.dataframe(train_data_sample)
    n_train = train_data.shape[0]
    st.write("Les données d'entraînement contiennent un total de", n_train, "observations.")

    # Afficher les 10 premières observations des données de test
    if st.checkbox("Afficher les 10 premières observations des données de test"):
        test_data_sample = test_data.head(10)
        #st.markdown("**Les 10 premières observations de l'échantillon de test:**")
        st.dataframe(test_data_sample)
    n_test = test_data.shape[0]
    st.write("Les données de test contiennent un total de", n_test, "observations.") 

    st.write("La variable cible est Sentiment et elle n'est pas présente dans les données de test. Pour suite, la variable Sentiment est codée comme suit : 0 correspond à NA (Ne peut pas dire), 1 correspond à Négatif, 2 correspond à Positif, et 3 correspond à Neutre (Pas de sentiment).")


    # Lien vers la base de données Kaggle
    st.markdown("Source de données : [Kaggle](https://www.kaggle.com/datasets/tanyadayanand/analyzing-sentiments-related-to-various-products)")





