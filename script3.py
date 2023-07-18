import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt



def generate_wordcloud(train_data):
    st.subheader("Nuage de mots des tweets")
    st.write("Visualisez les termes les plus fréquents dans les tweets")

    # Sélection du type de produit
    types_produits = list(train_data['Product_Type'].unique())
    types_produits.insert(0, "Tous les produits")
    type_produit = st.selectbox("Sélectionnez le type de produit", types_produits)

    # Sélection du sentiment
    sentiments = {0: 'NA', 1: 'Négatif', 2: 'Positif', 3: 'Neutre'}
    sentiments[-1] = "Tous les sentiments"
    sentiment = st.selectbox("Sélectionnez le sentiment", list(sentiments.keys()), format_func=lambda x: sentiments[x])

    # Filtrage des tweets
    if type_produit == "Tous les produits" and sentiment == -1:
        tweets_filtrés = train_data
    elif type_produit == "Tous les produits":
        tweets_filtrés = train_data[train_data['Sentiment'] == sentiment]
    elif sentiment == -1:
        tweets_filtrés = train_data[train_data['Product_Type'] == type_produit]
    else:
        tweets_filtrés = train_data[(train_data['Product_Type'] == type_produit) & (train_data['Sentiment'] == sentiment)]

    texte_tweets = " ".join(tweets_filtrés['Product_Description'])

    # Vérification s'il y a des tweets
    if len(texte_tweets) == 0:
        st.warning("Aucun tweet disponible pour la combinaison sélectionnée.")
    else:
        # Nombre de mots à afficher
        nombre_mots_affichage = st.slider("Nombre de mots à afficher", key="slider_mots_affichage", min_value=1, max_value=100, value=10)

        # Génération du nuage de mots avec la fréquence des mots
        wordcloud = WordCloud(width=800, height=400, max_words=nombre_mots_affichage, background_color='white', relative_scaling=0.5).generate(texte_tweets)

        # Affichage du nuage de mots
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout()
        st.pyplot(plt)

    text = """
    Le nuage de mots est une représentation visuelle des mots les plus fréquents dans les tweets, catégorisés par produits et sentiments. Les mots apparaissent plus grands dans le nuage s'ils sont plus fréquents dans les tweets correspondant aux catégories sélectionnées.

    En analysant le nuage de mots, nous avons identifié les mots les plus importants en choisissant tous les produits, tous les sentiments et en affichant les 12 mots les plus fréquents. Parmi ces mots, "SXSW" et "Austin" se sont démarqués en raison de leur fréquence plus élevée par rapport aux autres mots. Ces mots sont moins couramment utilisés dans la vie quotidienne, mais ils ont une importance particulière dans les tweets analysés.

    Ces observations nous ont permis de découvrir que les tweets analysés sont étroitement liés au festival SXSW, qui se déroule à Austin. Le terme "SXSW" est l'acronyme du festival South by Southwest, un événement majeur de l'industrie du divertissement et de la technologie. La présence fréquente de "SXSW" dans les tweets, ainsi que du mot "Austin", suggère que les utilisateurs partagent leurs expériences, leurs opinions et leurs réactions concernant ce festival.

    Cette découverte est le fruit de l'analyse des mots les plus fréquents dans les tweets, telle que visualisée dans le nuage de mots. En comprenant ces mots clés et en évaluant les sentiments exprimés dans les tweets, nous pouvons mieux appréhender l'opinion des participants, extraire des informations précieuses sur leurs expériences et leurs réactions, et fournir des informations exploitables aux organisateurs du festival et aux autres parties intéressées.

    En résumé, l'analyse du nuage de mots nous a permis d'identifier les mots les plus importants dans les tweets, notamment "SXSW" et "Austin", qui sont liés au festival SXSW. Ces mots clés sont essentiels pour mieux comprendre les expériences des participants et améliorer continuellement l'événement. Le nuage de mots a été une ressource précieuse pour cette découverte, en mettant en évidence les mots les plus fréquents et en fournissant des insights sur le contenu des tweets analysés.
    """
    st.write(text)