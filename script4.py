import pandas as pd
import streamlit as st

def find_tweets_with_text(text, sentiment, type_produit, train_data):
    # Convertir le texte recherché en minuscules pour une recherche insensible à la casse
    text = text.lower()

    # Filtrer les tweets contenant le texte spécifié (insensible à la casse)
    matching_tweets = train_data[train_data['Product_Description'].str.lower().str.contains(text)]

    # Filtrer les tweets en fonction du sentiment
    if sentiment != -1:
        matching_tweets = matching_tweets[matching_tweets['Sentiment'] == sentiment]

    # Filtrer les tweets en fonction du type de produit
    if type_produit != "Tout":
        matching_tweets = matching_tweets[matching_tweets['Product_Type'] == type_produit]

    # Renvoyer les tweets correspondants
    return matching_tweets['Product_Description'].tolist()

def Tweet_Mots_Cles(train_data):
    # Titre de l'application
    st.subheader("Recherche de tweets par mot clé")

    # Sélection du type de produit
    types_produits = list(train_data['Product_Type'].unique())
    types_produits.insert(0, "Tout")
    type_produit = st.selectbox("Sélectionnez le type de produit", types_produits)

    # Sélection du sentiment
    sentiments = {0: 'NA', 1: 'Négatif', 2: 'Positif', 3: 'Neutre'}
    sentiments[-1] = "Tout"
    sentiment = st.selectbox("Sélectionnez le sentiment", list(sentiments.keys()), format_func=lambda x: sentiments[x])

    # Champ de saisie du texte
    search_text = st.text_input("Entrez un bout de texte à rechercher dans les tweets")

    # Nombre de tweets à afficher
    nombre_tweets_affichage = st.slider("Nombre de tweets à afficher", min_value=1, max_value=100, value=10, step=1)

    # Bouton de recherche
    if st.button("Rechercher"):
        # Vérifier si un texte de recherche a été saisi
        if search_text:
            # Appeler la fonction find_tweets_with_text pour trouver les tweets correspondants
            matching_tweets = find_tweets_with_text(search_text, sentiment, type_produit, train_data)

            # Filtrer le nombre de tweets à afficher
            tweets_affichage = matching_tweets[:nombre_tweets_affichage]

            # Afficher les tweets correspondants
            if tweets_affichage:
                st.success(f"Tweets correspondants (affichage des {nombre_tweets_affichage} premiers tweets) :")
                for tweet in tweets_affichage:
                    st.write(tweet)
            else:
                st.warning("Aucun tweet correspondant trouvé.")
        else:
            st.warning("Veuillez saisir un texte à rechercher.")
            
    # Résumé de l'analyse
    st.markdown("""
    Cet outil nous permet de rechercher des tweets dans l'ensemble des données d'entraînement en fonction d'un mot présent dans le tweet, du type de produit et du sentiment. Dans cette section, nous nous concentrons sur les tweets négatifs.

    En analysant la répartition des tweets par sentiments et par produits, nous avons constaté que la majorité des tweets négatifs sont liés au type de produit 6. En examinant de plus près le nuage de mots, nous avons remarqué que les termes "Design Headaches", "iPad Design" et "tapworthy" sont fréquemment mentionnés dans les tweets négatifs associés au type de produit 6, mais ils sont rares dans l'ensemble des données.

    Nous avons donc utilisé notre outil de recherche de mots-clés pour analyser les tweets négatifs contenant ces termes. Cette analyse a révélé une tendance générale de moqueries et de critiques envers le design de l'iPad 2 lors du festival SXSW.

    Les utilisateurs expriment leur mécontentement envers le design de l'iPad, en mentionnant des problèmes de conception, des difficultés d'utilisation et des critiques sur l'esthétique de l'appareil.

    Ces résultats sont basés sur l'analyse des tweets et des opinions exprimées par les utilisateurs. Il est important de noter que les problèmes spécifiques liés au design de l'iPad 2 peuvent varier d'un utilisateur à l'autre, mais des problèmes d'ergonomie, de fragilité, de poids, d'esthétique et d'innovation ont été fréquemment mentionnés.
                
    Ces observations nous aident à mieux comprendre les réactions des utilisateurs face au design de l'iPad 2 et peuvent être utiles pour identifier les aspects à améliorer dans les futurs développements de produits.
    
    En résumé, notre démarche pour l'analyse des sentiments des tweets est la suivante : nous avons examiné les différents types de produits afin de déterminer ceux qui suscitent le plus de tweets négatifs. À l'aide d'un nuage de mots, nous avons identifié les mots les plus fréquents dans les tweets négatifs pour ces types de produits. Ensuite, nous avons analysé spécifiquement les tweets négatifs contenant ces mots clés importants.
    """)

