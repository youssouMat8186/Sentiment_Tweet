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

