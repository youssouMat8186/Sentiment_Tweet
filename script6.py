import pandas as pd
import streamlit as st
import joblib
import plotly.express as px
from nltk.stem import PorterStemmer
import base64
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def tokenize(text):
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in text.split()]

# Fonction pour charger le modèle
def load_model():
    modele_chemin = 'modele_svm.pkl'
    loaded_model = joblib.load(modele_chemin)
    return loaded_model


def deploiement():
    st.subheader("Déploiement du modèle")

    # Fonction pour prétraiter les données et effectuer les prédictions
    def make_predictions(uploaded_file):
        df_test = pd.read_csv(uploaded_file, delimiter=',')
        X_test = df_test['Product_Description']
        Product_Type = df_test['Product_Type']

        # Charger le modèle
        loaded_model = load_model()

        # Prétraiter les données
        X_test_processed = [tokenize(text) for text in X_test]
        X_test_processed_str = [' '.join(tokens) for tokens in X_test_processed]

        # Faire les prédictions
        y_pred = loaded_model.predict(X_test_processed_str)
        y_pred_proba = loaded_model.predict_proba(X_test_processed_str)

        # Arrondir les probabilités à 4 décimales
        y_pred_proba = y_pred_proba.round(4)

        # Créer un DataFrame avec les prédictions et les probabilités
        Resultats_prediction = pd.DataFrame({
            'Product_Description': X_test,
            'Product_Type': Product_Type,
            'Sentiment_Predicted': [labels_sentiments.get(sentiment, "N/A") for sentiment in y_pred],
            'Probability': y_pred_proba.max(axis=1)  # La probabilité maximale de la classe prédite
        })

        # Mapper les étiquettes des sentiments
        Resultats_prediction['Sentiment_Predicted'] = Resultats_prediction['Sentiment_Predicted'].map(labels_sentiments)

        return Resultats_prediction

    # Fonction pour afficher la répartition des sentiments
    def display_sentiment_distribution(data):
        st.markdown("**Répartition des tweets par sentiment prédit et par type de produit**")
        # Conversion de la colonne 'Product_Type' en chaîne de caractères
        data['Product_Type'] = data['Product_Type'].astype(str)

        # Groupement des tweets par sentiments et produits
        sentiment_counts = data.groupby(['Sentiment_Predicted', 'Product_Type']).size().reset_index(name='Count')

        # Création du graphique avec Plotly Express
        fig = px.bar(sentiment_counts, x='Product_Type', y='Count', color='Sentiment_Predicted',
                     labels={'Product_Type': 'Produit', 'Count': 'Nombre de tweets', 'Sentiment_Predicted': 'Sentiment Prédit'})

        # Affichage du graphique dans Streamlit
        st.plotly_chart(fig)

    # Définition de labels_sentiments
    labels_sentiments = {'Negatif': 'Negatif', 'Positif': 'Positif', 'Neutre': 'Neutre'}

    # Demander à l'utilisateur de télécharger le fichier CSV
    uploaded_file = st.file_uploader("Uploader le fichier Test.csv", type="csv")

    if uploaded_file is not None:
        # Faire les prédictions et obtenir le DataFrame des résultats
        Resultats_prediction = make_predictions(uploaded_file)

        # Afficher les prédictions dans une application Streamlit
        st.dataframe(Resultats_prediction[['Product_Description', 'Sentiment_Predicted', 'Product_Type', 'Probability']])

        # Bouton pour télécharger les données de prédiction au format CSV
        csv_data = Resultats_prediction.to_csv(index=False)
        b64 = base64.b64encode(csv_data.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="resultats_prediction.csv">Télécharger les données de prédiction en CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

        # Afficher la répartition des sentiments
        display_sentiment_distribution(Resultats_prediction)

        # Boutons pour afficher les tweets par sentiment prédit
        if st.button("Afficher les tweets prédits négatifs"):
            negative_tweets = Resultats_prediction[Resultats_prediction['Sentiment_Predicted'] == 'Negatif']
            negative_tweets = negative_tweets.sort_values(by='Probability', ascending=False)
            st.dataframe(negative_tweets)

            # Bouton pour télécharger les tweets prédits négatifs au format CSV
            csv_data = negative_tweets.to_csv(index=False)
            b64 = base64.b64encode(csv_data.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="tweets_negatifs.csv">Télécharger les tweets prédits négatifs en CSV</a>'
            st.markdown(href, unsafe_allow_html=True)

        if st.button("Afficher les tweets prédits neutres"):
            neutral_tweets = Resultats_prediction[Resultats_prediction['Sentiment_Predicted'] == 'Neutre']
            neutral_tweets = neutral_tweets.sort_values(by='Probability', ascending=False)
            st.dataframe(neutral_tweets)

            # Bouton pour télécharger les tweets prédits neutres au format CSV
            csv_data = neutral_tweets.to_csv(index=False)
            b64 = base64.b64encode(csv_data.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="tweets_neutres.csv">Télécharger les tweets prédits neutres en CSV</a>'
            st.markdown(href, unsafe_allow_html=True)

        if st.button("Afficher les tweets prédits positifs"):
            positive_tweets = Resultats_prediction[Resultats_prediction['Sentiment_Predicted'] == 'Positif']
            positive_tweets = positive_tweets.sort_values(by='Probability', ascending=False)
            st.dataframe(positive_tweets)

            # Bouton pour télécharger les tweets prédits positifs au format CSV
            csv_data = positive_tweets.to_csv(index=False)
            b64 = base64.b64encode(csv_data.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="tweets_positifs.csv">Télécharger les tweets prédits positifs en CSV</a>'
            st.markdown(href, unsafe_allow_html=True)


