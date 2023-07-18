import streamlit as st
import pandas as pd
import base64

def Bienvenue_dansApp():
    st.subheader("Déploiement du modèle")
    st.subheader("Bienvenue dans l'application de prédiction de sentiments de tweets !")
    
    # Étape 1 : Télécharger le fichier de données
    st.markdown("**Étape 1 : Télécharger le fichier de données**")
    st.markdown("Vous avez deux options pour télécharger le fichier Test.csv :")
    st.markdown("1. Cliquez sur le lien Kaggle ci-dessous pour télécharger le fichier :")
    st.markdown("[Télécharger Test.csv depuis Kaggle](https://www.kaggle.com/datasets/tanyadayanand/analyzing-sentiments-related-to-various-products)")
    st.markdown("2. Appuyez sur le bouton ci-dessous pour télécharger le fichier manuellement :")
    
    # Bouton pour télécharger le fichier Test.csv
    if st.button("Télécharger Test.csv"):
        csv_data = pd.read_csv('Test.csv')
        csv_data.to_csv('Test.csv', index=False)  # Save DataFrame to a CSV file for download
        b64 = base64.b64encode(open('Test.csv', 'rb').read()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="Test.csv">Cliquez ici pour télécharger Test.csv</a>'
        st.markdown(href, unsafe_allow_html=True)
    
    # Étape 2 : Charger le fichier de données
    st.markdown("**Étape 2 : Charger le fichier de données**")
    st.markdown("Maintenant que le fichier \"Test.csv\" est disponible, nous allons le charger dans l'application. Pour ce faire, appuyez sur le bouton \"Browse files\" ci-dessous pour charger automatiquement le fichier :")
    
    # Étape 3 - Partie 1 : Afficher les prédictions et la répartition des sentiments
    st.markdown("**Étape 3 - Partie 1 : Afficher les prédictions et la répartition des sentiments prédits et produits**")
    st.write("""Une fois que le fichier a été chargé avec succès, notre modèle d'intelligence artificielle analysera les tweets et prédira les sentiments associés à chacun d'entre eux. Les tweets seront classés en trois catégories : négatifs, neutres ou positifs.""")        
    st.write("""Ensuite, nous afficherons les prédictions des tweets avec les informations suivantes : "Description du produit", "Sentiment prédit", "Type de produit" et "Probabilité de prédiction". Une figure illustrant la répartition des tweets par sentiment prédit et par type de produit sera également affichée.""")
    # Étape 3 - Partie 2 : Afficher les tweets prédits négatifs, neutres et positifs
    st.markdown("**Étape 3 - Partie 2 : Afficher les tweets prédits négatifs, neutres et positifs**")
    st.markdown("Dans cette partie, nous afficherons les tweets prédits négatifs, neutres et positifs, triés par probabilités de prédiction. Cette approche nous permettra de détecter plus rapidement les tweets prédits comme étant négatifs.")
    st.markdown("Si vous êtes intéressé par les tweets prédits négatifs, appuyez sur le bouton \"Afficher les tweets prédits négatifs\" ci-dessous. Vous verrez alors les tweets classés par ordre de probabilité, avec les tweets négatifs les plus probables en premier. De même, vous pourrez afficher les tweets prédits neutres ou positifs en appuyant sur les boutons correspondants.")
    st.markdown("N'hésitez pas à explorer et à interagir avec les résultats obtenus. Vous pouvez consulter les tweets classés par sentiment et voir les prédictions les plus probables en premier.")
    
    # Contact
    st.markdown("**Contact**")
    st.markdown("""Si vous avez des questions concernant le fonctionnement de l'application ou les résultats, n'hésitez pas à nous contacter par courriel à youssouphamarega@gmail.com. Nous serons ravis de vous aider !""")
    st.markdown("Merci d'utiliser notre application de prédiction de sentiments sur Twitter. Nous espérons que vous trouverez cette expérience intéressante et utile !")
