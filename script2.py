import streamlit as st
import plotly.express as px
def display_sentiment_distribution(train_data):
    st.title("Analyse des Sentiments des Tweets par Produit")
    st.write("Cette section analyse les sentiments exprimés dans les tweets en fonction des différents types de produits. "
             "Nous utiliserons une représentation graphique pour visualiser la répartition des tweets par sentiments et produits. "
             "Ensuite, nous afficherons les mots les plus fréquents dans les tweets à l'aide d'un nuage de mots, en fonction des catégories sélectionnées. "
             "Enfin, nous proposerons un outil de recherche pour trouver des tweets spécifiques en fonction d'un mot clé, du type de produit et du sentiment, "
             "en mettant l'accent sur les tweets négatifs pour identifier les problèmes potentiels liés aux produits.")
    st.subheader("Répartition des tweets par sentiments et par produit")

    labels_sentiments = {0: 'NA', 1: 'Négatif', 2: 'Positif', 3: 'Neutre'}

    # Conversion de la colonne 'Product_Type' en chaîne de caractères
    train_data['Product_Type'] = train_data['Product_Type'].astype(str)

    # Groupement des tweets par sentiments et produits
    sentiment_counts = train_data.groupby(['Sentiment', 'Product_Type']).size().reset_index(name='Count')

    # Mapping des étiquettes des sentiments
    sentiment_counts['Sentiment'] = sentiment_counts['Sentiment'].map(labels_sentiments)

    # Création du graphique avec Plotly Express
    fig = px.bar(sentiment_counts, x='Product_Type', y='Count', color='Sentiment',
                 labels={'Product_Type': 'Produit', 'Count': 'Nombre de tweets', 'Sentiment': 'Sentiment'})

    # Affichage du graphique dans Streamlit
    st.plotly_chart(fig)


    # Calcul des pourcentages
    pourcentage = train_data['Sentiment'].value_counts(normalize=True) * 100

    # Texte de l'analyse
    analyse_texte = f"""
    L'analyse de la répartition des tweets par sentiments et par produits met en évidence les points suivants :

    - Le type de produit 9 est le plus souvent mentionné dans les tweets, avec une proportion élevée de tweets positifs. Cela suggère que ce type de produit est populaire et apprécié par les utilisateurs.

    - En ce qui concerne les autres types de produits, on observe une plus grande proportion de tweets neutres et négatifs dans l'ensemble des données, à l'exception du type de produit 9. Cela indique que la majorité des discussions sur les produits, à l'exception du type de produit 9, ont tendance à être neutres ou négatives en termes de sentiments exprimés. De plus, la proportion de tweets neutres est plus élevée que celle des tweets négatifs.

    - En termes de répartition des sentiments, les tweets se répartissent comme suit : {pourcentage[1]:.2f}% de tweets négatifs, {pourcentage[3]:.2f}% de tweets neutres et {pourcentage[2]:.2f}% de tweets positifs. Il est important de noter que les données relatives aux sentiments des tweets sont fortement déséquilibrées, avec très peu de tweets exprimant un sentiment négatif. Cela indique une prédominance de réactions positives et neutres dans les discussions liées à ces produits.

    - Comparativement aux autres types de produits, le type de produit 6 se distingue en présentant un nombre plus élevé de tweets négatifs. Cela suggère que ce type de produit est moins apprécié et suscite des réactions négatives de la part des utilisateurs.

    - En outre, il est important de noter que les produits 0, 4 et 1 suscitent très peu d'intérêt, car ils sont rarement mentionnés dans les tweets. Cela suggère que ces types de produits ne sont pas populaires ou ne génèrent pas beaucoup de discussions en ligne par rapport aux autres types de produits.

    En résumé, l'analyse des tweets par sentiments et par produits révèle que le type de produit 9 est le plus mentionné, avec une majorité de tweets positifs. Les autres types de produits ont tendance à générer plus de tweets neutres et négatifs. La répartition des sentiments montre une faible proportion de tweets négatifs dans l'ensemble des données. Les produits 0, 4 et 1 suscitent peu d'intérêt et sont rarement mentionnés dans les tweets.
    """

    # Affichage du texte dans Streamlit
    st.write(analyse_texte)
