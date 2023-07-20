import streamlit as st
import pandas as pd
from sklearn.metrics import recall_score




def Introduction():
    # Affichage du texte dans Streamlit
    st.header("Développement du modèle de prédiction de sentiment de tweets")
    st.write("Pour développer notre modèle de prédiction de sentiment  tweets, nous utilisons l'ensemble de données d'entraînement Train.csv provenant de Kaggle. Dans ce modèle, la variable cible 'Sentiment' est codée avec les valeurs suivantes : 0 pour 'NA' (Non attribué), 1 pour 'Négatif', 2 pour 'Positif' et 3 pour 'Neutre'. Lorsque la variable cible est égale à 'NA', cela signifie que le sentiment associé à un tweet ne peut pas être déterminé, et nous considérons alors cette variable comme manquante pour cette observation.")

    st.write("Afin de garantir la qualité de nos données d'entraînement, nous excluons les observations dont la variable cible est manquante de notre ensemble d'entraînement. Cela nous permet de nous assurer que seules les observations avec des sentiments attribués sont utilisées pour la formation de notre modèle.")

    st.code("""
    df = pd.read_csv('Train.csv', delimiter=',')
    mapping = {0: 'NA', 1: 'Négatif', 2: 'Positif', 3: 'Neutre'}
    df['Sentiment'] = df['Sentiment'].replace(mapping)
    df = df.drop(df[df['Sentiment'] == 'NA'].index)
    """, language="python")

    st.write("Étant donné le déséquilibre des classes et la faible représentation des tweets négatifs dans notre ensemble de données, il est crucial de prendre en compte cette spécificité lors du choix des modèles et des hyperparamètres. Pour évaluer les performances des modèles, nous utilisons la sensibilité de la classe négative de la variable 'Sentiment'. Cette métrique mesure la capacité du modèle à détecter correctement les tweets négatifs.")

    st.write("En utilisant la sensibilité de la classe négative comme métrique pour le choix des modèles et des hyperparamètres, nous nous assurons que notre modèle est capable de détecter efficacement les tweets négatifs, malgré leur faible fréquence. Nous accordons ainsi une importance particulière à la capacité du modèle à traiter cette classe spécifique et à minimiser les faux négatifs, c'est-à-dire les cas où le modèle ne parvient pas à détecter les tweets négatifs. Nous avons défini une fonction pour calculer la sensibilité de la classe négative. Cette fonction, appelée 'neg_class_recall', est utilisée pour évaluer la performance du modèle en termes de détection des tweets négatifs.")

    st.code("""
    # Définition de la fonction de score pour la sensibilité de la classe négative
    def neg_class_recall(y_true, y_pred):
    recall = recall_score(y_true, y_pred, labels=['Négatif'], average=None)
    return recall[0]
    """, language="python")

    st.write("Le processus de développement de notre modèle de prédiction de tweets comprend deux parties distinctes. La première partie consiste à choisir le modèle d'apprentissage automatique le plus approprié en évaluant plusieurs modèles sur l'ensemble de test. La seconde partie concerne la recherche des meilleurs paramètres pour le modèle choisi, en utilisant la recherche par grille.")

def Choix_Modele():
    st.subheader("Choix du modèle d'apprentissage automatique")
    st.write("Pour sélectionner le modèle d'apprentissage automatique le plus approprié, nous avons extrait les variables indépendantes et dépendantes de notre ensemble de données. La variable indépendante 'Avis' correspond aux tweets, tandis que la variable dépendante 'Sentiment' représente les sentiments associés aux tweets.")

    st.code("""
    Avis = df['Product_Description']
    Sentiment = df['Sentiment']
    """)

    st.write("Ensuite, nous avons effectué un prétraitement des données en utilisant la méthode CountVectorizer. CountVectorizer est une méthode qui convertit le texte en une représentation numérique basée sur le compte des mots. Cela permet de transformer le texte brut en une représentation vectorielle utilisable par les algorithmes d'apprentissage automatique.")

    st.code("""
    vectorizer = CountVectorizer(stop_words='english')
    """)

    st.write("Nous avons également spécifié l'argument 'stop_words' à 'english' dans CountVectorizer afin d'exclure les mots très courants tels que les articles, les prépositions, les conjonctions, etc. Ces mots ne sont pas utiles pour les algorithmes d'apprentissage automatique et n'augmentent que la dimensionnalité des variables sans apporter de réelle valeur informative.")

    st.write("Nous avons ensuite divisé nos données en un ensemble d'entraînement et un ensemble de test, en allouant 20% des données à l'ensemble de test.")

    st.code("""
    X_train, X_test, y_train, y_test = train_test_split(Avis, Sentiment, test_size=0.2, random_state=0)
    """)

    st.write("Nous avons défini plusieurs modèles d'apprentissage automatique à évaluer, notamment : forêt aléatoire (Random Forest), réseaux de neurones à perceptron multicouches (MLP), classificateur à vecteur de support machine (SVC), naïve Bayes multinomial (MultinomialNB) et régression logistique (Logistic Regression). Chaque modèle a été entraîné sur l'ensemble d'entraînement et évalué sur l'ensemble de test.")

    st.code("""
    models = [
        ('Random Forest', RandomForestClassifier(max_depth=200, random_state=123)),
        ('MLP', MLPClassifier(hidden_layer_sizes=(64, 64), activation='relu', random_state=123)),
        ('SVC', SVC(kernel='linear', C=1.0, random_state=123)),
        ('MultinomialNB', MultinomialNB()),
        ('Logistic Regression', LogisticRegression(random_state=123)),
        ('SVC_linear', SVC(kernel='linear', C=1.0, random_state=123))
    ]
    """)

    st.write("Nous avons utilisé la métrique de sensibilité de la classe négative, calculée à l'aide de la fonction 'neg_class_recall' que nous avons définie précédemment, pour évaluer les performances des modèles. Le modèle avec la plus grande sensibilité de la classe négative sera choisi.")

    # Création du tableau des résultats
    results_df = pd.DataFrame({
        'Modèle': ['Random Forest', 'MLP', 'SVC', 'MultinomialNB', 'Logistic Regression'],
        'Sensibilité négative': [0.174, 0.220, 0.279, 0.139, 0.209],
        'Exactitude': [0.662, 0.642, 0.682, 0.664, 0.676]
    })

    # Définition de la colonne 'Modèle' comme index
    results_df = results_df.transpose()

    # Affichage du tableau des résultats
    st.markdown("**Résultats de performance des modèles sur l'ensemble de test**")
    st.table(results_df)

    # Affichage du meilleur modèle
    st.write("Le meilleur modèle est SVC. Il a obtenu une exactitude de 0.682 et un score de sensibilité de la classe négative de 0.279.")

    st.write("""
    Après avoir exploré différentes méthodes de vectorisation de texte, telles que TF-IDF et les Word Embeddings, ainsi que des techniques de réduction de dimensionnalité comme PCA, nous avons également utilisé des techniques d'échantillonnage de la variable cible, telles que le suréchantillonnage et le sous-échantillonnage, pour remédier au problème de déséquilibre de la variable Sentiment. Cependant, nous avons constaté que malgré ces efforts, ces approches n'ont pas conduit à une amélioration significative du score de sensibilité de la classe négative. Les résultats obtenus pour cette métrique spécifique n'ont pas été satisfaisants.
             """)
    st.write("""
    Par conséquent, nous avons décidé de nous concentrer principalement sur la méthode CountVectorizer en combinaison avec le modèle SVC (Support Vector Classifier).
    """)

def Optimisation_Hyper_Params():
    st.subheader("Optimisation des hyperparamètres du modèle SVC")
    st.write("""
             L'optimisation des hyperparamètres du modèle SVC (Support Vector Classifier) est une étape essentielle dans le processus de développement de notre modèle de prédiction de tweets. Les hyperparamètres sont des paramètres du modèle qui ne sont pas appris à partir des données, mais qui doivent être définis avant l'entraînement du modèle. L'objectif de cette étape est de trouver les valeurs optimales des hyperparamètres afin d'améliorer les performances du modèle.
             """)
    
    st.markdown("**Fonction de tokenization avec le stemmer PorterStemmer**")
    st.write("""Tout d'abord, nous définissons une fonction de tokenization qui utilise le stemmer PorterStemmer pour mettre les mots en minuscules et les réduire à leur forme racine. Nous avons choisi cette approche afin de normaliser le texte, de réduire la dimensionnalité des données et d'optimiser la rapidité d'exécution. Cependant, il est important de noter que le stemmer PorterStemmer ne tient pas compte du contexte spécifique des tweets et nécessite d'autres techniques pour traiter les éléments spécifiques aux réseaux sociaux, ce qui peut potentiellement affecter la performance de notre modèle d'analyse de sentiment.""")
    
    st.markdown("**Regroupement distinct des types de produits basé sur la répartition des sentiments**")
    st.write("""Pendant notre analyse de la répartition des tweets par sentiments et par produits, nous avons observé que les types de produits 0, 1, 2, 3, 4, 5, 6, 7 et 8 généraient principalement des réactions neutres et négatives, tandis que le type de produit 9 était plus souvent mentionné dans les tweets et suscitait principalement des réactions positives. Pour intégrer ces observations dans notre modèle, nous avons pris la décision de regrouper les types de produits en deux groupes distincts.
    
    Dans le premier regroupement, nous avons divisé les types de produits en deux groupes distincts : le premier groupe comprend les types de produits 0, 1, 2, 3, 4, 5, 6, 7 et 8, tandis que le deuxième groupe ne comprend que le type de produit 9. Cette approche a été choisie en raison des différences significatives dans les réactions et les sentiments exprimés par les utilisateurs pour ces deux groupes distincts. Cependant, pour le deuxième regroupement, nous avons considéré les neuf types de produits ensemble, sans les diviser en groupes distincts.
    
    Nous avons utilisé le test de chi-deux pour analyser les différences entre les observations observées et attendues pour chaque regroupement. Les résultats ont montré des statistiques du chi-deux élevées et des p-valeurs très proches de zéro pour les deux regroupements, indiquant des divergences significatives entre les observations observées et attendues. Cependant, il est important de noter que le premier regroupement présente deux degrés de liberté, tandis que le deuxième regroupement en a dix-huit.
    
    Sur la base de ces résultats, nous pouvons conclure que le premier regroupement, avec ses deux degrés de liberté, est plus approprié. Un nombre plus faible de degrés de liberté indique une plus grande puissance statistique et une meilleure capacité à détecter des différences significatives. Par conséquent, le premier regroupement, avec une statistique du chi-deux plus élevée et un nombre de degrés de liberté plus faible, est préférable dans ce contexte.""")
    
    st.write("""En conséquence, nous avons ajouté la mention "Produit_9" aux tweets correspondants dans notre fonction de tokenization pour tenir compte de cette particularité lors du prétraitement des données.
    Ce prétraitement spécifique, combiné au regroupement distinct des types de produits, a permis d'améliorer légèrement l'exactitude globale du modèle tout en maintenant approximativement la sensibilité de la classe négative.""")
    st.code("""
    from nltk.stem import PorterStemmer

    def tokenize(text):
        stemmer = PorterStemmer()
        if not df[df['Numero_Produit'] == 9].empty:
        text += " Produit_9"
        return [stemmer.stem(word) for word in text.split()]
    """,language="python")
    
    st.markdown("**Création du pipeline de traitement des données**")
    st.write("""Ensuite, nous créons un pipeline qui comprend une étape de tokenization utilisant le CountVectorizer. Nous spécifions également l'argument 'stop_words' à 'english' pour exclure les mots courants de l'anglais. Cela permet d'éliminer le bruit inutile des mots fréquents et de se concentrer sur les mots plus informatifs. Toutefois, il est important de noter que l'exclusion des stop words peut avoir un impact sur la sémantique des tweets.""")

    st.code("""
    pipeline = Pipeline([
    ('stemmer', CountVectorizer(
        stop_words='english',
        tokenizer=tokenize
    )),
    ('classifier', SVC(random_state=123, probability=True))
    ])
    """,language = "python")
    
    st.markdown("**Division des données en ensembles d'entraînement et de test**")

    st.write("""Nous divisons nos données en ensembles d'entraînement et de test en utilisant la fonction train_test_split, en maintenant la proportion des classes avec l'argument 'stratify=Sentiment'. Cette stratégie garantit que nos ensembles d'entraînement et de test contiennent des proportions équilibrées des différentes classes de sentiments. Cela est essentiel pour évaluer les performances du modèle de manière impartiale et éviter tout biais potentiel lié à la répartition des sentiments.""")

    st.code("""
    # Division des données en ensembles de formation et de test
    X_train, X_test, y_train, y_test = train_test_split(Avis, Sentiment, test_size=0.25, 
                                                    stratify=Sentiment, random_state=456)
            """,language="python")
    
    st.markdown("**Définition du dictionnaire de paramètres pour la recherche par grille**")
    st.write("""Nous avons pris soin de définir des plages de valeurs appropriées en tenant compte des caractéristiques de notre ensemble de données et des connaissances existantes. Cependant, il est important de noter que les plages de valeurs que nous avons choisies ne sont pas exhaustives et d'autres valeurs pourraient également être explorées. La sélection des plages de valeurs est souvent basée sur des connaissances préalables, des expériences antérieures ou des recommandations de la littérature.""")
    st.code("""
    # Définition des valeurs à tester pour les paramètres 'C', 'kernel', 'gamma' et 'degree'
    param_grid = {
    'classifier__C': [0.01, 0.1, 1, 10],
    'classifier__kernel': ['linear', 'rbf', 'poly'],
    'classifier__gamma': [0.01, 0.1, 1, 10],
    'classifier__degree': [2, 3, 4]
}
            """,language="python")
    
    st.markdown("**Création de l'objet GridSearchCV pour la recherche par grille**")
    st.write("""Nous créons ensuite un objet GridSearchCV en utilisant la méthode de recherche par grille. Cet objet combine notre pipeline, les paramètres à tester, la fonction de scoring 'neg_class_recall' que nous avons définie précédemment, ainsi que la validation croisée stratifiée avec cinq plis (StratifiedKFold). La stratification dans la validation croisée garantit que chaque pli de l'ensemble d'entraînement contient une distribution similaire des différentes classes de la variable 'Sentiment'. Cela permet d'éviter les biais liés à un déséquilibre de classe et assure une évaluation plus juste des performances du modèle. Enfin, nous utilisons la méthode fit pour entraîner la recherche par grille sur l'ensemble d'entraînement.""")
    st.code("""
    # Création de l'estimateur de la recherche par grille avec la métrique de scoring personnalisée
    scoring = make_scorer(neg_class_recall)
    skf = StratifiedKFold(n_splits=5)

    # Création de l'objet GridSearchCV avec la validation croisée stratifiée
    grid_search = GridSearchCV(pipeline, param_grid, scoring=scoring, cv=skf)

    # Entraînement de la recherche par grille
    grid_search.fit(X_train, y_train)
            """,language="python")
      
    st.write("""Après avoir effectué la recherche par grille, nous avons sélectionné le modèle SVM avec les meilleurs paramètres pour notre tâche de prédiction de tweets. Les paramètres optimaux sont les suivants :
             
    - 'C' (Coefficient de régularisation) : 10
             
    - 'degree' (Degré du noyau polynômial) : 2
             
    - 'gamma' (Coefficient du noyau) : 0.01
             
    - 'kernel' (Type de noyau) : 'linear
             """)
def Text_Resultas():
    # Affichage des résultats dans Streamlit
    st.subheader("Résultats du modèle :")
    # Affichage de la matrice de confusion
    st.image('confusion_matrix.png', caption='Matrice de confusion')
    text_resultas = """
    Notre modèle SVC avec les hyperparamètres suivants : C=10, degré=2, gamma=0.01, noyau='linear', a été choisi comme meilleur modèle principalement en raison de sa sensibilité élevée de la classe négative. Il a obtenu un score de sensibilité de 0.425, ce qui indique sa capacité à détecter efficacement les tweets négatifs malgré leur faible fréquence.

    En examinant la matrice de confusion, nous pouvons observer la répartition des prédictions du modèle pour chaque classe.

    Notre modèle présente une exactitude de 0.65, ce qui indique sa capacité à prédire avec précision les sentiments des tweets. Cependant, nous nous concentrons particulièrement sur le score de rappel (sensibilité) pour la classe 'Négatif'. Avec un score de 0.425, notre modèle est capable de rappeler environ 42,5 % des tweets réellement négatifs de notre ensemble de données.

    Comparé à la lecture de tous les tweets, où seulement 6 % sont négatifs dans l'ensemble des données, notre modèle offre une approche plus performante pour détecter les tweets négatifs. Il est capable de repérer un pourcentage significatif de tweets négatifs parmi ceux qui sont réellement négatifs. Cela est particulièrement pertinent dans des contextes où l'identification rapide des sentiments négatifs est nécessaire.

    Ces résultats mettent en évidence l'efficacité de notre modèle d'apprentissage automatique dans la détection des sentiments et soulignent l'importance de continuer à améliorer notre modèle pour une meilleure détection des tweets négatifs, compte tenu de leur rareté dans les données.
    """
    st.write(text_resultas)
    # Affichage du titre
    st.subheader("Analyse des courbes d'apprentissage pour améliorer la détection de tweets négatifs")

    # Affichage de la figure 'courbes_apprentissage.png'
    st.image('courbes_apprentissage.png', use_column_width=True)

    # Affichage du texte
    st.write("Le graphique des courbes d'apprentissage montre comment la performance du modèle (dans ce cas, la capacité à détecter les tweets négatifs) évolue en fonction de la taille de l'ensemble d'entraînement. Voici ce que nous pouvons conclure du graphique :")
    st.write("1. **Courbe d'apprentissage pour l'ensemble d'entraînement (bleu)** :")
    st.write("- Lorsque la taille de l'ensemble d'entraînement est petite (10% des tweets), le modèle a une performance initiale basse pour détecter les tweets négatifs.")
    st.write("- À mesure que la taille de l'ensemble d'entraînement augmente, le score de rappel pour les tweets négatifs augmente également.")
    st.write("- Cela signifie que lorsque le modèle dispose de plus de tweets dans l'ensemble d'entraînement, il apprend mieux à détecter les tweets negatifs.")
    st.write("2. **Courbe d'apprentissage pour l'ensemble de test (orange)** :")
    st.write("- La courbe pour l'ensemble de test suit une tendance similaire à celle de l'ensemble d'entraînement, ce qui est généralement une bonne indication.")
    st.write("- Cependant, le score de rappel pour les tweets négatifs sur l'ensemble de test est légèrement inférieur à celui sur l'ensemble d'entraînement. C'est normal car le modèle généralise moin bien sur des tweets qu'il n'a pas encore vus.")
    st.write("Recommandation :")
    st.write("Le graphique indique que l'ajout de tweets dans l'ensemble d'entraînement pourrait être bénéfique pour améliorer la détection des tweets négatifs. En augmentant la taille de l'ensemble d'entraînement avec plus de tweets négatifs, le modèle aura une meilleure compréhension des caractéristiques spécifiques aux tweets négatifs et améliorera sa performance.")
    st.write("Cependant, il est important de garder à l'esprit la nécessité d'utiliser une approche de validation croisée (cross-validation) pour évaluer les performances du modèle avec différentes portions de tweets d'entraînement et de test. Cela permettra de garantir que le modèle généralise correctement sur de nouveaux tweets et évitera le surapprentissage (overfitting).")
    st.write("En conclusion, ajouter des tweets à l'ensemble d'entraînement est une mesure pertinente pour améliorer la performance du modèle dans la détection de tweets négatifs, mais il faut le faire avec prudence et en utilisant une validation croisée pour assurer la qualité des prédictions sur de nouveaux tweets.")



def Developement_modele():
    Introduction()
    Choix_Modele()
    Optimisation_Hyper_Params()
    Text_Resultas()







