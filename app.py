import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from nltk.stem import PorterStemmer
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import joblib
import base64
import script1
from script1 import load_data_and_description
from script2 import display_sentiment_distribution
from script3 import generate_wordcloud
from script4 import Tweet_Mots_Cles
from script5 import Introduction,Choix_Modele, Optimisation_Hyper_Params,Text_Resultas,Developement_modele
from script6 import tokenize, load_model, deploiement


train_data = pd.read_csv('Train.csv',delimiter=',')
test_data = pd.read_csv('Test.csv',delimiter=',')

load_data_and_description(train_data,test_data)
display_sentiment_distribution(train_data)
generate_wordcloud(train_data)
Tweet_Mots_Cles(train_data)
Developement_modele()
deploiement()




