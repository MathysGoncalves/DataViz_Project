import pandas as pd
import datetime as dt
import numpy as np
import time
import random
import pandas_profiling

import streamlit as st
import streamlit.components.v1 as components 
import altair as alt
import seaborn as sns
import plotly.express as px
import pydeck as pdk
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as RMSE
import joblib
from matplotlib.pyplot import figure


pd.set_option('display.max_columns', None)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="wide")

def st_log(func):
    def log_func(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time() - start
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)

        st.text("Log (%s): the function `%s` tooks %0.4f seconds" % (current_time, func.__name__, end))
        file1 = open("Logs.txt","a")
        file1.write("\nLog (%s): the function `%s` tooks %0.4f seconds" % (current_time, func.__name__, end))
        file1.close()
        return res

    return log_func

############ Data Loading & Preparation ############

@st.cache(allow_output_mutation=True)
def load_data():
    #df_2016 = pd.read_csv("full_2016.csv")
    #df_2017 = pd.read_csv("full_2017.csv")
    #df_2018 = pd.read_csv("full_2018.csv")
    #df_2019 = pd.read_csv("full_2019.csv")
    #df = pd.read_csv("full_2020.csv")
    df = pd.read_csv("Sample.csv")

    #df = pd.concat([df_2018, df_2019, df_2020], ignore_index=True)
    
    df['Date'] = pd.to_datetime(df['date_mutation'])
    df['Day'] = df['Date'].dt.weekday
    df['Month'] = df['Date'].dt.month
    df['prix_m2'] = df['valeur_fonciere'] / df['surface_reelle_bati']
    df['count'] = 1

    df = drop_data(df)

    df['code_departement'] = df['code_departement'].astype(str)
    df['latitude'] = pd.to_numeric(df['latitude']) 
    df['longitude'] = pd.to_numeric(df['longitude'])
    #df = df.dropna(subset=['longitude', 'latitude'])
    return df

@st.cache(allow_output_mutation=True)
def drop_data(df):
    df_drop = df.drop(columns=['numero_disposition', 'adresse_numero', 'adresse_suffixe', 'adresse_code_voie',
                        'code_commune', 'ancien_code_commune', 'ancien_nom_commune', 'id_parcelle',	'ancien_id_parcelle',	'numero_volume', 'lot1_numero',	'lot1_surface_carrez', 'lot2_numero', 'lot2_surface_carrez', 'lot3_numero',	'lot3_surface_carrez', 'lot4_numero', 'lot4_surface_carrez', 'lot5_numero', 'lot5_surface_carrez', 'code_nature_culture', 'code_nature_culture_speciale', 'nombre_lots', 'code_type_local', 'nature_culture', 'nature_culture_speciale', 'date_mutation'])

    df_dupli = df_drop.drop_duplicates()#Delete exact Same rows
    df_dropna = df_dupli.dropna(subset=['longitude', 'latitude', 'valeur_fonciere', 'type_local'])
    return df_dropna

@st.cache(allow_output_mutation=True)
def sample_data(df, fract):
    sample = df.sample(frac=fract, random_state=2)
    return sample

@st.cache(allow_output_mutation=True)
def load_model(filename):
    return joblib.load(filename)

############ Plot Definition ############

def px_line_bar(col, df, x, y):
    fig = px.bar(df, x=x, y=y)
    #fig.add_hline(y.mean())
    col.plotly_chart(fig)

def var_importance(model, col):
	importance = model.feature_importances_# get importance
	plt.bar([x for x in range(len(importance))], importance) # plot feature importance
	col.pyplot(plt.show())

############      Main     ############

@st_log
def main():

    df = load_data()
    sample = sample_data(df, 0.005)

    st.markdown("<h1 style='text-align: center;'>Data Viz Project - « Demandes de valeurs foncières »</h1>", unsafe_allow_html=True)

    st.text("")
    st.markdown("<h5>The project idea is quite simple : Applying the process of visual data exploration we have seen during the labs to the dataset Demandes de valeurs foncières</h5>", unsafe_allow_html=True)

    #Sidebar 

    if st.checkbox('Use Filter (much slower)'):
        st.sidebar.title("Filter")

        slider1, slider2 = st.sidebar.date_input('Select Date', [df['Date'].min(), df['Date'].max()])
        mask1 = sample['Date'].isin([slider1, slider2])

        types = st.sidebar.multiselect("Select a local type:",  pd.unique(df['type_local']))
        mask2 = sample['type_local'].isin(types)

        commune = st.sidebar.multiselect("Select a city:",  pd.unique(df['nom_commune']))
        mask3 = sample['nom_commune'].isin(commune)

        surface_min, surface_max = st.sidebar.slider('Select a range of surface', df['surface_reelle_bati'].min(), df['surface_reelle_bati'].max(), (0.0, 1000.0))
        mask4 = sample['surface_reelle_bati'].isin([surface_min, surface_max])
        
        piece = st.sidebar.multiselect('Select a nulber of room', pd.unique(df['nombre_pieces_principales']))
        mask5 = sample['nombre_pieces_principales'].isin(piece)

        masks = mask1 & mask2 & mask3 & mask4 & mask5

        if st.sidebar.button('Apply'):
            sample = sample[masks]


    #Description
    col1, col2 = st.columns(2)

    col1.text("Head of the dataset")
    col1.write(sample.head(10))

    col2.text("Tail of the dataset")
    col2.write(sample.tail(10))

    col1, col2 = st.columns(2)

    if st.checkbox('Show description'):
        st.text("")
        st.text("")       
        col1.text("Dataset Description")
        st.text("")
        col1.write(sample.describe())

        col2.text("Nan after preparation")
        nan = sample.isna().sum()
        col2.write(nan)
        if st.checkbox('Show more description (Very slow computation...)'):
            col1.text("Pandas Profiling Description")
            st.text("")
            st.write(sample.profile_report())   

    #Search 
    st.write("")
    st.write("")
    st.write("")
    st.markdown("<h5 style='text-align: center;'>Here you can make a research on a mutation if you know it's Id : </h5>", unsafe_allow_html=True)
    st.write("")

    id_input = st.text_input("Search mutation by Id", "Ex : 2020-805242")
    mask = df['id_mutation'] == id_input
    search = df[mask]
    st.dataframe(search)

    #Predict
    st.write("")
    st.write("")
    st.write("")
    st.markdown("<h5 style='text-align: center;'>Here you can make a prediction on the \"valeur fonciere\" :</h5>", unsafe_allow_html=True)
    st.write("The model was previously prepared on the \"prediction\" notebook and saved thanks to joblib. The model is NOT accurate because it is not the goal of the exercise")
    st.write("")
    col1, col2, col3, col4, col5 = st.columns(5)

    code_postal = col1.text_input("code_postal", 93320)
    surf_bat = col2.text_input("surface_reelle_bati", 57)
    piece_p = col3.text_input("nombre_pieces_principales", 3)
    surf_terr = col4.text_input("surface_terrain", 700)
    
    new_input = [[code_postal, surf_bat, piece_p, surf_terr]]

    #model = load_model('ExtraTrees.sav')
    #new_output = model.predict(new_input)
    
    #col5.metric(label="Sell Price", value="%.2f" %new_output)
    col5.metric(label="Sell Price", value="Prediction...")

    st.text("")
    col1, col2 = st.columns((1, 2))
    if st.checkbox('Show Accuracy of the model'):
        st.text("")
        st.text("")
        col1.write("Features importance")
        var_importance(model, col1)
        col2.write("Valeur Fonciere predicted")
        col2.image("Scatter.png")
        

    # Plot 
    st.text("")
    st.text("")
    st.text("")
    st.markdown("<h5 style='text-align: center;'>Here the proportion of the local type and the nature on the full dataset :</h5>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    fig = px.pie(df, values='count', names='type_local', title='Repartition of local type')
    col1.plotly_chart(fig)

    fig = px.pie(df, values='count', names='nature_mutation', title='Repartition of matution nature')
    col2.plotly_chart(fig)

    #Metrics
    st.text("")
    st.text("")
    st.text("")
    st.markdown("<h5 style='text-align: center;'>Here we can look some important metrics of our our Dataset on the number of mutations or the evolution of the price. Use filter in the sidebar to play with it ! </h5>", unsafe_allow_html=True)
    st.text("")
    col1, col2 = st.columns((1,2))
    
    average_pricem2 = sample['prix_m2'].mean()
    nbr_mutation = len(sample)
    average_price = sample['valeur_fonciere'].mean()
    month_analysis = sample.groupby('Month')['valeur_fonciere'].mean()

    col2.line_chart(month_analysis)
    st.text("")
    st.text("")

    col1.metric(label="Average Price m²", value="%.2f" %average_pricem2)
    col1.metric(label="Nbr of mutations", value=nbr_mutation)
    col1.metric(label="Average Price", value="%.2f" %average_price)


    #bar plot for top10 commune
    st.text("")
    st.text("")
    st.text("")
    st.markdown("<h5 style='text-align: center;'>Here the cities and departments with the most sales!</h5>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    top_commune = (df.groupby('nom_commune')['count'].sum()).nlargest(15, keep='last').reset_index()
    px_line_bar(col1, top_commune, 'nom_commune', 'count')

    #bar
    top_departement = (df.groupby('code_departement')['count'].sum()).nlargest(15).reset_index()
    px_line_bar(col2, top_departement, 'code_departement', 'count')


    #Map des mutations
    st.text("")
    st.text("")
    st.text("")
    st.markdown("<h5 style='text-align: center;'>Here is a map of the repartition of our mutations. Play with the filter ! (With a sample of the dataset)</h5>", unsafe_allow_html=True)
    st.write("")
    st.map(sample)


if __name__ == "__main__":
    main()