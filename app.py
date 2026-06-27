import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
import joblib

import plotly.io as pio
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(layout='wide')

df = pd.read_csv('cleaned_netflix_shows_data.csv')

st.title("Netflix Shows Analysis")


x = st.sidebar.checkbox("Show Data", True, key=1)

if x:
    st.header("Dataset Sample")
    st.dataframe(df.head(6))

    
col_name = st.sidebar.selectbox("Select Column", df.columns[3:10])
col1, col2 = st.columns(2)

with col1:
    
    if col_name in ['country', 'genres', 'language', 'lead_actor']:
        st.subheader("Bar Chart")
        top = df[col_name].value_counts().head(10).index
        df_top = df[df[col_name].isin(top)]
        quantity = df_top[col_name].value_counts().reset_index()
        quantity.columns = [col_name, 'count']
        bar_fig = px.bar(quantity, x=col_name, y='count')
        st.plotly_chart(bar_fig)
    else:
        st.subheader("Histogram")
        st_hist_fig = px.histogram(df, x = col_name)
        st.plotly_chart(st_hist_fig)
    
      
with col2:
    if col_name in ['country', 'genres', 'language', 'lead_actor']:
        st.subheader("Pie Chart")
        data = df[col_name].value_counts().head(10).reset_index()
        st_pie_fig = px.pie(data, names=col_name, values='count')
        st.plotly_chart(st_pie_fig)
    else:
            st.subheader("Boxplot")
            st_box_fig = px.box(df, y = col_name)
            st.plotly_chart(st_box_fig)


tab1, tab2 = st.tabs(['Analysis','Regression'])


with tab1:
    st.title("Characterizing The Best Shows On Netflix")
    st.header("Relationship Between Popularity & Rating")
    scat_fig = px.scatter(df, x = 'rating', y = 'popularity', marginal_x = 'histogram', marginal_y = 'histogram')
    st.plotly_chart(scat_fig)
    st.write("Popularity and Rating have a very weak correlation.")
    st.subheader("Relationships with other variables:")
    numeric_df = df.select_dtypes(include=['number'])
    heat_fig = px.imshow(numeric_df.corr().round(3), text_auto = True)
    st.plotly_chart(heat_fig)
    st.write("As is shown, popularity is less predictable by given data than rating.")
    
    top = df['country'].value_counts().head(10).index
    df_top = df[df['country'].isin(top)]
    
    st.header("The Top Countries (In Terms of Popularity vs Rating)")
    
    popularity = df_top.groupby('country')['popularity'].mean().sort_values(ascending=False).head(5).reset_index()
    popularity.columns = ['country', 'avg_popularity']
    
    rating = df_top.groupby('country')['rating'].mean().sort_values(ascending=False).head(5).reset_index()
    rating.columns = ['country', 'avg_rating']
    
    top_countries = make_subplots(rows=1, cols=2, subplot_titles=("In Terms Of Popularity", "In Terms Of Rating"))
    top_countries.add_trace(go.Bar(x=popularity['country'], y = popularity['avg_popularity'], name='Popularity'), row=1, col=1)
    top_countries.add_trace(go.Bar(x=rating['country'], y = rating['avg_rating'], name='Rating'), row=1, col=2)
    st.plotly_chart(top_countries)
    st.write("""The most popular shows come from countries with large populations like the Philippines and India, 
             while the highly rated shows come from more culturally influential countries like Japan or the US.""")
    
    
    top = df['genres'].value_counts().head(10).index
    df_top = df[df['genres'].isin(top)]
    
    st.header("The Top Genres (In Terms of Popularity vs Rating)")
    
    popularity = df_top.groupby('genres')['popularity'].mean().sort_values(ascending=False).head(5).reset_index()
    popularity.columns = ['genres', 'avg_popularity']
    
    rating = df_top.groupby('genres')['rating'].mean().sort_values(ascending=False).head(5).reset_index()
    rating.columns = ['genres', 'avg_rating']
    
    top_countries = make_subplots(rows=1, cols=2, subplot_titles=("In Terms Of Popularity", "In Terms Of Rating"))
    top_countries.add_trace(go.Bar(x=popularity['genres'], y = popularity['avg_popularity'], name='Popularity'), row=1, col=1)
    top_countries.add_trace(go.Bar(x=rating['genres'], y = rating['avg_rating'], name='Rating'), row=1, col=2)
    st.plotly_chart(top_countries)
    st.write("""The most popular shows are usually non-fiction types that depict more realistic content,
             while the highly rated shows lean more toward fiction.""")


model = joblib.load('rfr_model.pkl')
selector = joblib.load('selector.pkl')
training_columns = joblib.load('training_columns.pkl')
freq_country = joblib.load('country_freq.pkl')
freq_lead_actor = joblib.load('actor_freq.pkl')

with tab2:
    st.header("Predicting Rating of a TV Show")
    st.subheader("Input:")
    
    countries = (df['country'].str.split(',').explode().str.strip().unique())
    countries = sorted(countries)
    country = st.selectbox(
        "Country",
        countries
    )

    years = list(range(2010,2027))
    release_year = st.selectbox(
        "Release Year", 
        years
    )
    
    genres = (df['genres'].str.split(',').explode().str.strip().unique())
    genres = sorted(genres)
    genre = st.selectbox(
        "Main Genre",
        genres
    )
    
    languages = sorted(df['language'].unique())
    language = st.selectbox(
        "Language",
        languages
    )
    
    popularity = st.slider(
        "Popularity",
        0.0,
        10.0,
        5.0
    )
    
    vote_count = st.slider(
        "Vote Count",
        0.0,
        10.0,
        5.0
    )
    
    lead_actor = st.text_input(
        "Lead Actor"
    )
    
    date_added = st.date_input("Date Added", value=date.today())
    year_added = date_added.year
    month_added = date_added.month
    day_of_week_added = date_added.weekday()

    input_df = pd.DataFrame({
        'country_freq': [country],
        'release_year': [release_year],
        'genres': [genre],
        'language': [language],
        'popularity': [popularity],
        'vote_count': [vote_count],
        'lead_actor_freq': [lead_actor],
        'year_added': [year_added],
        'month_added': [month_added],
        'day_of_week_added': [day_of_week_added]
    })

    input_df['country_freq'] = (
        input_df['country_freq']
        .map(freq_country)
        .fillna(0)
    )
    input_df['lead_actor_freq'] = (
        input_df['lead_actor_freq']
        .map(freq_lead_actor)
        .fillna(0)
    )

    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(
        columns=training_columns,
        fill_value=0
    )

    if st.button("Predict Rating"):
        input_selected = selector.transform(input_df)

        prediction = model.predict(input_selected)

        st.success(
            f"Predicted Rating: {prediction[0]}"
        )
