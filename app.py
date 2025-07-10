import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
from sklearn.ensemble import RandomForestRegressor

sns.set_theme(style="whitegrid")

@st.cache_data
def load_data():
    df = pd.read_csv("googleplaystore.csv")
    df_reviews = pd.read_csv("googleplaystore_user_reviews.csv")
    return df, df_reviews

df, df_reviews = load_data()

st.title("📱 Google Play Store Apps Dashboard")
-
# Data Cleaning

df['Reviews'] = pd.to_numeric(df['Reviews'], errors='coerce')
df['Installs'] = df['Installs'].astype(str).str.replace('[+,]', '', regex=True)
df['Installs'] = pd.to_numeric(df['Installs'], errors='coerce')
df['Price'] = df['Price'].astype(str).str.replace('$', '', regex=False).replace('Free', '0')
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

def size_to_kb(size):
    if size == 'Varies with device' or pd.isna(size):
        return np.nan
    elif 'M' in size:
        return float(size.replace('M', '').replace(',', '')) * 1024
    elif 'k' in size:
        return float(size.replace('k', '').replace(',', ''))
    else:
        return np.nan

df['Size_KB'] = df['Size'].apply(size_to_kb)
df['Last Updated'] = pd.to_datetime(df['Last Updated'], errors='coerce')

# Sentiment polarity
df_reviews['Translated_Review'] = df_reviews['Translated_Review'].fillna("")
df_reviews['Polarity'] = df_reviews['Translated_Review'].apply(lambda x: TextBlob(x).sentiment.polarity)
app_sentiment = df_reviews.groupby('App')['Polarity'].mean().reset_index()
df = df.merge(app_sentiment, on='App', how='left')

# Show basic info
st.subheader("Dataset Preview")
st.write(df.head())

# EDA Visuals

st.subheader("Ratings Distribution")
fig, ax = plt.subplots()
sns.histplot(df['Rating'], bins=20, ax=ax)
st.pyplot(fig)

st.subheader("Top Categories")
top_categories = df['Category'].value_counts().head(10)
fig, ax = plt.subplots()
sns.barplot(y=top_categories.index, x=top_categories.values, ax=ax)
st.pyplot(fig)

st.subheader("Installs vs. Rating")
fig, ax = plt.subplots()
sns.scatterplot(x='Installs', y='Rating', data=df, ax=ax)
ax.set_xscale('log')
st.pyplot(fig)

st.subheader("Paid vs. Free Rating")
fig, ax = plt.subplots()
sns.boxplot(x='Type', y='Rating', data=df, ax=ax)
st.pyplot(fig)

# Sentiment Analysis Plots

st.subheader("Sentiment Polarity vs. Rating")
fig, ax = plt.subplots()
sns.scatterplot(x='Polarity', y='Rating', data=df, ax=ax)
st.pyplot(fig)

df['Rating_Bin'] = pd.cut(df['Rating'], bins=[0, 2, 3, 4, 5],
                          labels=['Low', 'Fair', 'Good', 'Excellent'])

st.subheader("Boxplot: Polarity by Rating Bin")
fig, ax = plt.subplots()
sns.boxplot(x='Rating_Bin', y='Polarity', data=df, ax=ax)
st.pyplot(fig)

#Simple Prediction

st.subheader("📈 Predict App Rating")

reviews = st.slider("Reviews", 0, int(df['Reviews'].max()), 1000)
installs = st.slider("Installs", 0, int(df['Installs'].max()), 10000)
price = st.slider("Price ($)", 0.0, 50.0, 0.0)
size_kb = st.slider("Size (KB)", 0, int(df['Size_KB'].max()), 10000)
polarity = st.slider("Polarity", -1.0, 1.0, 0.0)

df_model = df[['Rating', 'Reviews', 'Installs', 'Price', 'Size_KB', 'Polarity', 'Category']].dropna()
df_model = pd.get_dummies(df_model, columns=['Category'], drop_first=True)

X = df_model.drop('Rating', axis=1)
y = df_model['Rating']

model = RandomForestRegressor()
model.fit(X, y)

# Create dummy input
user_input = pd.DataFrame([[reviews, installs, price, size_kb, polarity] + [0]*(X.shape[1]-5)], columns=X.columns)

pred_rating = model.predict(user_input)[0]
st.success(f"✅ Predicted Rating: {pred_rating:.2f}")

#  Download
st.subheader("📥 Download Cleaned Data")
st.download_button(
    label="Download CSV",
    data=df.to_csv(index=False).encode('utf-8'),
    file_name='cleaned_googleplaystore.csv',
    mime='text/csv'
)
