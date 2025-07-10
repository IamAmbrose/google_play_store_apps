# 📱 Exploratory Data Analysis & Prediction: Google Play Store Apps

This project analyzes apps on the Google Play Store to uncover trends, patterns, and predictive insights about app ratings.  
It combines **data cleaning**, **visualization**, **sentiment analysis of user reviews**, **time series trends**, and a **machine learning model** to predict app ratings.

---

## 📌 **Dataset**

- **Main:** `googleplaystore.csv`  
  - App details: name, category, reviews, installs, price, size, last updated, etc.
- **Reviews:** `googleplaystore_user_reviews.csv`  
  - User review text, sentiment polarity, and subjectivity.

---

## 🎯 **Objectives**

1️⃣ Clean raw app data and handle inconsistencies (e.g. `Installs`, `Price`, `Size`).  
2️⃣ Analyze key stats: ratings, installs, categories, top apps.  
3️⃣ Visualize trends: rating distribution, reviews vs. rating, top categories, free vs. paid.  
4️⃣ Perform **sentiment analysis** on user reviews:
   - Calculate average **sentiment polarity** per app.
   - Visualize **sentiment vs. rating** with scatterplots and boxplots.
5️⃣ Explore **time series** trends:
   - Analyze `Last Updated` to see how update recency relates to app ratings.
6️⃣ Build a **Random Forest model** to predict app ratings:
   - Input features: `Reviews`, `Installs`, `Price`, `Size`, `Category`, `Sentiment Polarity`.
   - Evaluate with **RMSE** — achieved **~0.23**!

---

## ⚙️ **Key Steps**

- ✅ **Data Cleaning:** Removed invalid values, converted strings to numeric, handled `Free` and `Varies with device`.
- ✅ **EDA:** Histograms, bar plots, scatterplots, boxplots.
- ✅ **Sentiment Analysis:** Used `TextBlob` to compute polarity.
- ✅ **Merge:** Joined sentiment scores to main dataset.
- ✅ **Time Series:** Parsed `Last Updated` and grouped by year.
- ✅ **Modeling:** Random Forest with `scikit-learn`.

---

## 📊 **Key Findings**

- ⭐ Most apps cluster around **4.0–4.5** ratings.
- 📈 High installs ≠ high ratings — but more reviews generally signal quality.
- 😊 Apps with **higher sentiment polarity** in user reviews tend to have higher ratings.
- 🔄 Apps updated recently tend to maintain better ratings.
- 💡 The prediction model predicts ratings within **± 0.23 stars**.

---

## 📈 **Live Dashboard**

View the interactive Streamlit dashboard here:  
**[📊 Open Dashboard]()**

---

## 📂 **Notebook**

See the full Jupyter Notebook here:  
**[📘 View on GitHub](https://github.com/IamAmbrose/google_play_store_apps/blob/main/google%20play%20store%20app%20.ipynb)**

---

## 📚 **Libraries**

- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `TextBlob`
- `scikit-learn`

---

## 🚀 **Run It Yourself**

1️⃣ Clone this repo  
2️⃣ Install dependencies:
```
pip install streamlit seaborn textblob scikit-learn
streamlit run app.py
```
For more, let's connect me on linkedin(https://www.linkedin.com/in/ambrose-henry-m-30bb84235/)

Ambrose Henry
 2025
