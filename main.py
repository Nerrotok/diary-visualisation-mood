import streamlit as st
import plotly.express as px
import glob
from nltk.sentiment import SentimentIntensityAnalyzer

st.header("Diary Tone")
st.subheader("Positivity")

# Get diary entry directories
diary_entries = [entry for entry in glob.glob("diary/*.txt")]

# Get the entry dates without the directory
entry_dates = [entry.strip(".txt").replace(entry[0:6], "",) for entry in diary_entries]

# Make a list of the diary entry texts
diary_entries_text = []
for entry in diary_entries:
    with open(entry, "r", encoding="utf8") as file:
        stringy_entry = file.read()
    diary_entries_text.append(stringy_entry)

analyser = SentimentIntensityAnalyzer()

# Get the sentiment dictionaries from the texts
sentiment_list = []
for string in diary_entries_text:
    score = analyser.polarity_scores(string)
    sentiment_list.append(score)

pos_values = [i["pos"] for i in sentiment_list]

figure_pos = px.line(x=entry_dates, y=pos_values, labels={"x": "Entry Dates", "y": "Positivity"})
st.plotly_chart(figure_pos)

st.subheader("Negativity")

neg_values = [i["neg"] for i in sentiment_list]

figure_neg = px.line(x=entry_dates, y=neg_values, labels={"x": "Entry Dates", "y": "Negativity"})
st.plotly_chart(figure_neg)
