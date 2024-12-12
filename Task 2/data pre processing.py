#Sentimental analysis on X(Twitter)  on topic of AI
#Data set taken from kaggle
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
from textblob import TextBlob

#downloading  the vader_lexicon for the sentiment analysis foor this project
nltk.download('vader_lexicon')
#Dataset loading
data = pd.read_csv(r"C:\Users\chintu\Desktop\Large_English_Language_scrapper.csv")



# Initialize VADER sentiment intensity analyzer
sia = SentimentIntensityAnalyzer()

# Function to calculate sentiment using VADER
def vader_sentiment_score(text):
    return sia.polarity_scores(text)['compound']

def vader_sentiment_label(text):
    compound = vader_sentiment_score(text)
    if compound >= 0.05:
        return 'Positive'
    elif compound <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Apply sentiment analysis with VADER
data['vader_sentiment'] = data['content'].apply(vader_sentiment_score)
data['sentiment_label'] = data['content'].apply(vader_sentiment_label)

# Data cleaning (example: removing nulls)
data = data.dropna(subset=['content'])

# Save the results to a new file
data.to_csv(r"C:\Users\chintu\Desktop\Processed_Sentiment_Analysis.csv", index=False)

# Display the results
print(data[['content', 'vader_sentiment', 'sentiment_label']])
