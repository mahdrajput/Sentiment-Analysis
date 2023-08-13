from flask import Flask, render_template, request
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

app = Flask(__name__)

# Initialize the SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

def classify_sentiment(compound_score):
    if compound_score >= 0.5:
        return 'Positive 🙂'
    elif 0.1 <= compound_score < 0.5:
        return 'Happy 😃'
    elif -0.1 <= compound_score < 0.1:
        return 'Neutral ☹️'
    elif -0.5 <= compound_score < -0.1:
        return 'Sad 😢'
    else:
        return 'Angry 😡'

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form['text']
    sentiment_scores = sia.polarity_scores(text)

    sentiment = classify_sentiment(sentiment_scores['compound'])

    return render_template('home.html', text=text, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
