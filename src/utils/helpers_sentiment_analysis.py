import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from transformers import pipeline
from nrclex import NRCLex
import ast

# nltk.download('vader_lexicon')
# nltk.download('punkt')
# nltk.download('wordnet')

def preprocess_text(text):
    """Preprocess text: tokenization, lemmatization"""
    tokens = word_tokenize(text.lower())
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)

def get_nltk_sentiment(text):
    """NLTK Sentiment Analysis using VADER"""
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    return 1 if scores['pos'] > 0 else 0

def get_textblob_sentiment(text):
    """TextBlob Sentiment Analysis"""
    analysis = TextBlob(text)
    return analysis.sentiment.polarity  # Returns a polarity score (-1 to 1)

def get_vader_sentiment(text):
    """VADER Sentiment Analysis"""
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)['compound']

def extract_emotions(text):
    """Emotion analysis using NRC Lexicon"""
    try:
        emotion = NRCLex(text)
        return emotion.raw_emotion_scores
    except Exception as e:
        return str(e)

def classify_sentiment_from_emotions(emotions):
    """Classify sentiment based on emotions"""
    positive_emotions = ['positive', 'anticipation', 'surprise', 'joy', 'trust']
    negative_emotions = ['negative', 'anger', 'fear', 'disgust', 'sadness']
    positive_score = sum(emotions.get(emotion, 0) for emotion in positive_emotions)
    negative_score = sum(emotions.get(emotion, 0) for emotion in negative_emotions)
    return 'POSITIVE' if positive_score > negative_score else 'NEGATIVE'

def huggingface_sentiment_analysis(texts, model_name="distilbert-base-uncased-finetuned-sst-2-english", batch_size=16):
    """Hugging Face Transformer Sentiment Analysis in batches"""
    sentiment_analyzer = pipeline("sentiment-analysis", model=model_name, truncation=True)
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        results.extend(sentiment_analyzer(batch))
    return results
