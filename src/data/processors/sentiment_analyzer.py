"""
Sentiment analysis using VADER
"""
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy

# Initialize analyzers
sia = SentimentIntensityAnalyzer()
nlp = spacy.load("en_core_web_sm")


def analyze_sentiment(text):
    """
    Analyze sentiment of text using VADER

    Args:
        text: Text to analyze

    Returns:
        float: Compound sentiment score (-1 to 1)
    """
    sentiment_scores = sia.polarity_scores(text)
    return sentiment_scores['compound']


def extract_companies_from_text(text, company_lookups):
    """
    Extract company names from text using NER

    Args:
        text: Text to analyze
        company_lookups: Dictionary from load_company_tickers_json()

    Returns:
        list: List of (company_name, ticker) tuples
    """
    # Import here to avoid circular dependency
    from src.data.processors.company_validator import validate_company_exists

    doc = nlp(text)
    companies = []

    for ent in doc.ents:
        if ent.label_ == "ORG":
            if company_lookups:
                is_valid, ticker, official_name = validate_company_exists(ent.text, company_lookups)
                if is_valid:
                    companies.append((official_name, ticker))

    return companies