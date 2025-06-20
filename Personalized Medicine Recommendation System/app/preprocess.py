import re
import spacy

nlp = spacy.load("en_core_web_sm")

def clean_review_spacy(text):
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha and len(token) > 2]
    return ' '.join(tokens)

def preprocess_dataframe(df):
    df['clean_review'] = df['review'].apply(clean_review_spacy)
    return df
