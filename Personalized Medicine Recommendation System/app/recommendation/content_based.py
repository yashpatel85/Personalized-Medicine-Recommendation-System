import pandas as pd
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from typing import List

# Load spaCy English model
nlp = spacy.load("en_core_web_md")  # Use medium model for better vector quality

# Load and preprocess the dataset
def load_drug_data(filepath: str):
    df = pd.read_csv(filepath)
    df = df[['drugName', 'condition', 'review']].dropna()
    df['combined_text'] = df['drugName'].astype(str) + " " + df['condition'].astype(str) + " " + df['review'].astype(str)
    df.drop_duplicates(subset='combined_text', inplace=True)
    return df

# Convert text to spaCy vectors
def text_to_vector(text: str):
    return nlp(text).vector

# Build similarity matrix for top N results
def get_top_recommendations(user_input: str, df: pd.DataFrame, top_n: int = 5):
    user_vec = text_to_vector(user_input)
    doc_vectors = df['combined_text'].apply(text_to_vector)
    similarities = cosine_similarity([user_vec], list(doc_vectors))[0]
    
    top_indices = similarities.argsort()[::-1][:top_n]
    recommendations = df.iloc[top_indices][['drugName', 'condition', 'review']]
    return recommendations.to_dict(orient='records')


med_conditions = {
    "headache": ["Tylenol", "Advil"],
    "cold": ["Benadryl", "Zyrtec"],
    "diabetes": ["Metformin", "Januvia"],
}


def recommend_medicines(condition: str):
    condition = condition.lower()
    for key in med_conditions:
        if key in condition:
            return med_conditions[key]
            
            return [(row["drugName"], row["condition"], row["review"]) for idx, row in top_matches.iterrows()]
        
    return ["No recommendations found."]