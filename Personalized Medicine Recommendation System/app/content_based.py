from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class ContentRecommender:
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index()
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.similarity_matrix = None
        self.tfidf_matrix = None

    def build_model(self):
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['clean_review'])
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)

    def recommend(self, drug_name, condition=None, top_k=5):
        drug_rows = self.df[self.df['drugName'].str.lower() == drug_name.lower()]
        if condition:
            drug_rows = drug_rows[drug_rows['condition'].str.lower() == condition.lower()]
        if drug_rows.empty:
            return f"No matches found for '{drug_name}' with condition '{condition}'"

        idx = drug_rows.iloc[0]['index']
        sim_scores = list(enumerate(self.similarity_matrix[int(idx)]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        top_indices = [i for i, _ in sim_scores[1:top_k+1]]
        return self.df.iloc[top_indices][['drugName', 'condition', 'rating', 'clean_review']].drop_duplicates('drugName')
