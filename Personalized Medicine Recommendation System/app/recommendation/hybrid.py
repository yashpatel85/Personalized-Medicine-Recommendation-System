import pandas as pd
import numpy as np
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Load content-based model
nlp = spacy.load("en_core_web_md")

# Load datasets
drugs = pd.read_csv("D:\Personalized Medicine Recommendation System\project\data\drugsComTrain_raw.csv")
ratings = drugs[['drugName', 'rating', 'userId']]  # Assume you've added userId

# Collaborative Filtering (KNN)
pivot_table = ratings.pivot_table(index='userId', columns='drugName', values='rating').fillna(0)
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(pivot_table.values)

def content_based_recommend(symptom: str, top_n=5):
    symptom_vector = nlp(symptom).vector
    drug_vectors = [nlp(desc).vector for desc in drugs['review']]
    similarities = cosine_similarity([symptom_vector], drug_vectors).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    return drugs.iloc[top_indices]['drugName'].unique().tolist()

def hybrid_recommend(user_id: int, symptom: str, top_n=5):
    cbf = content_based_recommend(symptom, top_n=10)

    # Find similar users via collaborative filtering
    if str(user_id) in pivot_table.index:
        user_vec = pivot_table.loc[user_id].values.reshape(1, -1)
        distances, indices = model_knn.kneighbors(user_vec, n_neighbors=3)
        similar_users = pivot_table.iloc[indices[0]]
        top_collab = similar_users.mean(axis=0).sort_values(ascending=False).head(10).index.tolist()
    else:
        top_collab = []

    # Combine results
    combined = list(dict.fromkeys(cbf + top_collab))[:top_n]
    return combined
