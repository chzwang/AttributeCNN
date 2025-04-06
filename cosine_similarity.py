import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

df = pd.read_csv('celeb_attr.csv')
def top_k(query, k=3):
    filenames = df.iloc[:, 0].values
    features = df.iloc[:, 1:].values

    query_vector = np.array(query).reshape(1, -1)

    similarities = cosine_similarity(query_vector, features).flatten()
    top_k_idx = np.argsort(similarities)[::-1][:k]

    return pd.DataFrame({
        "filename": filenames[top_k_idx],
        "similarity": similarities[top_k_idx]
    })