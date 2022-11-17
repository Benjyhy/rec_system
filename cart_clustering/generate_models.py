# Import all necessary libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
import pickle
from sklearn.decomposition import TruncatedSVD
import os

# Define carts : a ticket contains one or more libelles
carts = pd.read_csv("KaDo.csv", usecols=["TICKET_ID", "LIBELLE"])

# Get a list of all unique tickets
tickets = np.unique(carts["TICKET_ID"])

# Apply fonction using list of unique tickets and the carts_endoded df
tickets_items_list = pickle.load(open("items_list_for_ticket.txt", "rb"))

# Vectorize each string and return all vectors and feature names
def prepSparseMatrix(list_of_str):
    cv = CountVectorizer(token_pattern=r"[^\,\ ]+", lowercase=False)
    sparseMatrix = cv.fit_transform(list_of_str)
    return sparseMatrix, cv.get_feature_names_out()


# Get sparseMatrix
sparseMatrix, feature_names = prepSparseMatrix(tickets_items_list)

# Create necessary directories if needed
necessary_paths = ["models/dbscan/", "models/kmeans/", "models/spectral/", "models/dd/"]
for n_p in necessary_paths:
    exists = os.path.exists(n_p)
    if not exists:
        os.makedirs(n_p)
        print(f"{n_p} was created")

# Transform matrix in value pairs
svd = TruncatedSVD()

for i in range(0, 5):
    df_svd = svd.fit_transform(sparseMatrix[i * 10000 : (i + 1) * 10000])

    # Fitting Data on Model and saving it with pickle (kmeans)
    kmeans = KMeans(
        n_clusters=8, init="k-means++", max_iter=300, n_init=10, random_state=0
    )
    kmeans.fit(df_svd)
    pickle.dump(kmeans, open(f"models/kmeans/{i}.pkl", "wb"))

    # Fitting Data on Model and saving it with pickle (dbscan)
    dbscan = DBSCAN(eps=0.2, min_samples=5, algorithm="ball_tree")
    dbscan.fit(df_svd)
    pickle.dump(dbscan, open(f"models/dbscan/{i}.pkl", "wb"))

    # Fitting Data on Model and saving it with pickle (spectral)
    spectral = SpectralClustering(
        n_clusters=8, assign_labels="discretize", random_state=0
    )
    spectral.fit(df_svd)
    pickle.dump(spectral, open(f"models/spectral/{i}.pkl", "wb"))
