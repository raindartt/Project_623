import pandas as pd
import numpy as np
import string
import random

# used for word2vec demo
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_samples, silhouette_score

# used for randomforest_demo
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# to visualize tree
import pydot
from sklearn.tree import export_graphviz
from subprocess import call
from IPython.display import Image

# https://stackoverflow.com/questions/17778394/list-highest-correlation-pairs-from-a-large-correlation-matrix-in-pandas
def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]


# https://stackoverflow.com/questions/17778394/list-highest-correlation-pairs-from-a-large-correlation-matrix-in-pandas
def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop


# from CSCE 623 Homework 4
def correlation_views(df, print_top=5, diag=-2):
    corr = df.corr()
    np.fill_diagonal(corr.values, diag)
    with pd.option_context('precision', 2):
        display(corr.style.background_gradient(vmin=-1.0, vmax=1.0, cmap='bwr'))
    corr = get_top_abs_correlations(df=df, n=print_top)
    print("Top Absolute Correlations (excluding diagonals)")
    print(corr)
    return corr


# from https://dylancastillo.co/nlp-snippets-cluster-documents-using-word2vec/
def vectorize(list_of_docs, model):
    """Generate vectors for list of documents using a Word Embedding

    Args:
        list_of_docs: List of documents
        model: Gensim's Word Embedding

    Returns:
        List of document vectors
    """
    features = []

    for tokens in list_of_docs:
        zero_vector = np.zeros(model.vector_size)
        vectors = []
        for token in tokens:
            if token in model.wv:
                try:
                    vectors.append(model.wv[token])
                except KeyError:
                    continue
        if vectors:
            vectors = np.asarray(vectors)
            avg_vec = vectors.mean(axis=0)
            features.append(avg_vec)
        else:
            features.append(zero_vector)
    return features


# from https://dylancastillo.co/nlp-snippets-cluster-documents-using-word2vec/
def mbkmeans_clusters(
    X, 
    k, 
    mb, 
    print_silhouette_values, 
):
    """Generate clusters and print Silhouette metrics using MBKmeans

    Args:
        X: Matrix of features.
        k: Number of clusters.
        mb: Size of mini-batches.
        print_silhouette_values: Print silhouette values per cluster.

    Returns:
        Trained clustering model and labels based on X.
    """
    km = MiniBatchKMeans(n_clusters=k, batch_size=mb).fit(X)
    print(f"For n_clusters = {k}")
    print(f"Silhouette coefficient: {silhouette_score(X, km.labels_):0.2f}")
    print(f"Inertia:{km.inertia_}")

    if print_silhouette_values:
        sample_silhouette_values = silhouette_samples(X, km.labels_)
        print(f"Silhouette values:")
        silhouette_values = []
        for i in range(k):
            cluster_silhouette_values = sample_silhouette_values[km.labels_ == i]
            silhouette_values.append(
                (
                    i,
                    cluster_silhouette_values.shape[0],
                    cluster_silhouette_values.mean(),
                    cluster_silhouette_values.min(),
                    cluster_silhouette_values.max(),
                )
            )
        silhouette_values = sorted(
            silhouette_values, key=lambda tup: tup[2], reverse=True
        )
        for s in silhouette_values:
            print(
                f"    Cluster {s[0]}: Size:{s[1]} | Avg:{s[2]:.2f} | Min:{s[3]:.2f} | Max: {s[4]:.2f}"
            )
    return km, km.labels_



# combined snippets from https://dylancastillo.co/nlp-snippets-cluster-documents-using-word2vec/
def word2vec_demo(names=[], sentences=[], vector_size=100, workers=1, seed=random.seed(42)):
    model = Word2Vec(sentences=sentences, vector_size=vector_size, workers=workers, seed=seed)
    vectorized_docs = vectorize(sentences, model=model)
    print(f'Rows of vectorized array: {len(vectorized_docs)}, Length of first vector: {len(vectorized_docs[0])}')
    clustering, cluster_labels = mbkmeans_clusters(
        X=vectorized_docs,
        k=50,
        mb=3072,
        print_silhouette_values=True,
    )
    df_clusters = pd.DataFrame({
        "text": names,
        "tokens": sentences,
        "cluster": cluster_labels
    })
    display(df_clusters)
    print("Most representative terms per cluster (based on centroids):")
    for i in range(50):
        tokens_per_cluster = ""
        most_representative = model.wv.most_similar(positive=[clustering.cluster_centers_[i]], topn=5)
        for t in most_representative:
            tokens_per_cluster += f"{t[0]} "
        print(f"Cluster {i}: {tokens_per_cluster}")
    test_cluster = 1
    most_representative_docs = np.argsort(
        np.linalg.norm(vectorized_docs - clustering.cluster_centers_[test_cluster], axis=1)
    )
    print('Most representative documents per cluster:')
    for d in most_representative_docs[:8]:
        print(sentences[d])
        print("-------------")
        
    return df_clusters



def randomforest_demo(sentences=[], y=[], ynames=['target'], seed=42):
    vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
    X = vectorizer.fit_transform(sentences).toarray()
#     display(vectorizer.vocabulary_)
    word_names = vectorizer.vocabulary_
    tfidfconverter = TfidfTransformer()
    X2 = tfidfconverter.fit_transform(X).toarray()
    display(X2)
    rand=np.random.RandomState(seed)
#     display(X2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand)
    classifier = RandomForestClassifier(n_estimators=1000, random_state=rand)
    classifier.fit(X_train, y_train)
#     display(classifier.classes_)
    y_pred = classifier.predict(X_test)
    
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    print(accuracy_score(y_test, y_pred))
#     visualize_rf(classifier, word_names)
    return classifier


# https://towardsdatascience.com/how-to-visualize-a-decision-tree-from-a-random-forest-in-python-using-scikit-learn-38ad2d75f21c
def visualize_rf(model, ynames):
    # Export as dot file
    estimator = model.estimators_[5]
    img_dot = 'tree.dot'
    img_png = 'tree.png'
    export_graphviz(estimator, out_file=img_dot,
#                     class_names = ynames,
                    rounded = True, proportion = False, 
                    precision = 2, filled = True)
    
#     call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])
#     Image(filename = 'tree.png')
    (graph,) = pydot.graph_from_dot_file(img_dot)
    graph.write_png(img_png)
    graph.draw(img_png)
    return