import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import warnings
import random
from collections import Counter

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

# import graphviz
from sklearn.tree import export_graphviz
from subprocess import call
from IPython.display import Image

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import HalvingGridSearchCV


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
        for j in range(0, i + 1):
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


def y_encoder(text, names):
    """converts text to numbers based on its index in the array of names"""
    return names.index(text)


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
    # print('Vectorizer vocabulary')
    # display(vectorizer.vocabulary_)
    word_names = vectorizer.vocabulary_
    tfidfconverter = TfidfTransformer()
    rand = np.random.RandomState(seed)
    # X2 = tfidfconverter.fit_transform(X).toarray()
    # print('X2')
    # display(X2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand)
    classifier = RandomForestClassifier(n_estimators=1000, random_state=rand)
    classifier.fit(X_train, y_train)
    print('classifier classes')
    display(classifier.classes_)
    y_pred = classifier.predict(X_test)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))
    # visualize_rf(classifier, word_names)
    return classifier


# https://towardsdatascience.com/how-to-visualize-a-decision-tree-from-a-random-forest-in-python-using-scikit-learn-38ad2d75f21c
def visualize_rf(model, ynames):
    # Export as dot file
    estimator = model.estimators_[5]
    img_dot = 'tree.dot'
    img_png = 'tree.png'
    export_graphviz(estimator, out_file=img_dot,
                    class_names=ynames,
                    rounded=True, proportion=False,
                    precision=2, filled=True)

    # call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])
    # Image(filename='tree.png')
    (graph,) = pydot.graph_from_dot_file(img_dot)
    graph.write_png(img_png)
    graph.draw(img_png)
    return


# vocabulary counter originally developed for binary classification Naive Bayes and Logistic Regression
# expanded here for multiclass problem
def build_vocab_word_counts(x, y):
    """
    parameters:
    x - an array of arrays of tokenized word strings that make up sentences
    y - an array of numbers that correspond to classification label for each sentence in x

    returns two dictionaries:
    1. total_vocab is a dictionary where keys are words in x and values are
    arrays that count the number of times each word appears in each class of y
    2. class_vocab is a dictionary where keys are class numbers and values are
    Counter objects where keys are words and values are counts for each word in that class
    """
    # vocabulary of all words seen in any sentence. keys are words.
    # values are arrays whose indices correspond to class number and values are the count of that word in that class
    total_vocab = {}
    class_vocab = {}
    num_classes = y.max() + 1
    for index, token_list in x.items():
        class_num = y[index]
        if class_num not in class_vocab:
            class_vocab[class_num] = Counter()

        for word in token_list:
            # add or increment word count to total vocabulary
            if word not in total_vocab:
                total_vocab[word] = np.zeros((num_classes,))
            total_vocab[word][class_num] += 1

            # add or increment word count to class vocabulary
            if word not in class_vocab[class_num]:
                class_vocab[class_num][word] = 0
            class_vocab[class_num][word] += 1

    return total_vocab, class_vocab


# def txt_to_class_num(x, class_vocab, increment='one'):
#     - increment: whether to increment by 1 or dict[key] value for each word found
def txt_to_class_num(x, class_vocab):
    """
    Parameters:
    - x: array of sentences
    - class_vocab: dictionary of Counter objects with vocabularies for each class

    Returned vars:
    - x_class_nums: 2D array of length(x) by [number of classes] which contains the increments of
        how many words in each sentence are found in each class's vocabulary
    - x_sent_lens: 1D array of length(x) which contains the length of each sentence
    """
    num_observations = len(x)
    num_classes = len(class_vocab)
    x_class_nums = np.zeros((num_observations, num_classes))
    x_sent_lens = np.zeros(num_observations)
    i = -1
    for index, sentence in x.items():
        i += 1
        x_sent_lens[i] = len(sentence)
        for k in range(num_classes):
            class_counter = class_vocab[k]
            for word in sentence:
                if word in class_counter:
                    x_class_nums[i][k] += 1
                    # if increment == 'one':
                    #     x_class_nums[i][k] += 1
                    # elif increment == 'value':
                    #     x_class_nums[i][k] += class_counter[word]

    return x_class_nums, x_sent_lens


def txt_to_total_num(x, total_vocab):
    """
    Parameters:
    - x: array of sentences
    - total_vocab: dictionary of Counter objects with vocabularies for each class

    Returned vars:
    - x_total_nums: 2D array of length(x) by [number of words] which contains the increments of
        each vocabulary word in each sentence
    - x_sent_lens: 1D array of length(x) which contains the length of each sentence
    """
    num_observations = len(x)
    num_words = len(total_vocab)
    x_total_nums = np.zeros((num_observations, num_words))
    x_sent_lens = np.zeros(num_observations)
    vocab_list = list(total_vocab.keys())
    i = -1
    for index, sentence in x.items():
        i += 1
        x_sent_lens[i] = len(sentence)
        for word in sentence:
            if word in vocab_list:
                x_total_nums[i][vocab_list.index(word)] += 1

    return x_total_nums, x_sent_lens


# snipped from https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f
@ignore_warnings(category=ConvergenceWarning)
def model_test(features=[], labels=[], folds=5, seed=623):
    models = [
        RandomForestClassifier(n_estimators=200, max_depth=7, random_state=seed),
        LinearSVC(max_iter=2500, random_state=seed),
        MultinomialNB(),
        LogisticRegression(solver='lbfgs', max_iter=500, multi_class='multinomial', random_state=seed),
    ]
    cv_df = pd.DataFrame(index=range(folds * len(models)))
    entries = []
    for model in models:
        model_name = model.__class__.__name__
        scores = cross_val_score(model, features, labels, scoring='f1_micro', cv=StratifiedKFold(folds))
        for fold_idx, score in enumerate(scores):
            entries.append((model_name, fold_idx, score))
    cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'f1_micro'])

    return cv_df


# snipped from https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f
def model_eval(cv_df, ylabel='f1_micro'):
    sns.boxplot(x='model_name', y=ylabel, data=cv_df)
    sns.stripplot(x='model_name', y=ylabel, data=cv_df,
                  size=8, jitter=True, edgecolor="gray", linewidth=2)
    plt.show()

    display(cv_df.groupby('model_name').f1_micro.mean())

    return


def example_split(X, y, folds=2):
    """
    Splits X and y into # of folds and charts the numerical X on histograms to verify class matching distributions
    """
    skf = StratifiedKFold(n_splits=folds)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        train_vocab, train_class_vocab = build_vocab_word_counts(X_train, y_train)
        X_train_num, X_train_lens = txt_to_total_num(X_train, train_vocab)
        X_test_num, X_test_lens = txt_to_total_num(X_test, train_vocab)

    display(f'total x shape: {X.shape}')
    display(f'total y shape: {y.shape}')
    display(f'train X shape with {folds - 1}/{folds} samples: {X_train_num.shape}')
    display(f'train y shape: {y_train.shape}')
    display(f'train X shape with 1/{folds} samples: {X_test_num.shape}')
    display(f'test y shape: {y_test.shape}')

    fig = plt.figure(figsize=(6, 6))
    grid = plt.GridSpec(2, 2, hspace=0.2, wspace=0.2)
    whole_hist = fig.add_subplot(grid[:, 0])
    train_hist = fig.add_subplot(grid[0, 1])
    test_hist = fig.add_subplot(grid[1, 1])
    whole_hist.hist(y, bins=32)
    whole_hist.set_title('Whole')
    train_hist.hist(y_train, bins=32, color='green')
    train_hist.set_title('Train')
    test_hist.hist(y_test, bins=32, color='maroon')
    test_hist.set_title('Test')
    plt.show

    return


# snipped from hw5
def cache_mount(seed):
    # taken from 623 hw5_solution
    letters = list(string.ascii_lowercase)
    rng = np.random.default_rng(seed=seed)
    if 'google.colab' in str(get_ipython()):
        from google.colab import drive
        drive.mount('/content/drive')
        cache_store = '/content/drive/MyDrive/search_cache_csce623_spring_2021_hw5_' + ''.join(
            rng.choice(letters) for i in range(10))
    else:
        cache_store = 'search_cache_' + ''.join(rng.choice(letters) for i in range(10))
    return cache_store


# snipped and modified from hw5
def model_search(X_train, y_train, model='rfc', seed=42, folds=5):
    cache_store = cache_mount(seed)

    if model == 'rfc':
        pipeline = Pipeline(steps=[
            ('rfc', RandomForestClassifier(random_state=seed))
        ], memory=cache_store)
        param_grid = {
            'rfc__n_estimators': [100, 300, 500],
            'rfc__max_depth': [3, 7, 9, 13, 15, 17],
            'rfc__min_samples_split': [2, 4, 8],
            'rfc__bootstrap': [False],
            'rfc__max_samples': [0.3, 0.6, 0.9, None]
        }
    elif model == 'logreg':
        pipeline = Pipeline(steps=[
            ('logreg', LogisticRegression(random_state=seed))
        ], memory=cache_store)
        param_grid = {
            'logreg__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag'],
            'logreg__max_iter': [100, 250, 500],
            'logreg__multi_class': ['auto']
        }

    elif model == 'mnb':
        pipeline = Pipeline(steps=[
            ('mnb', MultinomialNB())
        ], memory=cache_store)
        param_grid = {
            'mnb__alpha': np.arange(0, 1.01, 0.01)
        }
    else:
        print('No model specified, so no search will initiate.')
        return

    grid_search = HalvingGridSearchCV(pipeline, param_grid, n_jobs=1, verbose=5, cv=folds)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        grid_search.fit(X_train, y_train)

    return grid_search


# snipped from hw5
def display_search_results(grid_search, filename):
    df = pd.DataFrame(columns=['mean_test', '2xstd_test', 'mean_train', '2xstd_train', 'params'])
    means_test = grid_search.cv_results_['mean_test_score']
    stds_test = grid_search.cv_results_['std_test_score']
    means_train = grid_search.cv_results_['mean_train_score']
    stds_train = grid_search.cv_results_['std_train_score']
    for mean_test, std_test, mean_train, std_train, params in zip(means_test, stds_test, means_train, stds_train,
                                                                  grid_search.cv_results_['params']):
        # print("%0.3f (+/-%0.03f) for %r"
        #         % (mean_train, std * 2, params))
        df.loc[len(df.index)] = [mean_test, 2 * std_test, mean_train, 2 * std_train, params]

    best_loc = np.nanargmax(means_test)
    print(
        f'Best: {means_test[best_loc]:.3f} (+/-{stds_test[best_loc] * 2:.3f}) for {grid_search.cv_results_["params"][best_loc]}')

    with pd.option_context("display.max_colwidth", 1000):
        display(df.sort_values(by='mean_test', ascending=False))

    if filename:
        df.to_pickle(filename)

    return df


# snipped from hw5
def show_conf_matrix(y_train, y_train_pred, num_classes=32, zero_diag=False, title='', file='', ylabels=[]):
    labels = np.arange(num_classes)
    conf_mx = confusion_matrix(y_train, y_train_pred)
    if zero_diag:
        np.fill_diagonal(conf_mx, 0)
    row_sums = conf_mx.sum(axis=1, keepdims=True)
    norm_conf_mx = conf_mx / row_sums  # np.max(conf_mx) #row_sums
    figure = plt.figure(figsize=(20, 15))
    axes = figure.add_subplot(111)
    caxes = axes.matshow(norm_conf_mx, cmap=plt.cm.Blues)
    figure.colorbar(caxes)

    axes.set_xticks(np.arange(len(labels)))
    axes.set_yticks(np.arange(len(labels)))
    if ylabels:
        axes.set_yticklabels(ylabels)
    axes.set_xlabel('Predicted Class', fontsize='x-large', fontweight='bold', labelpad=25)
    axes.set_ylabel('True Class', fontsize='x-large', fontweight='bold')

    for (row, col), z in np.ndenumerate(norm_conf_mx):
        axes.text(col, row+0.2, '{:.0%}'.format(z), ha='center', va='center', color=[168 / 255, 74 / 255, 50 / 255],
                  fontsize='large', fontweight='bold')
    for (row, col), z in np.ndenumerate(conf_mx):
        axes.text(col, row-0.2, z, ha='center', va='center', color=[168 / 255, 74 / 255, 50 / 255],
                  fontsize='large', fontweight='bold')
    plt.grid(None)
    plt.title(title)

    if file:
        plt.savefig(file)
    plt.show()


def class_count(y, num_classes):
    counts = np.zeros(num_classes, dtype=np.uint16)
    for i in y:
        counts[i] += 1

    return counts
