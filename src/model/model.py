# We define here the usefull functions to create the model and use it

import pandas as pd
import re
from collections import defaultdict
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import numpy as np
import gensim
import gensim.downloader as api
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer


def get_data():
    """
    Load the data from the csv file
    """
    data = pd.read_csv('../data/MovieSummaries/plot_summaries_cleaned.csv')
    meta_data = pd.read_csv('../data/MovieSummaries/movie.metadata.tsv', sep='\t', header=None)
    meta_data.columns = ["movie_id",1,"movie_name",3,4,5,6,7,"genre"]

    valid_words = set()
    with open('../data/words.txt') as word_file:
        valid_words = set(word_file.read().split())

    return data, meta_data, valid_words

def save_data(data):
    """
    Save the data to a csv file
    """
    data.to_csv('../data/MovieSummaries/plot_summaries_cleaned_fit_model.csv', index=False)

def get_model():
    model = model = gensim.models.KeyedVectors.load_word2vec_format('../data/w2v/6/model.txt', binary=False)
    return model


def w2v_preprocessing(data, valid_words, model):
    """
    Preprocess the data for the word2vec model
    """
    def filter_words_not_in_model_helper(word): # We explain a bit later why me choose to filter out words not in the model
        if word in model and word in valid_words:
            return word
        else:
            return " "
    def filter_words_not_in_model(s):
        return " ".join([filter_words_not_in_model_helper(word) for word in s.split()])

    def filter_basic_patterns(s):
        pattern = "|".join([
        "\\d+", # Matches digits.
        r'http?://\S+|www\.\S+', # Matches url links
        ",", "\.", ":", "\(", "\)","_", "\{", "\}", "\?", "!", "&", "/", "\[", "\]", "\|", "#", "%", "\"", "\'", ";", "-", '®', 'à', '>', '<', '=', 'ü', "\*"
        ])
        # Cast uppercase letters to lowercase
        s = s.lower()
        s = re.sub(pattern, " ", s) # replace by spaces to avoid a:b or a,b becoming ab instead of a b. 
        return s
    
    def preprocess(s):
        s = filter_basic_patterns(s)
        s = filter_words_not_in_model(s)
        s = re.sub(r"\s+", " ", s).strip() # removing uncessary spaces
    
        return s
    
    data["Plot cleaned"] = data["Plot_summary"].apply(preprocess)
    words2vec = set()
    words_not_in_model = set()

    for description in data["Plot cleaned"]:
        for word in description.split():
            if word in model and word in valid_words:
                words2vec.add(word)
            else :
                words_not_in_model.add(word)
    print("Number of words in the model: ", len(words2vec))
    print("Number of words not in the model (should be 0 now): ", len(words_not_in_model))

    return data, words2vec

def get_vectorized_data(data):
    """
    Get the vectorizer to transform the data
    """
    vectorizer = CountVectorizer()
    X_dense = vectorizer.fit_transform(data["Plot cleaned"])

    X_sparse = X_dense.tocsr()

    # Construct maps
    word_to_index = {word: i for i, word in enumerate(vectorizer.get_feature_names_out())}
    movie_id_to_index = {movie_id: i for i, movie_id in enumerate(data['Wikipedia_movie_ID'])}
    
    return X_sparse, word_to_index, movie_id_to_index

def compute_tf_idf(X_sparse):
    """
    Compute the tf-idf matrix
    """
    # Compute the term frequency
    tf = csr_matrix(X_sparse/X_sparse.sum(axis=1))
    # Compute the inverse document frequency
    idf = csr_matrix(np.log(X_sparse.shape[0]/(X_sparse>0).sum(axis=0)).flatten())
    # Compute the tf-idf matrix
    tf_idf = tf.multiply(idf)
    return tf_idf

def compute_movie_vectors(tf_idf, model, word_to_index, movie_id_to_index, data):
    """
    Compute the movie vectors
    """
    
    def compute_vector_movie(document, movieID):
        d_words = document.split()
        weighted_vec = np.zeros((model.vector_size,))
        count = 0
  
        if len(d_words) == 0:
            return np.zeros((model.vector_size,))
        else:
            for word in d_words:
                if word in word_to_index.keys():
                    weighted_vec += tf_idf[movie_id_to_index[movieID], word_to_index[word]] * model[word]
                    count += 1
                else:
                    #print(f"Word {word} not in the word_to_index")``
                    continue
        return weighted_vec / count
    
    movie_vectors = np.zeros((tf_idf.shape[0], model.vector_size))
    
    for i, movieID in enumerate(data['Wikipedia_movie_ID']):
        movie_vectors[i] = compute_vector_movie(data["Plot cleaned"][i], movieID)
    return movie_vectors



