import re
import pandas as pd
from gensim.models import Word2Vec

def load_plot_summaries(file_path):
    ids_test = []
    movies_dict = {}

    # Open and read the file
    with open(file_path, 'r', encoding='utf-8') as file:

        # Read each line
        for line in file:
            line = line.strip()
            if '\t' in line:
                movie_id, description = line.split('\t', 1)

                # Store each movie as a dictionary entry
                movies_dict[int(movie_id)] = description
    return movies_dict

def preprocess_text(text, stop_words):
    if pd.isna(text):  
        return ""
    words = re.sub(r'[^a-z\s]+', ' ', text.lower()).split()
    words = [word for word in words if word not in stop_words and word.isalpha() and len(word) > 3]
    return ' '.join(words)


def create_w2v(ddf, start_year, end_year):
    filtered_ddf = ddf[(ddf['year'] >= start_year) & (ddf['year'] <= end_year)]

    filtered_ddf = filtered_ddf.reset_index(drop=True)
    filtered_ddf['combined_text'] = filtered_ddf['title'] + filtered_ddf['excerpt']

    filtered_ddf['processed_text'] = filtered_ddf['combined_text'].apply(preprocess_text, meta=('processed_text', 'str'))

    preprocessed_sentences = filtered_ddf['processed_text'].apply(lambda x: x.split())
    preprocessed_sentences = preprocessed_sentences.compute().tolist()
    model = Word2Vec(sentences=preprocessed_sentences, vector_size=100, window=5, min_count=1, workers=4)
    model.wv.save_word2vec_format("word2vec.bin", binary=True)