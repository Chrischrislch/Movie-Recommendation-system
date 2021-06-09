import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

# Lemmatize: 
#nltk.download('wordnet')
# POS: 
#nltk.download('averaged_perceptron_tagger')
# Stop Words: 
#nltk.download('stopwords')

# read data
data = pd.read_csv('netflix_titles.csv')

data['genre'] = data['listed_in']
data['words'] = (data['title'] + ' ' + data['description'] +' ' + data['genre']).str.lower()
#data = data.head(10)

#indices = pd.Series(data.index, index = data['title']).drop_duplicates()
#print(indices)

# tokenize words
data['tokenized_words'] = data.apply(lambda row: word_tokenize(row['words']), axis=1)

# remove stop words
data['tokenized_words_without_sw'] = data.apply(lambda row: [word for word in row['tokenized_words'] if word.isalpha() and not word in stopwords.words('english')] , axis=1)
data = data[['tokenized_words_without_sw', 'genre']]

# stemming
#porter = PorterStemmer()
#data['stemmed'] = data.apply(lambda row: [porter.stem(word) for word in row['tokenized_words_without_sw']], axis=1)
#data = data[['stemmed', 'genre']] 

# Part-of-Speech
useless_pos_tags = ['CD', 'MD', 'RB', 'RBS ', 'RBR', 'WRB', 'WP']
data['words_pos'] = data.apply(lambda row: [word for (word, tag) in nltk.pos_tag(row['tokenized_words_without_sw']) if tag not in useless_pos_tags], axis=1)

# lemmatization
lemmatizer=WordNetLemmatizer()
data['lemmatized'] = data.apply(lambda row: [lemmatizer.lemmatize(word) for word in row['words_pos']], axis=1)
data = data[['lemmatized']]

# process genre
# data['tokenized_genre'] = data['genre'].str.lower().str.split(', ')

# data = data[['tokenized_genre','lemmatized']]
# # save the data
# # print(data)
# data['genre_description'] = data['tokenized_genre'] + data['lemmatized']
# data = data['genre_description']


# ------------- Word2Vec ---------------
from gensim.models import word2vec
import gensim
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# download text8 from (http://mattmahoney.net/dc/text8.zip)
# sentences = word2vec.Text8Corpus('text8')
# training
# model = word2vec.Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
# model = word2vec.word2vec.load()
# model.wv['man']

# Use pre-trained model
import gensim.downloader
import numpy as np
glove_vectors = gensim.downloader.load('glove-twitter-200')
# store the embeddings in a np array
embs = np.zeros([len(data),glove_vectors.vector_size])
unknown = []
for i in range(len(data)):
    row = (data.iloc[i]).tolist()[0]
    cnt = 0
    for word in row:
        try:
            embs[i] += glove_vectors[word]            
            cnt += 1
        except:
            unknown.append(word)   
            pass
    embs[i] /= cnt

#print(embs)

# One hot for director and cast

#start from here
#Cosine similarity matrix using linear_kernel function
import scipy.spatial
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
cosine_sim = linear_kernel(embs, embs)
print(cosine_sim)

#index mapping of the orginal title
data1 = pd.read_csv('netflix_titles.csv')
indices = pd.Series(data1.index, index = data1['title']).drop_duplicates()
#print(indices)
euclidean = scipy.spatial.distance.cdist(embs, embs, metric='euclidean')
#jaccard = scipy.spatial.distance.cdist(embs, embs,  metric='jaccard')
print(euclidean)
#print(jaccard)


#recommendations function

def recommend(title, cos=cosine_sim):
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cos[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 20 most similar movies with sid
    sim_scores = sim_scores[1:21]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 20 most similar movies
    return data1['title'].iloc[movie_indices]

'''#better results

def recommend(title, cos=euclidean):
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cos[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 20 most similar movies with sid
    sim_scores = sim_scores[1:21]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 20 most similar movies
    return data1['title'].iloc[movie_indices]

'''

print(recommend('A Good Wife'))






