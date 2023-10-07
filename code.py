import pandas as pd
import numpy as np
import ast
# functions:
def convert(obj):
        L=[]
        for i in ast.literal_eval(obj):
            L.append(i['name'])
        return L
def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if i != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L
def fetch_director(obj):
    l=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            l.append(i['name'])
            break
    return l
# importing datasets
credits=pd.read_csv("tmdb_5000_credits.csv")
movies=pd.read_csv("tmdb_5000_movies.csv")
# print(credits.info())
# print(movies.info())
# merging and selecting dataset
movies=movies.merge(credits,on='title')
movies=movies[['title','overview','genres','keywords','cast','crew']]
movies.isnull().sum() #to know nan values in column or not
movies.dropna(inplace=True)
# fetching important data
movies['genres']=movies['genres'].apply(convert)
movies['keywords']=movies['keywords'].apply(convert) # apply function Invoke function in parathesis on values of Series
movies['cast']=movies['cast'].apply(convert3)
movies['crew']=movies['crew'].apply(fetch_director)
movies['overview'] = movies['overview'].apply(lambda x: x.split(' '))
# Removing space
movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])
# Making a column deleting others
movies['tags']=movies['genres']+movies['keywords']+movies['cast']+movies['crew']
new = movies.drop(columns=['overview','genres','keywords','cast','crew'])
# processing on new dataset
new['tags']=new['tags'].apply(lambda x:" ".join(x))
new['tags']=new['tags'].apply(lambda x:x.lower())
import nltk
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
def stim(text):
    L = []
    for i in text.split():

        L.append(ps.stem(i))

    return " ".join(L)
new['tags']=new['tags'].apply(stim)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,stop_words='english')
vectors=cv.fit_transform(new['tags']).toarray()
from sklearn.metrics.pairwise import cosine_similarity
similarity=cosine_similarity(vectors)
def recommender(movie):
    movie_index=new[new['title']==movie].index[0]
    distance=similarity[movie_index]
    movies_list=sorted(list(enumerate(distance)),reverse=True,key=lambda x:x[1])[1:6]
    for i in movies_list:
        print(new.iloc[i[0]].title)
recommender('Batman Begins')