from flask import Flask, jsonify
from flask import request
import pandas as pd
from flask import Flask, render_template,Request,Response,request,jsonify
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from flask_cors import CORS, cross_origin
import requests
import json
from tmdbv3api import TMDb
import pickle as pkl
app = Flask(__name__)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


import numpy as np
import random
import bs4
import re
import operator
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import base64
import io
from matplotlib.pyplot import imread
import codecs
import numpy as np
from sklearn.metrics import pairwise_distances
from IPython.display import HTML
tmdb=TMDb()
tmdb.api_key='8b5da40bcd2b5fa4afe55c468001ad8a'
from  tmdbv3api import Movie
tmdb_movie=Movie()
df2=pd.read_csv("tmdb_5000_credits.csv")
knn1=pd.read_csv("tmdb_5000_movies.csv")

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
vectorizer=pkl.load(open('vectorizerer.pkl', 'rb'))
clt=pkl.load(open('nlp_model.pkl', 'rb'))

url = [
    "https://api.themoviedb.org/3/discover/movie?api_key=2c5341f7625493017933e27e81b1425e&primary_release_year=2015&adult=false",
    "http://api.themoviedb.org/3/discover/movie?api_key=2c5341f7625493017933e27e81b1425e&primary_release_year=2014&adult=false",
    "https://api.themoviedb.org/3/movie/popular?api_key=2c5341f7625493017933e27e81b1425e&language=en-US&page=1&adult=false",
    "https://api.themoviedb.org/3/movie/popular?api_key=2c5341f7625493017933e27e81b1425e&language=en-US&page=2&adult=false",
    "https://api.themoviedb.org/3/movie/popular?api_key=2c5341f7625493017933e27e81b1425e&language=en-US&page=3&adult=false",
    "https://api.themoviedb.org/3/discover/movie?api_key=2c5341f7625493017933e27e81b1425e&with_genres=18&adult=false",
    "http://api.themoviedb.org/3/discover/movie?api_key=2c5341f7625493017933e27e81b1425e&primary_release_year=2020&adult=false",
    "http://api.themoviedb.org/3/discover/movie?api_key=2c5341f7625493017933e27e81b1425e&primary_release_year=2019&adult=false",
    "http://api.themoviedb.org/3/discover/movie?api_key=2c5341f7625493017933e27e81b1425e&primary_release_year=2017&adult=false",
    "http://api.themoviedb.org/3/discover/movie?api_key=2c5341f7625493017933e27e81b1425e&primary_release_year=2016&adult=false",
    "https://api.themoviedb.org/3/discover/movie?api_key=2c5341f7625493017933e27e81b1425e&with_genres=27",
    "https://api.themoviedb.org/3/discover/movie?api_key=2c5341f7625493017933e27e81b1425e&with_genres=16"
  ]
  
def get_news():
    response = requests.get("https://www.imdb.com/news/top/?ref_=hm_nw_sm")
    soup = bs4.BeautifulSoup(response.text, 'html.parser')
    data = [re.sub('[\n()]', "", d.text) for d in soup.find_all('div', class_='news-article__content')]
    image = [m['src'] for m in soup.find_all("img", class_="news-article__image")]
    t_data = []
    for i in range(len(data)):
        t_data.append([image[i], data[i][1:len(data[i])-1]])
    return t_data

def getdirector(x):
    data = []
    result = tmdb_movie.search(x)
    movie_id = result[0].id
    response = requests.get(
        "https://api.themoviedb.org/3/movie/{}/credits?api_key=2c5341f7625493017933e27e81b1425e".format(
            movie_id))
    data_json = response.json()
    data.append(data_json)
    crew=data[0]['crew']
    director=[]
    for c in crew:
        if c['job']=='Director':
            director.append(c['name'])
            break
    return director

def get_swipe():
    data=[]
    val=random.choice(url)
    for i in range(5):
        lis=[]
        response = requests.get(
            val+"&page="+str(i+1))
        data_json = response.json()
        lis.append(data_json["results"])
        for i in lis[0]:
            data.append(i)
    return data

def getreview(x):
    data=[]
    result=tmdb_movie.search(x)
    movie_id=result[0].id
    response=requests.get("https://api.themoviedb.org/3/movie/{}/reviews?api_key=2c5341f7625493017933e27e81b1425e&language=en-US&page=1".format(movie_id))
    data_json=response.json()
    data.append(data_json)
    return data

def getrating(title):
    movie_review = []
    data=getreview(title)
    for i in data[0]['results']:
        pred=clt.predict(vectorizer.transform([i['content']]))
        if pred[0]=='positive':
            movie_review.append({
                "review":i['content'],
                "rating":"Good"
            })
        else:
            movie_review.append({
                "review": i['content'],
                "rating": "Bad"
            })
    return movie_review

def get_data(x):
    data=[]
    result=tmdb_movie.search(x)
    movie_id=result[0].id
    response=requests.get("https://api.themoviedb.org/3/movie/{}?api_key=2c5341f7625493017933e27e81b1425e".format(movie_id))
    response2=requests.get("https://api.themoviedb.org/3/movie/{}/credits?api_key=2c5341f7625493017933e27e81b1425e".format(movie_id))
    response3=requests.get("https://api.themoviedb.org/3/movie/{}/keywords?api_key=2c5341f7625493017933e27e81b1425e".format(movie_id))
    data_json=response.json()
    data_json2=response2.json()
    data_json3=response3.json()
    data.append(data_json)
    data.append(data_json2)
    data.append(data_json3)
    return data

def getcomb(movie_data):
    cast_data=movie_data[1]['cast']
    cast=[]
    for data in cast_data:
        cast.append(data['name'])
    crew=movie_data[1]['crew']
    director=[]
    for c in crew:
        if c['job']=='Director':
            director.append(c['name'])
            break
    genres=[]
    for x in movie_data[0]['genres']:
        genres.append(x['name'])
    keywords=[]
    for k in movie_data[2]['keywords']:
        keywords.append(k['name'])
    d=str(cast)+str(keywords)+str(genres)+director[0]+str(movie_data[0]['overview'])
    return d
    
def get_data2(x):
    data=[]
    result=tmdb_movie.search(x)
    movie_id=result[0].id
    trailer=requests.get("https://api.themoviedb.org/3/movie/{}/videos?api_key=2c5341f7625493017933e27e81b1425e&language=en-US".format(movie_id))
    response=requests.get("https://api.themoviedb.org/3/movie/{}?api_key=2c5341f7625493017933e27e81b1425e".format(movie_id))
    data_json = response.json()
    trailer=trailer.json()
    data.append(data_json)
    data.append(trailer)
    return data

# FLASK
app = Flask(__name__)
cors = CORS(app)
@app.route('/')
def index():
   return render_template("index.html")
@app.route('/getname',methods=["GET"])
def getnames():
   data=[]
   for i in df2["title_x"]:
       data.append(i)
   return jsonify(data)
@app.route('/getmovie/<movie_name>',methods=["GET"])
def getmovie(movie_name):
   data=get_data2(movie_name)
   return jsonify(data)

@app.route('/getreview/<movie_name>', methods=["GET"])
def getreviews(movie_name):
    data=getrating(movie_name)
    return jsonify(data)
@app.route('/getdirector/<movie_name>', methods=["GET"])
def getdirectorname(movie_name):
    data=getdirector(movie_name)
    return jsonify(data)
@app.route('/getswipe', methods=["GET"])
def getswipe():
    data=get_swipe()
    return jsonify(data)
@app.route('/getnews', methods=["GET"])
def getnewsdata():
    data=get_news()
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True,port=5000)

# CONTENT BASED RECOMMENDATIONS
def get_recommendations(title, user_id):
    movies_data = pd.read_csv('Main_data.csv')
    ratings_data = pd.read_csv('movie_rating.csv')

    def content_based_recommendations(title, movies_data, top_n=6):
        movies_data['comb'] = movies_data['title_x'] + movies_data['genres']
        flag = movies_data['title_x'].eq(title).any()
        if not flag:
            new_row = {'title_x': title, 'genres': ''}
            movies_data = movies_data.append(new_row, ignore_index=True)
        movies_data = movies_data.replace(pd.NA, '')
        tfidf = TfidfVectorizer(stop_words='english')
        count_matrix = tfidf.fit_transform(movies_data['comb'])
        cosine_sim = cosine_similarity(count_matrix, count_matrix)
        idx = movies_data[movies_data['title_x'] == title].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n + 1]
        movie_indices = [i[0] for i in sim_scores]
        return movies_data['title_x'].iloc[movie_indices]

# COLLABORATIVE BASED RECOMMENDATIONS

    def collaborative_filtering_recommendations(user_id, top_n=10):
        merged_data = pd.merge(movies_data, ratings_data, left_on='id', right_on='movieId')
        user_item_matrix = pd.pivot_table(data=merged_data, values='rating', index='userId', columns='movieId', fill_value=0)
        if user_id not in user_item_matrix.index:
            print("User ID doesn't exist.")
            return None
        item_similarity = pairwise_distances(user_item_matrix.T, metric='cosine')
        min_rating = user_item_matrix.min().min()
        max_rating = user_item_matrix.max().max()
        normalized_ratings = (user_item_matrix - min_rating) / (max_rating - min_rating)
        user_ratings = normalized_ratings.loc[user_id].values.reshape(1, -1)
        predicted_ratings = np.dot(user_ratings, item_similarity) / np.sum(item_similarity)
        predicted_ratings = predicted_ratings * (max_rating - min_rating) + min_rating
        top_movies_indices = np.argsort(-predicted_ratings)[0][:top_n]
        top_movies = movies_data[movies_data['id'].isin(top_movies_indices)]['title_x']
        return top_movies

#HYBRID BASED RECOMMENDATIONS
    def hybrid_recommendations(user_id, movie_title):
        content_based_recs = content_based_recommendations(movie_title, movies_data)
        collaborative_recs = collaborative_filtering_recommendations(user_id)
        hybrid_recs = pd.concat([content_based_recs, collaborative_recs])
        return hybrid_recs
    return hybrid_recommendations(user_id, title)

#METHOD TO  API CALL ON FRONTEND TO ADD USER ID ALSO
@app.route('/send/<movie_name>/<string:userId>', methods=["GET"])
def get(movie_name, userId):
    if request.method=="GET":
        val = get_recommendations(movie_name, userId)
        if val is None:
            return jsonify({"message":"movie not found in database"})
        val = list(val)
        result=[]
        try:
            for i in val:
                res=get_data2(i)
                result.append(res[0])
        except request.ConnectionError:
            return jsonify({"movie not found in database"})
        return jsonify(result)


#API CALL IN FRONTEND IN FORMAT /rate/84/1.5/YQ67YHFYTYUHF67686TR7T7T7868
@app.route('/rate/<movieId>/<float:rate>/<string:userId>', methods=["GET"])
def rate_movie(movieId, rate, userId):
    import csv
    data = {'userId': userId, 'movieId': movieId, 'rating': rate}
    filename = 'movie_rating.csv'
    with open(filename, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['userId', 'movieId', 'rating'])
        writer.writerow(data)
    return jsonify({'userId': userId, 'movieId': movieId, 'rating': rate})

