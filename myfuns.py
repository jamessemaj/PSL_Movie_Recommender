import pandas as pd
import requests
import numpy as np

# Load movies data
myurl = "https://liangfgithub.github.io/MovieData/movies.dat?raw=true"

response = requests.get(myurl)
movie_lines = response.text.strip().split('\n')
movie_data = [line.split("::") for line in movie_lines if line]

movies = pd.DataFrame(movie_data, columns=['movie_id', 'title', 'genres'])
movies['movie_id'] = movies['movie_id'].astype(int)
movies['movie_id'] = 'm' + movies['movie_id'].astype(str)
movies.set_index('movie_id', inplace=True)

genres = list(
    sorted(set([genre for genres in movies.genres.unique() for genre in genres.split("|")]))
)

# Load similarity matrix
S_top_k = pd.read_csv('S_top_k.csv', index_col=0)
S_top_k.index = S_top_k.index.astype(str)
S_top_k.columns = S_top_k.columns.astype(str)

# Load movie popularity
movie_popularity = pd.read_csv('movie_popularity.csv', index_col=0).squeeze("columns")
movie_popularity.index = movie_popularity.index.astype(str)

# Get top 100 movies
top_100_movie_ids = movie_popularity.index.tolist()[:100]
movies = movies.loc[top_100_movie_ids]
S_top_k = S_top_k.loc[top_100_movie_ids, top_100_movie_ids]
movie_popularity = movie_popularity.loc[top_100_movie_ids]

def get_displayed_movies():
    return movies

def get_recommended_movies(new_user_ratings):
 
    newuser = pd.Series(data=new_user_ratings)   
    newuser = newuser.reindex(top_100_movie_ids)
    recommendations = myIBCF(newuser)
    recommended_movies = movies.loc[recommendations]
    return recommended_movies

def get_popular_movies(genre: str):

    movies_in_genre = movies[movies['genres'].str.contains(genre)]
    if movies_in_genre.empty:
        return pd.DataFrame()
   
    movie_ids_in_genre = movies_in_genre.index.tolist()

    popular_movies_in_genre = movie_popularity.loc[movie_ids_in_genre]
  
    popular_movies_in_genre = popular_movies_in_genre.sort_values(ascending=False)
   
    top_10_movie_ids = popular_movies_in_genre.index[:10]

    top_10_movies = movies.loc[top_10_movie_ids]
    return top_10_movies

def myIBCF(newuser):
   
    newuser = newuser.reindex(S_top_k.columns)
    
    unrated_movies = newuser[newuser.isna()].index
    
    rated_movies = newuser[newuser.notna()].index
    rated_ratings = newuser.loc[rated_movies]
    
    predictions = pd.Series(index=unrated_movies, dtype=np.float64)
    
    for i in unrated_movies:
        S_i = S_top_k.loc[i]
        
        similar_movies = S_i[S_i.notna()].index
        
        common_movies = similar_movies.intersection(rated_movies)
        
        if len(common_movies) == 0:
            continue
        
       
        Sij = S_i.loc[common_movies]
        wj = rated_ratings.loc[common_movies]
        
   
        numerator = np.dot(Sij.values, wj.values)
        denominator = Sij.sum()
        
        if denominator == 0:
            continue  
        
      
        predictions.loc[i] = numerator / denominator
    
  
    predictions_sorted = predictions.dropna().sort_values(ascending=False)
    recommendations = predictions_sorted.index.tolist()
    
   
    if len(recommendations) < 10:
      
        excluded_movies = set(rated_movies).union(set(recommendations))
       
        additional_movies = [
            movie for movie in movie_popularity.index if movie not in excluded_movies
        ]
        
        for movie in additional_movies:
            recommendations.append(movie)
            if len(recommendations) >= 10:
                break
                
    return recommendations[:10]
