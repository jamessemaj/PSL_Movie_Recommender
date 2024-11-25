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
    """
    Returns the movies to be displayed for rating (top 100 popular movies).
    """
    return movies

def get_recommended_movies(new_user_ratings):
    """
    Generates movie recommendations based on the user's ratings.

    Parameters:
    - new_user_ratings: dict {movie_id: rating}

    Returns:
    - recommended_movies: DataFrame of recommended movies
    """
    # Convert to pandas Series
    newuser = pd.Series(data=new_user_ratings)
    # Ensure newuser is aligned with the movies
    newuser = newuser.reindex(top_100_movie_ids)
    # Call myIBCF
    recommendations = myIBCF(newuser)
    # Get movie details
    recommended_movies = movies.loc[recommendations]
    return recommended_movies

def get_popular_movies(genre: str):
    """
    Returns the top 10 popular movies in the specified genre.

    Parameters:
    - genre: str

    Returns:
    - top_10_movies: DataFrame of top 10 movies in the genre
    """
    # Filter movies in the specified genre
    movies_in_genre = movies[movies['genres'].str.contains(genre)]
    if movies_in_genre.empty:
        return pd.DataFrame()
    # Get the movie IDs
    movie_ids_in_genre = movies_in_genre.index.tolist()
    # Get the popular movies among them
    popular_movies_in_genre = movie_popularity.loc[movie_ids_in_genre]
    # Sort the movies by popularity
    popular_movies_in_genre = popular_movies_in_genre.sort_values(ascending=False)
    # Get the top 10 movies
    top_10_movie_ids = popular_movies_in_genre.index[:10]
    # Get the movie details
    top_10_movies = movies.loc[top_10_movie_ids]
    return top_10_movies

def myIBCF(newuser):
    """
    Recommends top 10 movies to the new user based on Item-Based Collaborative Filtering.

    Parameters:
    - newuser: pandas Series, with movie IDs as index and ratings as values.
               Non-rated movies should have NaN.

    Returns:
    - recommendations: List of recommended movie IDs
    """
    # Ensure newuser is aligned with S_top_k's columns
    newuser = newuser.reindex(S_top_k.columns)
    
    # Identify movies not rated by the user
    unrated_movies = newuser[newuser.isna()].index
    
    # Identify movies rated by the user
    rated_movies = newuser[newuser.notna()].index
    rated_ratings = newuser.loc[rated_movies]
    
    # Compute predictions for unrated movies
    predictions = pd.Series(index=unrated_movies, dtype=np.float64)
    
    # For each unrated movie i
    for i in unrated_movies:
        # Get similarities Sij for movie i with all other movies
        S_i = S_top_k.loc[i]
        
        # Movies similar to i (Sij is not NaN)
        similar_movies = S_i[S_i.notna()].index
        
        # Find intersection with movies rated by the user
        common_movies = similar_movies.intersection(rated_movies)
        
        # If there are no common movies, skip prediction for this movie
        if len(common_movies) == 0:
            continue
        
        # Get the similarities and user ratings for these movies
        Sij = S_i.loc[common_movies]
        wj = rated_ratings.loc[common_movies]
        
        # Compute numerator and denominator
        numerator = np.dot(Sij.values, wj.values)
        denominator = Sij.sum()
        
        if denominator == 0:
            continue  # Avoid division by zero
        
        # Compute predicted rating for movie i
        predictions.loc[i] = numerator / denominator
    
    # Recommend top 10 movies based on predictions
    predictions_sorted = predictions.dropna().sort_values(ascending=False)
    recommendations = predictions_sorted.index.tolist()
    
    # If fewer than 10 recommendations, fill with popular movies
    if len(recommendations) < 10:
        # Exclude movies already rated by the user and already recommended
        excluded_movies = set(rated_movies).union(set(recommendations))
        # Get movies not in excluded_movies
        additional_movies = [
            movie for movie in movie_popularity.index if movie not in excluded_movies
        ]
        
        # Add movies to recommendations until we have 10
        for movie in additional_movies:
            recommendations.append(movie)
            if len(recommendations) >= 10:
                break
    
    # Return the top 10 recommendations
    return recommendations[:10]
