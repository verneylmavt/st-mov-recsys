import os
import io
import requests
import streamlit as st
import numpy as np
import pandas as pd
import pyarrow as pa
import onnxruntime as ort
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------
# Model Information
# ----------------------
model_info = {
    "ncf": {
        "subheader": "Model: Neural Collaborative Filtering (NCF)",
        "pre_processing": """
Dataset = MovieLens 32M Dataset
        """,
        "parameters": """
Batch Size = 1024
User Size = 138,493
Item Size = 27,278
Number of MLP (Multi-Layer Perceptron) Layers = 2 (128+64)
Dropout Rate = 0.2
Learning Rate = 0.001
Epochs = 20
Optimizer = AdamW
Weight Decay = 0.01
Loss Function = MSELoss
        """,
        "model_code": """
class Model(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=50, hidden_layers=[128, 64], dropout=0.2):
        super(Model, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        layers = []
        input_size = embedding_dim * 2
        for hidden in hidden_layers:
            layers.append(nn.Linear(input_size, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_size = hidden
        self.mlp = nn.Sequential(*layers)
        self.output_layer = nn.Linear(input_size, 1)
    def forward(self, user, item):
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        x = torch.cat([user_emb, item_emb], dim=-1)
        x = self.mlp(x)
        rating = self.output_layer(x)
        return rating.squeeze()
        """
    }
}

# ----------------------
# Loading Function
# ----------------------

@st.cache_resource
def load_model(model_name):
    try:
        model_path = os.path.join("models", str(model_name), "model-q.onnx")
        ort_session = ort.InferenceSession(model_path)
    except FileNotFoundError:
        st.error(f"Model file not found for {model_name}. Please ensure 'model-q.onnx' exists in the model directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the model for {model_name}: {e}")
        st.stop()
    return ort_session

@st.cache_data
def load_movies(model_name):
    try:
        model_path = os.path.join("models", model_name, "movies.csv")
        movies = pd.read_csv(model_path)
        return movies
    except FileNotFoundError:
        st.error(f"Movies file not found for {model_name}. Please ensure 'movies.csv' exists in the model directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the movies for {model_name}: {e}")
        st.stop()
        
@st.cache_data
def load_ratings(model_name):
    try:
        model_path = f"https://drive.google.com/uc?id=1YAAL02PJ82kBEiMO87okSpHpUiCzIKe3"
        response = requests.get(model_path)
        response.raise_for_status()
        data = io.BytesIO(response.content)
        ratings = pd.read_parquet(data)
        return ratings
    except FileNotFoundError:
        st.error(f"Ratings file not found for {model_name}. Please ensure 'ratings.parquet' exists in the model directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the ratings for {model_name}: {e}")
        st.stop()

@st.cache_data
def load_embeddings(model_name):
    try:
        model_path = os.path.join("models", model_name, "embeddings.npy")
        ratings = np.load(model_path)
        return ratings
    except FileNotFoundError:
        st.error(f"embeddings.npy file not found for {model_name}. Please ensure 'embeddings.npy' exists in the model directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the embeddings for {model_name}: {e}")
        st.stop()
        
@st.cache_data
def load_genres():
    return ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'IMAX', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']


# ----------------------
# Prediction Function
# ----------------------

def recommend_similar_movies_name(movie_name, movies, movie_encoder, embeddings, top_n):
    movie_name = movie_name.strip()
    matching_movies = movies[movies['title'].str.contains(movie_name, case=False)]
    if matching_movies.empty:
        st.error("Movie Name Invalid")
        return None
    movie_id = matching_movies.iloc[0]['movieId']
    
    movie_encoded = movie_encoder.transform([movie_id])[0]
    target_embedding = embeddings[movie_encoded].reshape(1, -1)
    
    similarities = cosine_similarity(target_embedding, embeddings).flatten()
    similar_indices = similarities.argsort()[-(top_n + 1):-1][::-1]
    similar_movie_ids = movie_encoder.inverse_transform(similar_indices)
    
    recommendations = movies[movies['movieId'].isin(similar_movie_ids)][['movieId', 'title']]
    recommendations['similarity'] = similarities[similar_indices]
    recommendations["similarity"] = np.ceil(recommendations["similarity"] * 1000) / 100
    
    recommendations = recommendations.reset_index(drop=True)
    recommendations.index += 1
    
    recommendations.rename(columns={
    "movieId": "Movie ID",
    "title": "Title",
    "similarity": "Cosine Similarity"
    }, inplace=True)
    recommendations["Year"] = recommendations["Title"].str.extract(r"\((\d{4})\)")
    recommendations["Title"] = recommendations["Title"].str.replace(r" \(\d{4}\)", "", regex=True)
    recommendations = recommendations[["Movie ID", "Title", "Year", "Cosine Similarity"]]
    
    return recommendations


def recommend_by_genre_pop(genres, all_genres, movies, ratings, top_n):
    # valid_genres = sorted(all_genres)
    # for genre in genres:
    #     if genre not in valid_genres:
    #         print(f"Genre '{genre}' Invalid. Valid Genres: {valid_genres}")
    #         return pd.DataFrame()
    filtered_movies = movies
    
    for genre in genres:
        filtered_movies = filtered_movies[filtered_movies[genre] == 1]
    if filtered_movies.empty:
        st.error("No Movies Found w/ Specified Genre(s)")
        return None
    
    popularity = ratings.groupby('movieId').size().reset_index(name='rating_count')
    recommendations = filtered_movies.merge(popularity, on='movieId', how='left').fillna({'rating_count': 0})
    recommendations = recommendations.sort_values(by='rating_count', ascending=False)
    recommendations = recommendations[['movieId', 'title', 'rating_count']].head(top_n)
    recommendations["rating_count"] = recommendations["rating_count"].astype(int)
    recommendations = recommendations.reset_index(drop=True)
    recommendations.index += 1
    
    recommendations.rename(columns={
    "movieId": "Movie ID",
    "title": "Title",
    "rating_count": "Total Rating"
    }, inplace=True)
    recommendations["Year"] = recommendations["Title"].str.extract(r"\((\d{4})\)")
    recommendations["Title"] = recommendations["Title"].str.replace(r" \(\d{4}\)", "", regex=True)
    recommendations = recommendations[["Movie ID", "Title", "Year", "Total Rating"]]
    
    return recommendations



def recommend_combined_mix(movie_name, genres, all_genres, movies, movie_encoder, embeddings, ratings, top_n=10):
    similar_movies = recommend_similar_movies_name(movie_name, movies, movie_encoder, embeddings, top_n=top_n*2)
    if similar_movies.empty:
        return pd.DataFrame()
    
    similar_movie_ids = similar_movies["Movie ID"].astype(int).values
    
    filtered_movies = movies[movies['movieId'].isin(similar_movie_ids)].copy()
    # for genre in genres:
    #     if genre in all_genres:
    #         filtered_movies = filtered_movies[filtered_movies[genre] == 1]
    #     else:
    #         print(f"Genre '{genre}' Invalid.")
    #         return pd.DataFrame() 
    
    if filtered_movies.empty:
        st.error("No Movies Found w/ Specified Genre(s)")
        return None
    
    popularity = ratings.groupby('movieId').size().reset_index(name='rating_count')
    filtered_movies = filtered_movies.merge(popularity, on='movieId', how='left').fillna({'rating_count': 0})
    
    filtered_movies = filtered_movies.merge(similar_movies[['Movie ID', 'Cosine Similarity']],
                                            left_on='movieId', right_on='Movie ID', how='left')
    
    filtered_movies = filtered_movies.sort_values(by=['Cosine Similarity', 'rating_count'], ascending=[False, False])
    
    recommendations = filtered_movies[['movieId', 'title', 'Cosine Similarity', 'rating_count']].head(top_n)
    
    recommendations.rename(columns={
        'movieId': 'Movie ID',
        'title': 'Title',
        'rating_count': 'Total Rating'
    }, inplace=True)
    recommendations["Year"] = recommendations["Title"].str.extract(r"\((\d{4})\)")
    recommendations["Title"] = recommendations["Title"].str.replace(r" \(\d{4}\)", "", regex=True)
    recommendations = recommendations[["Movie ID", "Title", "Year", "Cosine Similarity", "Total Rating"]]
    recommendations = recommendations.reset_index(drop=True)
    recommendations.index += 1
    return recommendations

# ----------------------
# Page UI
# ----------------------
def main():
    st.title("Movie Recommender System")
    
    model_names = list(model_info.keys())
    model = st.selectbox("Select a Model", model_names)
    st.divider()
    
    # net = load_model(model)
    movies = load_movies(model)
    ratings = load_ratings(model)
    embeddings = load_embeddings(model)
    all_genres = load_genres()
    
    user_encoder = LabelEncoder()
    user_encoder.fit_transform(ratings['userId'])
    movie_encoder = LabelEncoder()
    movie_encoder.fit_transform(ratings['movieId'])
    
    st.subheader(model_info[model]["subheader"])
    
    # movie_name = st.text_input("Preferred Movie:")
    # selected_genres = st.multiselect(
    # "Preferred Genres",
    # all_genres
    # )
    # top_n = st.slider("Number of Recommendations", 0, 50, 10)

    # if st.button("Recommend"):
    #     if movie_name or selected_genres:
    #         with st.spinner('Recommending...'):
    #             if movie_name and not selected_genres:
    #                 recommendations = recommend_similar_movies_name(movie_name, movies, movie_encoder, embeddings, top_n)
    #             elif not movie_name and selected_genres:
    #                 recommendations = recommend_by_genre_pop(selected_genres, all_genres, movies, ratings, top_n)
    #             elif movie_name and selected_genres:
    #                 recommendations = recommend_combined_mix(movie_name, selected_genres, all_genres, movies, movie_encoder, embeddings, ratings, top_n)
    #             st.dataframe(recommendations)
    #     else:
    #         st.warning("Please enter either a preferred movie in the input box or select some genres.")
    
    with st.form(key="recommendation_form"):
        movie_name = st.text_input("Preferred Movie:")
        selected_genres = st.multiselect(
            "Preferred Genres",
            all_genres
        )
        top_n = st.slider("Number of Recommendations", 0, 50, 10)
        submit_button = st.form_submit_button(label="Recommend")
        
        if submit_button:
            if movie_name or selected_genres:
                with st.spinner('Recommending...'):
                    if movie_name and not selected_genres:
                        recommendations = recommend_similar_movies_name(movie_name, movies, movie_encoder, embeddings, top_n)
                    elif not movie_name and selected_genres:
                        recommendations = recommend_by_genre_pop(selected_genres, all_genres, movies, ratings, top_n)
                    elif movie_name and selected_genres:
                        recommendations = recommend_combined_mix(movie_name, selected_genres, all_genres, movies, movie_encoder, embeddings, ratings, top_n)
                    if recommendations is not None:
                        st.dataframe(recommendations)
            else:
                st.warning("Please enter either a preferred movie in the input box or select some genres.")

    
    # st.divider()
    st.feedback("thumbs")
    st.warning("""Disclaimer: This model has been quantized for optimization.
            Check here for more details: [GitHub Repo🐙](https://github.com/verneylmavt/st-mov-recsys)""")
    st.divider()
    
    st.subheader("""Pre-Processing""")
    st.code(model_info[model]["pre_processing"], language="None")
    
    st.subheader("""Parameters""")
    st.code(model_info[model]["parameters"], language="None")
    
    st.subheader("""Model""")
    st.code(model_info[model]["model_code"], language="python")
    
    if "forward_pass" in model_info[model]:
        st.subheader("Forward Pass")
        for key, value in model_info[model]["forward_pass"].items():
            st.caption(key)
            st.latex(value)
    else: pass

if __name__ == "__main__":
    main()