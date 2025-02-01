import os
import io
import requests
import streamlit as st
from streamlit_extras.chart_container import chart_container
from streamlit_extras.mention import mention
from streamlit_extras.echo_expander import echo_expander
import numpy as np
import pandas as pd
import onnxruntime as ort
from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import plotly.express as px

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

Number of Users = 138,493
Number of Movies = 27,278
Embedding Dimension = 50
Number of MLP (Multi-Layer Perceptron) Layers = 2 (128+64)
Dropout Rate = 0.2

Epochs = 20
Learning Rate = 0.001
Loss Function = MSELoss
Optimizer = AdamW
Weight Decay = 0.01
        """,
        "model_code": """
class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=50, hidden_layers=[128, 64], dropout=0.2):
        super(NCF, self).__init__()
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
# Database Engine Setup
# ----------------------
@st.cache_resource
def load_engine():
    db_host = "movie-ratings.clweeg2y8lzr.ap-southeast-1.rds.amazonaws.com"
    db_name = "postgres"
    db_user = "postgres"
    db_password = "postgres-root"
    db_port = "5432"
    engine = create_engine(f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")
    return engine

# ----------------------
# Loading Functions (non-ratings)
# ----------------------
@st.cache_resource
def load_model(model_name):
    try:
        model_path = os.path.join("models", model_name, "model-q.onnx")
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
def load_embeddings(model_name):
    try:
        model_path = os.path.join("models", model_name, "embeddings.npy")
        embeddings = np.load(model_path)
        return embeddings
    except FileNotFoundError:
        st.error(f"embeddings.npy file not found for {model_name}. Please ensure 'embeddings.npy' exists in the model directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the embeddings for {model_name}: {e}")
        st.stop()
        
@st.cache_data
def load_genres():
    return ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'IMAX', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

@st.cache_data
def load_training_data():
    training_data = {
        "Epoch": list(range(1, 15)),
        "Train Loss": [
            0.7816, 0.6662, 0.6286, 0.6047, 0.5881, 0.5767, 0.5683, 0.5621, 
            0.5572, 0.5533, 0.5503, 0.5477, 0.5456, 0.5439
        ],
        "Train MAE": [
            0.6713, 0.6177, 0.5989, 0.5865, 0.5778, 0.5717, 0.5674, 0.5641,
            0.5615, 0.5595, 0.5579, 0.5566, 0.5555, 0.5545
        ],
        "Validation Loss": [
            0.6874, 0.6539, 0.6345, 0.6226, 0.6151, 0.6107, 0.6119, 0.6072, 
            0.6094, 0.6076, 0.6067, 0.6082, 0.6073, 0.6069
        ],
        "Validation MAE": [
            0.6271, 0.6109, 0.6016, 0.5937, 0.5910, 0.5886, 0.5889, 0.5853,
            0.5862, 0.5855, 0.5857, 0.5872, 0.5877, 0.5848
        ]
    }
    return pd.DataFrame(training_data)

# ----------------------
# Recommendation Functions
# ----------------------
def recommend_similar_movies_name(movie_name, movies, movie_encoder, embeddings, top_n):
    movie_name = movie_name.strip()
    matching_movies = movies[movies['title'].str.contains(movie_name, case=False, regex=False)]
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


def recommend_by_genre_pop(genres, movies, top_n):
    filtered_movies = movies
    for genre in genres:
        filtered_movies = filtered_movies[filtered_movies[genre] == 1]
    if filtered_movies.empty:
        st.error("No Movies Found w/ Specified Genre(s)")
        return None
    
    engine = load_engine()
    
    with engine.connect() as conn:
        query = 'SELECT "movieId", COUNT(*) AS "rating_count" FROM "ratings" GROUP BY "movieId";'
        popularity = pd.read_sql(query, con=conn)
        st.write(len(popularity))
    
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


def recommend_combined_mix(movie_name, genres, movies, movie_encoder, embeddings, top_n=10):
    similar_movies = recommend_similar_movies_name(movie_name, movies, movie_encoder, embeddings, top_n=top_n*10)
    if similar_movies is None or similar_movies.empty:
        st.error("Movie Name Invalid")
        return None
    
    similar_movie_ids = similar_movies["Movie ID"].astype(int).values
    filtered_movies = movies[movies['movieId'].isin(similar_movie_ids)].copy()
    
    for genre in genres:
        filtered_movies = filtered_movies[filtered_movies[genre] == 1]
    if filtered_movies.empty:
        st.error("No Movies Found w/ Specified Genre(s)")
        return None
    
    engine = load_engine()
    
    with engine.connect() as conn:
        query = 'SELECT "movieId", COUNT(*) AS "rating_count" FROM "ratings" GROUP BY "movieId";'
        popularity = pd.read_sql(query, con=conn)
        st.write(len(popularity))
    
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

def visualize_recommendations(recommendations, similar_indices, input_movie_title, input_movie_embedding, embeddings):
    n_samples = len(similar_indices) + 1
    perplexity = min(30, n_samples - 1) if n_samples > 1 else 1
    all_embeddings = np.vstack([input_movie_embedding, embeddings[similar_indices]])
    
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    reduced_embeddings = tsne.fit_transform(all_embeddings)
    
    plot_data = pd.DataFrame({
        'x': reduced_embeddings[:, 0],
        'y': reduced_embeddings[:, 1],
        'Title': pd.concat([pd.Series([input_movie_title]), recommendations['Title']]).reset_index(drop=True),
        'Similarity': pd.concat([pd.Series([100.0]), recommendations['Cosine Similarity']]).reset_index(drop=True)
    })
    return plot_data

# ----------------------
# Page UI
# ----------------------
def main():
    st.title("Movie Recommender System")
    
    model_names = list(model_info.keys())
    model = st.selectbox("Select a Model", model_names)
    st.divider()
    
    movies = load_movies(model)
    embeddings = load_embeddings(model)
    all_genres = load_genres()
    
    movie_encoder = LabelEncoder()
    movie_encoder.fit(movies['movieId'])
    
    st.subheader(model_info[model]["subheader"])
    
    with st.form(key="mov_recsys_form"):
        movie_name = st.text_input("Preferred Movie:")
        st.caption("_e.g. Toy Story_")
        selected_genres = st.multiselect("Preferred Genres", all_genres)
        top_n = st.slider("Number of Recommendations", 0, 50, 10)
        submit_button = st.form_submit_button(label="Recommend")
        
        if submit_button:
            if movie_name or selected_genres:
                with st.spinner('Recommending...'):
                    if movie_name and not selected_genres:
                        recommendations = recommend_similar_movies_name(movie_name, movies, movie_encoder, embeddings, top_n)
                    elif not movie_name and selected_genres:
                        recommendations = recommend_by_genre_pop(selected_genres, movies, top_n)
                    elif movie_name and selected_genres:
                        recommendations = recommend_combined_mix(movie_name, selected_genres, movies, movie_encoder, embeddings, top_n)
                    if recommendations is not None:
                        st.dataframe(recommendations)
            else:
                st.warning("Please enter either a preferred movie in the input box or select some genres.")

if __name__ == "__main__":
    main()