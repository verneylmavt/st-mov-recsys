import os
import io
import requests
import streamlit as st
from streamlit_extras.chart_container import chart_container
from streamlit_extras.mention import mention
# from streamlit_extras.echo_expander import echo_expander
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
@st.cache_resource
def load_model_info():
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
    return model_info
# ----------------------
# Loading Function
# ----------------------
@st.cache_resource
def load_engine():
    db_host = "movie-ratings.clweeg2y8lzr.ap-southeast-1.rds.amazonaws.com"
    db_name = "postgres"
    db_user = "postgres"
    db_password = "postgres-root"
    db_port = "5432"
    engine = create_engine(f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")
    # engine = create_engine(f"postgresql+psycopg://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")
    return engine

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
# Prediction Function
# ----------------------

def recommend_similar_movies_name(movie_name, movies, movie_encoder, embeddings, top_n):
    movie_name = movie_name.strip()
    # st.write(movie_name)
    # st.write(f"{movies.shape}")
    # st.dataframe(movies.head())
    matching_movies = movies[movies['title'].str.contains(movie_name, case=False, regex=False)]
    # st.write(f" Hello: {matching_movies}")
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

# def recommend_similar_movies_name(movie_name, movies, movie_encoder, embeddings, top_n):
#     movie_name = movie_name.strip()
#     matching_movies = movies[movies['title'].str.contains(movie_name, case=False)]
#     if matching_movies.empty:
#         st.error("Movie Name Invalid")
#         return None
#     movie_id = matching_movies.iloc[0]['movieId']
    
#     movie_encoded = movie_encoder.transform([movie_id])[0]
#     target_embedding = embeddings[movie_encoded].reshape(1, -1)
    
#     similarities = cosine_similarity(target_embedding, embeddings).flatten()
#     similar_indices = similarities.argsort()[-(top_n + 1):-1][::-1]
#     similar_movie_ids = movie_encoder.inverse_transform(similar_indices)
    
#     recommendations = movies[movies['movieId'].isin(similar_movie_ids)][['movieId', 'title']]
#     recommendations['similarity'] = similarities[similar_indices]
#     recommendations["similarity"] = np.ceil(recommendations["similarity"] * 1000) / 100
    
#     recommendations = recommendations.reset_index(drop=True)
#     recommendations.index += 1
    
#     recommendations.rename(columns={
#     "movieId": "Movie ID",
#     "title": "Title",
#     "similarity": "Cosine Similarity"
#     }, inplace=True)
#     recommendations["Year"] = recommendations["Title"].str.extract(r"\((\d{4})\)")
#     recommendations["Title"] = recommendations["Title"].str.replace(r" \(\d{4}\)", "", regex=True)
#     recommendations = recommendations[["Movie ID", "Title", "Year", "Cosine Similarity"]]
    
#     return recommendations, similar_indices, matching_movies.iloc[0]['title'], target_embedding


def recommend_by_genre_pop(genres, movies, top_n):
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
    
    # popularity = ratings.groupby('movieId').size().reset_index(name='rating_count')
    # st.write(len(popularity))
    
    engine = load_engine()
    
    with engine.connect() as conn:
        query = 'SELECT "movieId", COUNT(*) AS "rating_count" FROM "ratings" GROUP BY "movieId";'
        popularity = pd.read_sql(query, con=conn)
        # st.write(len(popularity))
    
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
    if similar_movies.empty:
        st.error("Movie Name Invalid")
        return None
    
    similar_movie_ids = similar_movies["Movie ID"].astype(int).values
    
    filtered_movies = movies[movies['movieId'].isin(similar_movie_ids)].copy()
    # for genre in genres:
    #     if genre in all_genres:
    #         filtered_movies = filtered_movies[filtered_movies[genre] == 1]
    #     else:
    #         print(f"Genre '{genre}' Invalid.")
    #         return pd.DataFrame()
    for genre in genres:
        filtered_movies = filtered_movies[filtered_movies[genre] == 1]
    if filtered_movies.empty:
        st.error("No Movies Found w/ Specified Genre(s)")
        return None
    
    # popularity = ratings.groupby('movieId').size().reset_index(name='rating_count')
    
    engine = load_engine()
    
    with engine.connect() as conn:
        query = 'SELECT "movieId", COUNT(*) AS "rating_count" FROM "ratings" GROUP BY "movieId";'
        popularity = pd.read_sql(query, con=conn)
        # st.write(len(popularity))
    
    
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
    st.set_page_config(page_title="Movie Recommender System"
                    # layout="wide"
                    )
    st.title("Movie Recommender System")
    
    model_info = load_model_info()
    model_names = list(model_info.keys())
    model = st.selectbox("Select a Model", model_names)
    st.divider()
    
    # net = load_model(model)
    movies = load_movies(model)
    # ratings = load_ratings(model)
    embeddings = load_embeddings(model)
    all_genres = load_genres()
    training_data = load_training_data()
    
    # user_encoder = LabelEncoder()
    # user_encoder.fit_transform(ratings['userId'])
    # movie_encoder = LabelEncoder()
    # movie_encoder.fit_transform(ratings['movieId'])
    
    movie_encoder = LabelEncoder()
    movie_encoder.fit(movies['movieId'])
    
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
    
    with st.form(key="mov_recsys_form"):
        movie_name = st.text_input("Preferred Movie:")
        st.caption("_e.g. Toy Story_")
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
                        # recommendations, similar_indices, matching_movies, target_embedding = recommend_similar_movies_name(movie_name, movies, movie_encoder, embeddings, top_n)
                        # plot_data = visualize_recommendations(recommendations, similar_indices, matching_movies, target_embedding, embeddings)
                        # with chart_container(plot_data, tabs=('Chart 📈', 'Dataframe 📄')):
                        # st.scatter_chart(
                        #                 plot_data,
                        #                 x="x",
                        #                 y="y",
                        #                 color="Title",
                        #                 size="Similarity",
                        #             )
                    elif not movie_name and selected_genres:
                        recommendations = recommend_by_genre_pop(selected_genres, movies, top_n)
                    elif movie_name and selected_genres:
                        recommendations = recommend_combined_mix(movie_name, selected_genres, movies, movie_encoder, embeddings, top_n)
                    if recommendations is not None:
                        st.dataframe(recommendations, use_container_width=True)
            else:
                st.warning("Please enter either a preferred movie in the input box or select some genres.")
    # try:
    #     st.scatter_chart(
    #                     plot_data,
    #                     x="x",
    #                     y="y",
    #                     color="Title",
    #                     size="Similarity",
    #                 )
    # except:
    #     pass
    # st.divider()
    st.feedback("thumbs")
    st.warning("""Disclaimer: This model has been quantized for optimization.""")
    mention(
            label="GitHub Repo: verneylmavt/st-mov-recsys",
            icon="github",
            url="https://github.com/verneylmavt/st-mov-recsys"
        )
    mention(
            label="Other ML Tasks",
            icon="streamlit",
            url="https://verneylogyt.streamlit.app/"
        )
    st.divider()
    
    st.subheader("""Pre-Processing""")
    st.code(model_info[model]["pre_processing"], language="None")
    
    st.subheader("""Parameters""")
    st.code(model_info[model]["parameters"], language="None")
    
    st.subheader("""Model""")
    # st.code(model_info[model]["model_code"], language="python")
    from streamlit_extras.echo_expander import echo_expander
    with echo_expander(code_location="below", label="Code"):
        import torch
        import torch.nn as nn
        
        
        class Model(nn.Module):
            def __init__(self, num_users, num_items, embedding_dim=50, hidden_layers=[128, 64], dropout=0.2):
                super(Model, self).__init__()
                # Embedding Layer for User Representations
                self.user_embedding = nn.Embedding(num_users, embedding_dim)
                # Embedding Layer for Item Representations
                self.item_embedding = nn.Embedding(num_items, embedding_dim)
                # Weight Initialization for User Embedding
                nn.init.normal_(self.user_embedding.weight, std=0.01)
                # Weight Initialization for Item Embedding
                nn.init.normal_(self.item_embedding.weight, std=0.01)
                
                # Sequential Model for Multi-Layer Perceptron (MLP)
                layers = []
                # Input Size for MLP (Concatenated User and Item Embeddings)
                input_size = embedding_dim * 2
                # Hidden Layers Construction for MLP
                for hidden in hidden_layers:
                    # Fully Connected Layer for MLP
                    layers.append(nn.Linear(input_size, hidden))
                    # Activation Layer for Non-Linear Transformations
                    layers.append(nn.ReLU())
                    # Dropout Layer for Regularization
                    layers.append(nn.Dropout(dropout))
                    # Update Input Size for Next Layer
                    input_size = hidden
                # MLP Network for Learning User-Item Interactions
                self.mlp = nn.Sequential(*layers)
                
                # Output Layer for Predicting Ratings
                self.output_layer = nn.Linear(input_size, 1)
                
            def forward(self, user, item):
                # Embedding of Input Users
                user_emb = self.user_embedding(user)
                # Embedding of Input Items
                item_emb = self.item_embedding(item)
                # Concatenation of User and Item Embeddings
                x = torch.cat([user_emb, item_emb], dim=-1)
                # Transformation of Concatenated Embeddings w/ MLP
                x = self.mlp(x)
                # Transformation of MLP Output → Predicted Ratings
                rating = self.output_layer(x)
                # Squeezing of Output Ratings
                return rating.squeeze()
    
    if "forward_pass" in model_info[model]:
        st.subheader("Forward Pass")
        for key, value in model_info[model]["forward_pass"].items():
            st.caption(key)
            st.latex(value)
    else: pass

    st.subheader("""Training""")
    # st.line_chart(training_data.set_index("Epoch"))
    with chart_container(training_data):
        st.line_chart(training_data.set_index("Epoch"))
    
    st.subheader("""Evaluation Metrics""")
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Square Error (MSE)", "0.6059", border=True)
    col2.metric("Root Mean Squared Error (RMSE)", "0.7784", border=True)
    col3.metric("Mean Absolute Error (MAE)", "0.5853", border=True)
    
if __name__ == "__main__":
    main()