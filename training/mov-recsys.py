#!/usr/bin/env python
# coding: utf-8

# In[155]:


import os
import time
import random


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# In[ ]:


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity


# In[ ]:


from onnxruntime.quantization import quantize_dynamic, QuantType


# In[101]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ### NCF

# #### Seed Setting

# ```markdown
# In here, the code sets the random seed for reproducibility across random, NumPy, and PyTorch operations. This ensures consistent results by fixing the seed for both CPU and GPU computations.
# ```

# In[ ]:


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed()


# #### Data Loading

# ```markdown
# In here, the code loads the MovieLens dataset by reading the ratings.csv and movies.csv files from the specified dataset path using Pandas.
# ```

# In[102]:


dataset_path = '../../data/ml-32m/'


# In[103]:


ratings = pd.read_csv(os.path.join(dataset_path, 'ratings.csv'))
movies = pd.read_csv(os.path.join(dataset_path, 'movies.csv'))


# In[104]:


print(f'Number of ratings: {ratings.shape[0]}')
print(f'Number of users: {ratings.userId.nunique()}')
print(f'Number of movies: {ratings.movieId.nunique()}')


# #### Label Encoding

# ```markdown
# In here, the code encodes the userId and movieId columns into numerical labels using LabelEncoder, which transforms categorical user and movie identifiers into integer indices. It also determines the number of unique users and movies.
# ```

# In[105]:


user_encoder = LabelEncoder()
movie_encoder = LabelEncoder()

ratings['user'] = user_encoder.fit_transform(ratings['userId'])
ratings['movie'] = movie_encoder.fit_transform(ratings['movieId'])

num_users = ratings['user'].nunique()
num_movies = ratings['movie'].nunique()


# #### Data Splitting

# ```markdown
# In here, the code splits the ratings data into training, validation, and test sets using train_test_split. It first splits off 10% of the data for testing and then splits the remaining data into training and validation sets.
# ```

# In[60]:


train_val, test = train_test_split(ratings, test_size=0.1, random_state=42)
train, val = train_test_split(train_val, test_size=0.1111, random_state=42) 


# In[61]:


# print(f'Train Set: {train.shape[0]} samples')
# print(f'Validation Set: {val.shape[0]} samples')
# print(f'Test Set: {test.shape[0]} samples')


# #### Dataset and DataLoader

# ```markdown
# In here, the code defines a custom Dataset class named MovieLensDataset that facilitates the retrieval of user, movie, and rating data samples. This class is essential for creating data loaders that can feed data into the model during training and evaluation. Then, it creates instances of the MovieLensDataset for training, validation, and testing. Lastly, it wraps these datasets in DataLoader objects to enable efficient batching and shuffling of data during the training and evaluation processes.
# ```

# In[62]:


class MovieLensDataset(Dataset):
    def __init__(self, dataframe):
        self.users = dataframe['user'].values
        self.movies = dataframe['movie'].values
        self.ratings = dataframe['rating'].values.astype(np.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        user = self.users[idx]
        movie = self.movies[idx]
        rating = self.ratings[idx]
        return user, movie, rating


# In[63]:


train_dataset = MovieLensDataset(train)
val_dataset = MovieLensDataset(val)
test_dataset = MovieLensDataset(test)


# In[64]:


batch_size = 1024
train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_iter = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


# #### Model Definition

# ```markdown
# In here, the code defines the NCF class, a neural network model for collaborative filtering. The model consists of embedding layers for users and movies, followed by a multilayer perceptron (MLP) with ReLU activations and dropout for regularization. The final output layer predicts the rating by producing a single scalar value.
# 
# • User and Item Embeddings
# The embedding layers represent users and items (movies in this context) as dense vectors in a latent space of fixed dimensionality (embedding_dim). These embeddings capture unique characteristics of each user and item based on their interaction patterns. The embeddings are initialized with small random values (via nn.init.normal_) to start training effectively. As the model learns, these embeddings are updated to encode meaningful latent features relevant to predicting user preferences.
# 
# • Concatenation of User and Item Embeddings
# After retrieving the user and item embeddings, the model concatenates them along the last dimension. This operation merges the user and item information into a single vector, serving as the input to the MLP. This concatenated representation allows the MLP to learn the joint interaction between user and item features.
# 
# • Multi-Layer Perceptron (MLP)
# The MLP in the NCF model captures complex, non-linear relationships between users and items:
# - Structure:
# The input to the MLP is the concatenated embeddings of users and items.
# The MLP is composed of fully connected (Linear) layers, interleaved with ReLU activations and dropout.
# - Hidden Layers:
# Each hidden layer progressively transforms the input, learning higher-order feature interactions.
# The number and size of the hidden layers are configurable via the hidden_layers parameter.
# - Dropout Regularization:
# Dropout layers prevent overfitting by randomly deactivating neurons during training, ensuring the MLP generalizes well to unseen data.
# 
# • Output Layer
# The final layer of the model is a linear layer that outputs a single scalar value. This value represents the predicted interaction strength or rating between the user and item. The absence of an activation function in the output layer allows the model to predict continuous values (e.g., ratings) or logits (for binary tasks).
# ```

# In[65]:


class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=50, hidden_layers=[128, 64], dropout=0.2):
        super(NCF, self).__init__()
        
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


# In[67]:


# print(net)


# #### Training Function

# ```markdown
# In here, the code defines the train_epoch function, which handles the training process for one epoch. It iterates over the training data loader, performs forward passes, computes the loss, backpropagates the gradients, updates the model parameters, and accumulates the running loss and Mean Absolute Error (MAE) for monitoring.
# ```

# In[69]:


def train_epoch(net, train_iter, optimizer, criterion, device):
    net.train()
    running_loss = 0.0
    running_mae = 0.0
    for users, movies, ratings_batch in train_iter:
        users = users.to(device, dtype=torch.long)
        movies = movies.to(device, dtype=torch.long)
        ratings_batch = ratings_batch.to(device)

        optimizer.zero_grad()
        outputs = net(users, movies)
        loss = criterion(outputs, ratings_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * users.size(0)
        running_mae += torch.abs(outputs - ratings_batch).sum().item()

    epoch_loss = running_loss / len(train_iter.dataset)
    epoch_mae = running_mae / len(train_iter.dataset)
    return epoch_loss, epoch_mae


# #### Evaluation Function

# ```markdown
# In here, the code defines the evaluate_epoch function, which evaluates the model's performance on the validation set. It computes the loss and MAE without updating the model parameters, allowing for monitoring of the model's generalization ability.
# ```

# In[ ]:


def evaluate_epoch(net, val_iter, criterion, device):
    net.eval()
    running_loss = 0.0
    running_mae = 0.0
    with torch.no_grad():
        for users, movies, ratings_batch in val_iter:
            users = users.to(device, dtype=torch.long)
            movies = movies.to(device, dtype=torch.long)
            ratings_batch = ratings_batch.to(device)

            outputs = net(users, movies)
            loss = criterion(outputs, ratings_batch)

            running_loss += loss.item() * users.size(0)
            running_mae += torch.abs(outputs - ratings_batch).sum().item()

    epoch_loss = running_loss / len(val_iter.dataset)
    epoch_mae = running_mae / len(val_iter.dataset)
    return epoch_loss, epoch_mae


# #### Training

# ```markdown
# In here, the code implements the main training loop that runs for a specified number of epochs. For each epoch, it trains the model using the training data, evaluates it on the validation data, and records the training and validation losses and accuracies. It also incorporates early stopping by monitoring the validation loss and saving the best model.
# ```

# In[ ]:


embedding_dim = 50
hidden_layers = [128, 64]
dropout = 0.2
net = NCF(num_users, num_movies, embedding_dim, hidden_layers, dropout).to(device)


# In[68]:


criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.01)
num_epochs = 20

best_val_loss = float('inf')
patience = 3
trigger_times = 0

train_losses = []
val_losses = []
train_maes = []
val_maes = []


# In[ ]:


for epoch in range(1, num_epochs + 1):
    start_time = time.time()
    print(f"epoch {epoch}/{num_epochs}")
    
    epoch_loss, epoch_mae = train_epoch(net, train_iter, optimizer, criterion, device)
    train_losses.append(epoch_loss)
    train_maes.append(epoch_mae)
    
    val_epoch_loss, val_epoch_mae = evaluate_epoch(net, val_iter, criterion, device)
    val_losses.append(val_epoch_loss)
    val_maes.append(val_epoch_mae)

    end_time = time.time()
    epoch_time = end_time - start_time

    print(f"train loss: {epoch_loss:.4f}, train mae: {epoch_mae:.4f}, val loss: {val_epoch_loss:.4f}, val mae: {val_epoch_mae:.4f}, time: {epoch_time:.2f}s")

    if val_epoch_loss < best_val_loss:
        best_val_loss = val_epoch_loss
        trigger_times = 0
        torch.save(net.state_dict(), 'best_model.pth')
    else:
        trigger_times += 1
        if trigger_times >= patience:
            break


# #### Evaluation Metrics

# ```markdown
# In here, the code defines the cal_metrics function, which evaluates the trained model on the test dataset. It calculates and prints evaluation metrics including Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE) to assess the model's performance.
# ```

# In[70]:


net.load_state_dict(torch.load('best_model.pth'))


# In[71]:


def cal_metrics(net, test_iter):
    net.eval()
    test_predictions = []
    test_targets = []

    with torch.no_grad():
        for users, movies, ratings_batch in test_iter:
            users = users.to(device, dtype=torch.long)
            movies = movies.to(device, dtype=torch.long)
            ratings_batch = ratings_batch.to(device)

            outputs = net(users, movies)
            test_predictions.extend(outputs.cpu().numpy())
            test_targets.extend(ratings_batch.cpu().numpy())

    test_predictions = np.array(test_predictions)
    test_targets = np.array(test_targets)

    test_mse = mean_squared_error(test_targets, test_predictions)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(test_targets, test_predictions)
    
    print(f'MSE: {test_mse:.4f}')
    print(f'RMSE: {test_rmse:.4f}')
    print(f'MAE: {test_mae:.4f}')

    return None


# In[72]:


cal_metrics(net, test_iter)


# #### User-Based Recommendations 

# ```markdown
# In here, the code defines the recommend_movies function, which generates movie recommendations for an existing user by predicting ratings for movies the user hasn't rated yet. It encodes the user ID, filters out already rated movies, predicts ratings using the trained model, and selects the top-rated movies to recommend.
# ```

# In[211]:


def recommend_movies(user_id, net, movies_df, user_encoder, movie_encoder, ratings_original, num_recommendations=10):
    net.eval()
    user_encoded = user_encoder.transform([user_id])[0]
    
    all_movies = np.arange(num_movies)
    user_data = ratings_original[ratings_original['userId'] == user_id]
    rated_movie_ids = user_data['movieId'].unique()
    rated_movies = movie_encoder.transform(rated_movie_ids)
    movies_to_predict = np.setdiff1d(all_movies, rated_movies)
    
    user_tensor = torch.tensor([user_encoded] * len(movies_to_predict)).to(device, dtype=torch.long)
    movie_tensor = torch.tensor(movies_to_predict).to(device, dtype=torch.long)
    
    with torch.no_grad():
        predictions = net(user_tensor, movie_tensor).cpu().numpy()
    
    top_indices = predictions.argsort()[-num_recommendations:][::-1]
    top_movie_indices = movies_to_predict[top_indices]
    top_movie_ids_original = movie_encoder.inverse_transform(top_movie_indices)
    recommended_movies = movies_df[movies_df['movieId'].isin(top_movie_ids_original)][['movieId', 'title']]
    
    return recommended_movies


# In[212]:


recommended = recommend_movies(2, net, movies, user_encoder, movie_encoder, ratings, 10)
print(recommended)


# #### Embedding Extraction and Similarity Computation

# In[174]:


print(movies.head())


# In[175]:


print(ratings.head())


# ##### Movie Embeddings Extraction

# ```markdown
# In here, the code defines the get_movie_embeddings function, which retrieves the movie embeddings from the trained model. It normalizes these embeddings to unit vectors, facilitating the computation of cosine similarities between movies.
# ```

# In[176]:


def get_movie_embeddings(net):
    movie_embeddings = net.item_embedding.weight.data.cpu().numpy()
    movie_embeddings_normalized = movie_embeddings / np.linalg.norm(movie_embeddings, axis=1, keepdims=True)
    return movie_embeddings_normalized


# In[213]:


movie_embeddings_normalized = get_movie_embeddings(net)


# ##### Cosine Similarity Computation

# ```markdown
# In here, the code defines the compute_cosine_similarity function, which calculates the cosine similarity matrix for the provided embeddings. This matrix quantifies the similarity between each pair of movie embeddings.
# ```

# In[177]:


def compute_cosine_similarity(embeddings):
    similarity_matrix = cosine_similarity(embeddings)
    return similarity_matrix


# ##### Genre Processing

# ```markdown
# In here, the code processes the movie genres by splitting the genre strings, encoding them into multi-hot vectors, and appending these encoded genres to the movies_inf DataFrame. This allows the model to consider genre information during recommendations.
# ```

# In[192]:


movies_inf = movies.copy()
ratings_inf = ratings.copy()


# In[193]:


movies_inf['genres'] = movies_inf['genres'].apply(lambda x: [genre.strip() for genre in x.split('|')])
all_genres = sorted(set(genre for sublist in movies_inf['genres'] for genre in sublist if genre != '(no genres listed)'))


# In[194]:


def multi_hot_encode(genres, all_genres):
    return [1 if genre in genres else 0 for genre in all_genres]


# In[195]:


movies_inf['genres_encoded'] = movies_inf['genres'].apply(lambda x: multi_hot_encode(x, all_genres))
genres_matrix = np.vstack(movies_inf['genres_encoded'].values)
genres_df = pd.DataFrame(genres_matrix, columns=all_genres)
movies_inf = pd.concat([movies_inf, genres_df], axis=1)


# #### Similarity-Based Recommendations

# ```markdown
# In here, the code defines the recommend_similar_movies function, which recommends movies similar to a given movie based on cosine similarity of their embeddings. It ensures the input movie exists, retrieves its embedding, computes similarities with all other movies, and selects the top similar movies to recommend.
# ```

# In[133]:


def recommend_similar_movies(movie_id, movies_df, movie_encoder, movie_embeddings, top_n=10):
    if movie_id not in movies_df['movieId'].values:
        print("Movie ID not found in the dataset.")
        return pd.DataFrame()
    movie_encoded = movie_encoder.transform([movie_id])[0]
    
    target_embedding = movie_embeddings[movie_encoded].reshape(1, -1)

    similarities = cosine_similarity(target_embedding, movie_embeddings).flatten()
    similar_indices = similarities.argsort()[-(top_n + 1):-1][::-1]
    similar_movie_ids = movie_encoder.inverse_transform(similar_indices)
    
    recommended_movies = movies_df[movies_df['movieId'].isin(similar_movie_ids)][['movieId', 'title']]
    recommended_movies = recommended_movies.copy()
    recommended_movies['similarity'] = similarities[similar_indices]
    
    return recommended_movies.reset_index(drop=True)


# In[135]:


movie_id_example = movies_inf['movieId'].iloc[0]
similar_movies = recommend_similar_movies(movie_id_example, movies_inf, movie_encoder, movie_embeddings_normalized, top_n=10)
print(similar_movies)


# #### Genre-Based Recommendations

# ```markdown
# In here, the code defines the recommend_by_genre function, which recommends movies based on specified genres. It filters movies that match all provided genres, calculates their popularity based on rating counts, sorts them by popularity, and returns the top N recommendations.
# ```

# In[141]:


def recommend_by_genre(genres, movies_df, ratings_original, top_n=10):
    valid_genres = sorted(all_genres)
    for genre in genres:
        if genre not in valid_genres:
            print(f"Genre '{genre}' is not recognized. Valid genres are: {valid_genres}")
            return pd.DataFrame()
        
    filtered_movies = movies_df
    for genre in genres:
        filtered_movies = filtered_movies[filtered_movies[genre] == 1]
    
    if filtered_movies.empty:
        print("No movies found with the specified genres.")
        return pd.DataFrame()
    
    popularity = ratings_original.groupby('movieId').size().reset_index(name='rating_count')
    recommended = filtered_movies.merge(popularity, on='movieId', how='left').fillna(0)
    recommended = recommended.sort_values(by='rating_count', ascending=False)
    recommended = recommended[['movieId', 'title', 'rating_count']].head(top_n)
    return recommended.reset_index(drop=True)


# In[144]:


genres_example = ['Comedy']
genre_based_movies = recommend_by_genre(genres_example, movies_inf, ratings_inf, top_n=10)
print(genre_based_movies)


# #### Combined Recommendations

# ```markdown
# In here, the code defines the recommend_combined function, which generates movie recommendations by combining similarity-based and genre-based approaches. It ensures the input movie exists, retrieves similar movies, filters them based on specified genres, incorporates popularity metrics, and returns the top N combined recommendations.
# ```

# In[147]:


def recommend_combined(movie_id, genres, net, movies_df, movie_encoder, movie_embeddings, ratings_original, top_n=10):

    if movie_id not in movies_df['movieId'].values:
        print("Movie ID not found in the dataset.")
        return pd.DataFrame()
    
    movie_encoded = movie_encoder.transform([movie_id])[0]
    
    target_embedding = movie_embeddings[movie_encoded].reshape(1, -1)
    
    similarities = cosine_similarity(target_embedding, movie_embeddings).flatten()
    similar_indices = similarities.argsort()[-(top_n * 2 + 1):-1][::-1]
    similar_movie_ids = movie_encoder.inverse_transform(similar_indices)
    similar_movies = movies_df[movies_df['movieId'].isin(similar_movie_ids)].copy()
    similar_movies['similarity'] = similarities[similar_indices]
    
    for genre in genres:
        if genre in all_genres:
            similar_movies = similar_movies[similar_movies[genre] == 1]
        else:
            print(f"Genre '{genre}' is not recognized.")
            return pd.DataFrame()
    
    if similar_movies.empty:
        print("No similar movies found with the specified genres.")
        return pd.DataFrame()
    
    popularity = ratings_original.groupby('movieId').size().reset_index(name='rating_count')
    
    similar_movies = similar_movies.merge(popularity, on='movieId', how='left').fillna(0)
    similar_movies = similar_movies.sort_values(by=['similarity', 'rating_count'], ascending=[False, False])
    
    recommended = similar_movies[['movieId', 'title', 'similarity', 'rating_count']].head(top_n)
    
    return recommended.reset_index(drop=True)


# In[153]:


movie_id_example_combined = 10
genres_example_combined = ['Action', 'Adventure']
combined_recommended = recommend_combined(
    movie_id_example_combined,
    genres_example_combined,
    net,
    movies_inf,
    movie_encoder,
    movie_embeddings_normalized,
    ratings_inf,
    top_n=10
)
print(combined_recommended)

