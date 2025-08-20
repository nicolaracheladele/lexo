import streamlit as st
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import os
from sklearn.metrics.pairwise import cosine_similarity

# Load Data
@st.cache_data
def load_data():
    ratings = pd.read_csv("goodbooks-10k/ratings.csv")
    books = pd.read_csv("goodbooks-10k/books.csv")
    return ratings, books

ratings, books = load_data()

# Pivot user-book rating matrix
user_book_matrix = ratings.pivot(index='user_id', columns='book_id', values='rating').fillna(0)

# Compute user-user cosine similarity
user_sim = cosine_similarity(user_book_matrix)
user_sim_df = pd.DataFrame(user_sim, index=user_book_matrix.index, columns=user_book_matrix.index)

# UI: select "seed" users similar to current taste
selected_users = st.multiselect('Select similar users:', user_book_matrix.index.tolist())

# Find candidate books (liked by selected users, but not rated by current user)
def get_recommendations(selected_users, current_user_id, top_n=5):
    # Get books rated by selected users with high ratings
    candidate_books = ratings[
        (ratings['user_id'].isin(selected_users)) & (ratings['rating'] >= 4)
    ]['book_id'].unique()

    # Filter out books already read by current user
    read_books = ratings[ratings['user_id'] == current_user_id]['book_id'].unique()
    rec_books = [b for b in candidate_books if b not in read_books]

    # Get top N books by average rating from selected users
    avg_ratings = ratings[
        (ratings['user_id'].isin(selected_users)) & (ratings['book_id'].isin(rec_books))
    ].groupby('book_id')['rating'].mean()

    top_books = avg_ratings.sort_values(ascending=False).head(top_n).index.tolist()
    return top_books

current_user = st.number_input('Your user ID:', min_value=1, max_value=10000, value=1)
if selected_users:
    recommendations = get_recommendations(selected_users, current_user)
    st.write("Recommended books:")
    for book_id in recommendations:
        title = books.loc[books['book_id'] == book_id, 'title'].values[0]
        st.write(f"- {title}")

        # Let user rate the book
        rating = st.slider(f'Rate "{title}"', 1, 5, 3, key=f'rate_{book_id}')
        if st.button(f'Submit rating for "{title}"', key=f'btn_{book_id}'):
            # Add rating to ratings dataset (in-memory)
            new_rating = pd.DataFrame({
                'user_id': [current_user],
                'book_id': [book_id],
                'rating': [rating]
            })
            ratings = pd.concat([ratings, new_rating], ignore_index=True)
            st.success(f'Rating submitted for "{title}"!')
else:
    st.info("Select similar users to get recommendations.")