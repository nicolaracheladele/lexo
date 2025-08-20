import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split


# Read dataset
ratings = pd.read_csv("goodbooks-10k/ratings.csv")

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['user_id', 'book_id', 'rating']], reader)

trainset, testset = train_test_split(data, test_size=0.2)

model = SVD()
model.fit(trainset)

pred = model.predict(uid=123, iid=456)  # Predict rating
print(pred)
