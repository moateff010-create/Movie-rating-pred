import pandas as pd
from surprise import SVD,Dataset,Reader
from surprise import accuracy
from surprise.model_selection import train_test_split
from fastapi import FastAPI,HTTPException
app = FastAPI(title= "Movie Recommendation API",
              description= "API For Moivie Recoomendation By Using SVD")
cols = ["user_id","item_id","ratings","timestamp"]
ratings = pd.read_csv(r"D:/datasets/ml-100k/u.data",sep="\t",names=cols)
reader = Reader(rating_scale=(1,5))
data = Dataset.load_from_df(ratings[["user_id","item_id","ratings"]],reader)
trainset,testset = train_test_split(data,test_size=0.2,random_state=42)
model = SVD(n_factors = 50,random_state =42)
model.fit(trainset)
predictions = model.test(testset)
rmse = accuracy.rmse(predictions)
movies = pd.read_csv(r"D:/datasets/ml-100k/u.item",sep="|",encoding="latin-1",header=None,usecols=[0,1],names=["item_id","title"])
@app.get("/recommend/{user_id}")
def get_top_n_recommendations(user_id : int,n : int = 5):
    if user_id not in ratings["user_id"].values:
        raise HTTPException(status_code=404,detail="user number {user_id} doesnt exist")
    watched = ratings[ratings["user_id"]==user_id]["item_id"].tolist()
    all_movies = movies["item_id"].unique()
    not_watched = [m for m in all_movies if m not in watched]
    predictions = []
    for movie_id in not_watched:
        pred_rating = model.predict(user_id,movie_id).est
        predictions.append((movie_id,pred_rating))
    predictions.sort(key=lambda x: x[1],reverse=True)
    top_n = predictions[:n]
    recommendations = []
    for movie_id,rating in top_n:
        title = movies[movies["item_id"] == movie_id]["title"].values[0]
        recommendations.append({
            "movie_id" : int(movie_id),
            "title" : title,
            "predicted_rating" :rating
        } ) 
    return{
        "user_id" : user_id,
        "count" : len(recommendations),
        "recommendations" : recommendations
    }        


