
import numpy as np
import pandas as pd

mv = pd.read_csv("movies.csv")
rt = pd.read_csv("ratings.csv")

data=pd.merge(left=mv, right=rt, how="outer", on="movieId")

data=data.dropna()
data=data.drop("timestamp", axis=1)


mat1 = data.pivot(index='userId', columns='movieId', values='rating').fillna(0)



mat = mat1.as_matrix()

mean = np.mean(mat, axis=1)
mat_new = mat-mean.reshape(-1,1)

from scipy.sparse.linalg import svds
U, sigma, vt = svds(mat_new, k=50)
sigma = np.diag(sigma)

all_user_rating = np.dot(np.dot(U, sigma), vt) + mean.reshape(-1,1)

all_user_rating=pd.DataFrame(all_user_rating, columns=mat1.columns)


def predict(user):
    usernum=user-1
    per=all_user_rating.iloc[usernum,:].sort_values(ascending=False)
    per_m=pd.merge(left=mv, right=per, on='movieId', how='outer')
    index=data[data['userId']==user]['movieId']
    filter=per_m['movieId'].isin(index)
    per_m=per_m[~filter]
    return per_m

user=int(input('Enter User Id for which you want predictions\n'))
per_m=predict(user)
print('20 Recommended movies are\n')
per_m.rename(columns={(user-1):'rate'}, inplace=True)
per_m=per_m.sort_values('rate',ascending=False)
per_m=per_m.iloc[:20,:]
print(per_m)
