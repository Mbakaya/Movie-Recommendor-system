import numpy as np
from lightm.datasets import fetch_movielens
from lightm import lightFM

data = fetch_movielens(min_rating=40)

print(repr(data['train']))
print(repr(data['test']))

model =lightFM(loss='wrap')

model.fit(data['train'],epochs=30,num_threads=2)


def movie_recommendor(model,data,user_ids):
n_users,n_items =data['train'].shape

for user_id in user_ids:
    known positives =data[item_labels]data['train'].tosr()[user_id].indices]
      #movies our model predicts they will like
        scores = model.predict(user_id, np.arange(n_items))
        #rank them in order of most liked to least
        top_items = data['item_labels'][np.argsort(-scores)]

        #print out the results
        print("User %s" % user_id)
        print("     Known positives:")

        for x in known_positives[:3]:
            print("        %s" % x)

        print("     Recommended:")

        for x in top_items[:3]:
            print("        %s" % x)

movie_recommendor(model, data, [3, 25, 450])
