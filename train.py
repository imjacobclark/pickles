import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

x = np.arange(0, 100, 1).reshape(-1, 1)
y = np.arange(1, 101, 1).reshape(-1, 1)

model = LinearRegression()

model.fit(x,y)

score = model.score(x, y)

with open("add-one-model.pkl", 'wb') as file:
  pickle.dump(model, file)