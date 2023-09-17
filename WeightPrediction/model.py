import numpy as np
from sklearn.linear_model import LinearRegression

# sample data
heights = np.array([160, 165, 170, 175, 180])
weights = np.array([60, 65, 70, 75, 80])

# linear regression model
model  = LinearRegression()

# reshape data
heights  = heights.reshape(-1,1)

# Train model
model.fit(heights, weights)

# make prediction
new_height = 185
predict_weight = model.predict(np.array([[new_height]]))
print(f"Predicted weight for height {new_height} is {predict_weight[0]}")

