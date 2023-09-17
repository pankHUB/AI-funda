import numpy as np
from sklearn.linear_model import LinearRegression

# **Data Collection**: We start by gathering height-weight data pairs
heights = np.array([160, 165, 170, 175, 180])
weights = np.array([60, 65, 70, 75, 80])

# **Model Selection**: We chose Linear Regression for its simplicity and power.
model  = LinearRegression()

# Data Pre-processing: Cleaned and split data for training/testing.
heights  = heights.reshape(-1,1)

# **Model Training**: Watch as the model adjusts its parameters to minimize errors.
model.fit(heights, weights)

# **Prediction**: With a trained model, we can predict weight from height.
new_height = 185
predict_weight = model.predict(np.array([[new_height]]))
print(f"Predicted weight for height {new_height} is {predict_weight[0]}")

