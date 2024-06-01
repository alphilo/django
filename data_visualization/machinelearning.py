# Importing necessary libraries
from django.shortcuts import render
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def machine_learning(request):
    # Sample dataset (features and target)
    x = np.array([[1], [2], [3], [4], [5]])  # Features (in this case, just one feature)
    y = np.array([2, 4, 6, 8, 10])            # Target

    # Splitting dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=12)
    print('x_train', X_train)
    print('y_train', y_train)
    print('x_test', X_test)
    print('y_test', y_test)

    # Creating and training the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    print('model', model)

    # Making predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluating the model
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)

    # Example of using the trained model for prediction
    new_data = np.array([[6], [7], [8], [9], [11]])
    predictions = model.predict(new_data)
    print("Predictions for new data:", predictions)
    return render(request, 'data_visualization/machine_learning.html', {'predictions': predictions})
