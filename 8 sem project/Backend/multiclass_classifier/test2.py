import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import mysql.connector
from io import BytesIO

size = 128

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Forward activation function
def forward_activation(A_prev, W, b, activation):
    Z = np.dot(W, A_prev) + b
    linear_cache = (A_prev, W, b)
    
    if activation == "sigmoid":
        A = sigmoid(Z)
        activation_cache = Z
    elif activation == "relu":
        A = np.maximum(0, Z)
        activation_cache = Z
    
    assert(Z.shape == (W.shape[0], A_prev.shape[1]))
    
    cache = (linear_cache, activation_cache)
    
    assert(A.shape == (W.shape[0], A_prev.shape[1]))
    
    return A, cache

# Forward propagation function
def forward_propagate(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2
    
    for i in range(1, L):
        A_prev = A
        A, cache = forward_activation(A_prev, parameters["W" + str(i)], parameters["B" + str(i)], "relu")
        caches.append(cache)
    
    A_last, cache = forward_activation(A, parameters["W" + str(L)], parameters["B" + str(L)], "sigmoid")
    caches.append(cache)
    
    return A_last, caches

# Load parameters
def load_parameters(filename):
    with open(filename, 'rb') as f:
        parameters = pickle.load(f)
    return parameters

# Connect to MySQL database
def connect_to_mysql():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="Abhishek@18",
        database="paintora"
    )

# Fetch images with NULL theme from the database
def fetch_images_with_null_theme(connection):
    cursor = connection.cursor()
    cursor.execute("SELECT id, image FROM paintings WHERE theme IS NULL")
    # print (cursor.fetchall())
    return cursor.fetchall()

# Update theme for a painting in the database
def update_theme(connection, painting_id, theme):
    cursor = connection.cursor()
    cursor.execute("UPDATE paintings SET theme = %s WHERE id = %s", (theme, painting_id))
    connection.commit()

# Load image from bytes
def load_image_from_bytes(image_bytes):
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Check for each set of parameters
def check_image(image_try, parameters):
    A_final, _ = forward_propagate(image_try, parameters)
    return A_final[0][0]

# List of parameter files
parameter_files = ['trained_parameter_nature_and_others.pkl', 
                   'trained_parameter_desert_and_others.pkl', 
                   'trained_parameter_mountains_and_others.pkl', 
                   'trained_parameter_architecture_and_others.pkl']

# Categories
categories = ['nature', 'desert', 'mountain',"architecture"]

# Load parameters
parameters_list = [load_parameters(file) for file in parameter_files]

# Connect to MySQL
connection = connect_to_mysql()

# Fetch images with NULL theme
images = fetch_images_with_null_theme(connection)

# Process and classify images
for image_id, image_data in images:
    image_array = np.frombuffer(image_data, dtype=np.uint8)
    image = load_image_from_bytes(image_array)
    image_resized = cv2.resize(image, (size, size))
    image_try = image_resized.reshape(1, -1).T / 255.0
    
    theme = None
    for parameters, category in zip(parameters_list, categories):
        prediction = check_image(image_try, parameters)
        if prediction >= 0.5:
            theme = category
            break
    
    if theme is not None:
        update_theme(connection, image_id, theme)
        print(f"Image with ID {image_id} belongs to: {theme}")
    else:
        print(f"No matching category found for image with ID {image_id}")

# Close MySQL connection
connection.close()
