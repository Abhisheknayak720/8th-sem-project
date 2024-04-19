import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import mysql.connector

# MySQL database connection parameters
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Abhishek@18',
    'database': 'paintora'
}

size = 128

# Load image from database
def load_image_from_database():
    try:
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()
        query = "SELECT id, image FROM paintings WHERE theme IS NULL"
        cursor.execute(query)
        result = cursor.fetchone()
        if result:
            image_id, image_blob = result
            image_array = np.frombuffer(image_blob, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            return image, image_id
        else:
            return None, None
    except mysql.connector.Error as err:
        print("Error while connecting to MySQL:", err)
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

# Update theme in the database
def update_theme_in_database(image_id, theme):
    try:
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()
        query = "UPDATE paintings SET theme = %s WHERE id = %s"
        cursor.execute(query, (theme, image_id))
        connection.commit()
    except mysql.connector.Error as err:
        print("Error while updating theme in MySQL:", err)
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

# Load parameters
def load_parameters(filename):
    with open(filename, 'rb') as f:
        parameters = pickle.load(f)
    return parameters

# Check for each set of parameters
def check_image(image_try, parameters):
    A_final, _ = forward_propagate(image_try, parameters)
    return A_final[0][0]

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

# List of parameter files
parameter_files = ['trained_parameter_nature_and_others.pkl', 
                   'trained_parameter_desert_and_others.pkl', 
                   'trained_parameter_mountains_and_others.pkl', 
                   'trained_parameter_architecture_and_others.pkl']

# Load image from the database
image, image_id = load_image_from_database()

if image is not None and image_id is not None:
    image_plot = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize image
    image_resized = cv2.resize(image, (size, size))
    image_resized = np.array(image_resized)
    image_array = image_resized.reshape(1, -1).T
    image_try = image_array / 255.0

    # Check against each set of parameters
    category = ['nature','desert','mountain','architecture']
    i = 0
    for parameter_file in parameter_files:
        parameters = load_parameters(parameter_file)
        prediction = check_image(image_try, parameters)

        if prediction >= 0.5:
            theme = category[i]
            update_theme_in_database(image_id, theme)
            print("Image belongs to:", parameter_file.split('_')[2])
            plt.imshow(image_plot)
            plt.title(category[i]+' DETECTED!')
            break
        i += 1
    else:
        print("Image belongs to: Other paintings")
        plt.imshow(image_plot)
        plt.title('This is not any of the specific categories')

    plt.show()
else:
    print("No unlabeled images found in the database.")
