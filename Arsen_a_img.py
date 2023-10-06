
# Solution for task 2 (Image Classifier) of lab assignment for FDA SS23 by [keshishyan_a_img.py]
# imports here
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# define additional functions here
def preprocess(X_train, X_test, y_train):

    # check input data size
    assert X_train.shape[1:] == X_test.shape[1:], "Input data shape mismatch"

    # normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # convert y_train to binary labels
    y_train = np.where(y_train > 0, 1, 0)

    return X_train, X_test, y_train.ravel()


    # --------------------------
    # add your data preprocessing, model definition, training and prediction between these lines

def train_predict(X_train, y_train, X_test):

    # check that the input has the correct shape
    assert X_train.shape == (len(X_train), 6336)
    assert y_train.shape == (len(y_train), 1)
    assert X_test.shape == (len(X_test), 6336)

    # preprocess data
    X_train, X_test,y_train = preprocess(X_train, X_test,y_train)

    # split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # define model
    """
    https://github.com/MuhammedBuyukkinaci/TensorFlow-Binary-Image-Classification-using-CNN-s    
    
    ---Binary classification model for an image classification task
    ---Two dense layers with ReLU activation, batch normalization, and dropout with a sigmoid activation function

    First layer is a dense layer with 128 neurons
    
    Second layer is a batch normalization layer
    
    Third layer is a dropout layer with a rate of 0.5 to prevent overfitting
    
    Fourth layer is another dense layer with 64 neurons
    
    The fifth and final layer is layer with a single neuron 
    using sigmoid activation, used to output a probability
    between 0 and 1 for the binary classification problem

    """
    model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=X_train.shape[1:]),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),  #  dropout layer with rate 0.5
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),  #  dropout layer with rate 0.5
    tf.keras.layers.Dense(1, activation='sigmoid')
])

    # compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    model.compile(optimizer = optimizer, loss = loss_fn, metrics = ['accuracy'])

    try:
        # fit model on training data  ; 50 epochs
        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), verbose=2)
        
        # evaluate model on validation set
        val_loss, val_acc = model.evaluate(X_val, y_val)
        
        # make predictions on test set
        y_pred = model.predict(X_test)
        y_pred = np.where(y_pred > 0.5, 1, 0)        

    except Exception as e:
        print(f"An error occurred during the fit method call on the model object. The error message is: {str(e)}")
        return None


    print('Validation accuracy:', val_acc)

    # test that the returned prediction has correct shape
    assert y_pred.shape == (len(X_test),) or y_pred.shape == (len(X_test), 1)


    return y_pred.ravel()

if __name__ == "__main__":
    # load data (please load data like that and let every processing step happen **inside** the train_predict function)
    # (change path if necessary)
    X_train = pd.read_csv("C:\\Users\\HP\\OneDrive\\Desktop\\FDAAssignment\\fda_lab_ss23\\X_train.csv")
    y_train = pd.read_csv("C:\\Users\\HP\\OneDrive\\Desktop\\FDAAssignment\\fda_lab_ss23\\y_train.csv", usecols=[0])

    # please put everything that you want to execute outside the function here!
    # call train_and_evaluate function with X_train, y_train, X_train as arguments
y_pred = train_predict(X_train, y_train, X_train)

# print predictions
print(y_pred)
