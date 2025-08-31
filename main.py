import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Reproducible, very important if we want to reproduce a case
np.random.seed(42) # for NP (datas)
tf.random.set_seed(42) # for TF (masses...)

class MNIST:
    def __init__(self):
        self.model = None
        self.history = None

    def load_dataset(self):
        model = keras.datasets.mnist.load_data()
        print("Model imported !!!")
        return model
            
    def preprocess_data(self, x_train, y_train, x_test, y_test):            
        # Pixels are between 0 and 255 but neuronal network works with 0 and 1
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        
        # We resize model because mnist images are 28 x 28px
        # -1 to keep the number of samples
        x_train = x_train.reshape(-1, 28 * 28)
        x_test = x_test.reshape(-1, 28 * 28)
        
        # One hot for network 
        # Allows 10 exists instead of 1 ([0,0,0,1,0,0,0,0,0,0])
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)
        
        return (x_train, y_train), (x_test, y_test)
    
    def build_model(self):        
        self.model = keras.Sequential([
            # Entry
            layers.Dense(128, activation='relu', input_shape=(784,)),
            layers.Dropout(0.2),  # random deactivation
            
            # Hide
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            
            # Exit with 10 classes (one-hot)
            # softmax = final activation
            layers.Dense(10, activation='softmax')
        ])
        
        # adam = classic optimizer
        # categorical_crossentropy works with one-shot
        # accuracy = % of precision
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def train_model(self, x_train, y_train, x_test, y_test, epochs=10):        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)
        ]
        
        # Train
        self.history = self.model.fit(
            x_train, y_train,
            batch_size=128, # number of samples grouped
            epochs=epochs, # number of passages
            validation_data=(x_test, y_test),
            callbacks=callbacks,
            verbose=1 # progression bar
        )
        
        return self.history
    
    def evaluate_model(self, x_test, y_test):        
        test_loss, test_accuracy = self.model.evaluate(x_test, y_test, verbose=0)
        
        print(x_test)
        y_pred = self.model.predict(x_test)
        print(y_pred)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1) 
        
        return test_accuracy, y_pred_classes, y_true_classes

def main(): 
    # 1 - Load model
    # 2 - Resize data
    # 3 - Build the model
    # 4 - Train model
    # 5 - Evaluate model
    # 6 - Predictions ????

    mnist = MNIST()

    (x_train, y_train), (x_test, y_test) = mnist.load_dataset()
    # print(mnist.load_dataset())
    (x_train, y_train), (x_test, y_test) = mnist.preprocess_data(x_train, y_train, x_test, y_test)
    
    mnist.build_model()
    
    mnist.train_model(x_train, y_train, x_test, y_test, epochs=15)
    
    test_accuracy, y_pred_classes, y_true_classes = mnist.evaluate_model(x_test, y_test)
    print("Test accuracy:", test_accuracy)

    # Prediction ??

if __name__ == "__main__":
    main()