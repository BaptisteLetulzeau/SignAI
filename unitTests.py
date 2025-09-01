import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import unittest
from unittest.mock import patch, MagicMock

from MNIST import MNIST

def run_mnist_training():
    mnist = MNIST()
    
    # Charger et préprocesser
    (x_train, y_train), (x_test, y_test) = mnist.load_dataset()
    (x_train, y_train), (x_test, y_test) = mnist.preprocess_data(x_train, y_train, x_test, y_test)
    
    # Construire et entraîner
    mnist.build_model()
    mnist.train_model(x_train, y_train, x_test, y_test, epochs=15)
    
    # Évaluer
    test_accuracy, y_pred_classes, y_true_classes = mnist.evaluate_model(x_test, y_test)
    print(f"Test accuracy: {test_accuracy}")

if __name__ == "__main__":
    run_mnist_training()

################################################ Unit TESTS ################################################
class TestMNIST(unittest.TestCase):
    def setUp(self):
        self.mnist = MNIST()

        # Test Data
        self.sample_x_train = np.random.randint(0, 256, (100, 28, 28))
        self.sample_y_train = np.random.randint(0, 10, (100,))
        self.sample_x_test = np.random.randint(0, 256, (20, 28, 28))
        self.sample_y_test = np.random.randint(0, 10, (20,))
    
    def test_init(self):
        mnist = MNIST()
        self.assertIsNone(mnist.model)
        self.assertIsNone(mnist.history)
    
    # def test_load_dataset_mock_only(self):
    #     with patch('tensorflow.keras.datasets.mnist.load_data') as mock_load:
    #         mock_load.return_value = "mocked_data"
    #         result = self.mnist.load_dataset()
    #         mock_load.assert_called_once()
    #         self.assertEqual(result, "mocked_data")
    
    def test_build_model(self):
        model = self.mnist.build_model()
        
        self.assertIsNotNone(model)
        self.assertIsNotNone(self.mnist.model)
        
        # 3 Dense + 2 Dropout
        self.assertEqual(len(model.layers), 5)
    
    def test_train_model(self):
        (x_train, y_train), (x_test, y_test) = self.mnist.preprocess_data(
            self.sample_x_train, self.sample_y_train,
            self.sample_x_test, self.sample_y_test
        )
        
        self.mnist.build_model()
        
        history = self.mnist.train_model(x_train, y_train, x_test, y_test, epochs=2)
        
        self.assertIsNotNone(history)
        self.assertIsNotNone(self.mnist.history)
        
        expected_keys = ['loss', 'accuracy', 'val_loss', 'val_accuracy']
        for key in expected_keys:
            self.assertIn(key, history.history)
            self.assertEqual(len(history.history[key]), 2)

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    
    tf.get_logger().setLevel('ERROR')
    
    unittest.main(verbosity=2)