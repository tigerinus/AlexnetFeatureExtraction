"""Train Feature Extraction"""
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet

# settings
TEST_DATA_RATIO = 0.1
TRAIN_VALID_RATIO = 0.7

# Load traffic signs data.
with open('train.p', mode='rb') as fd:
    DATA = pickle.load(fd)

X = DATA['features']
Y = DATA['labels']

DATA_SIZE = len(X)
assert DATA_SIZE == len(Y)

# Split data into training and validation sets.
X_TRAIN_VALID, X_TEST, Y_TRAIN_VALID, Y_TEST = train_test_split(
    X, Y,
    test_size=TEST_DATA_RATIO,
    random_state=0
)

X_TRAIN, X_VALID, Y_TRAIN, Y_VALID = train_test_split(
    X_TRAIN_VALID, Y_TRAIN_VALID,
    train_size=TRAIN_VALID_RATIO
)

# Define placeholders and resize operation.
X = tf.placeholder(tf.float32, (None, 32, 32, 3))
RESIZED = tf.image.resize_images(X, (227, 227))

# Pass placeholder as first argument to `AlexNet`.
FC7 = AlexNet(RESIZED, feature_extract=True)

# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
FC7 = tf.stop_gradient(FC7)

# TODO: Add the final layer for traffic sign classification.

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.

# TODO: Train and evaluate the feature extraction model.
