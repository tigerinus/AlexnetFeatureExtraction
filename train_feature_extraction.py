"""Train Feature Extraction"""
import pickle
import tensorflow as tf
import pandas as pd

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from alexnet import AlexNet

# settings
TEST_DATA_RATIO = 0.1
TRAIN_VALID_RATIO = 0.7

MU = 0
SIGMA = 0.1

RATE = 0.001
EPOCHS = 20
BATCH_SIZE = 300

# Load sign names
SIGN_NAMES = pd.read_csv('signnames.csv')
N_CLASSES = SIGN_NAMES.shape[0]

# Load traffic signs data.
with open('train.p', mode='rb') as fd:
    DATA = pickle.load(fd)

X_DATA = DATA['features']
Y_DATA = DATA['labels']

DATA_SIZE = len(X_DATA)
assert DATA_SIZE == len(Y_DATA)

# Split data into training and validation sets.
X_TRAIN_VALID, X_TEST, Y_TRAIN_VALID, Y_TEST = train_test_split(
    X_DATA, Y_DATA,
    test_size=TEST_DATA_RATIO,
    random_state=0
)

X_TRAIN, X_VALID, Y_TRAIN, Y_VALID = train_test_split(
    X_TRAIN_VALID, Y_TRAIN_VALID,
    train_size=TRAIN_VALID_RATIO
)

N_TRAIN = len(Y_TRAIN)

# Define placeholders and resize operation.
KEEP_PROB = tf.placeholder(tf.float32)

X = tf.placeholder(tf.float32, (None, 32, 32, 3))
Y = tf.placeholder(tf.int32, (None))
ONE_HOT_Y = tf.one_hot(Y, N_CLASSES)

# Pass placeholder as first argument to `AlexNet`.
RESIZED = tf.image.resize_images(X, (227, 227))
FC7 = AlexNet(RESIZED, feature_extract=True)

# Dropout
FC7 = tf.nn.dropout(FC7, KEEP_PROB)

# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
FC7 = tf.stop_gradient(FC7)

# Add the final layer for traffic sign classification. Input = 4096. Output = n_classes.
FC8_W = tf.Variable(tf.truncated_normal(shape=(4096, N_CLASSES), mean=MU, stddev=SIGMA))
FC8_B = tf.Variable(tf.zeros(N_CLASSES))
LOGITS = tf.add(tf.matmul(FC7, FC8_W), FC8_B, name='logits')

# Define loss, training, accuracy operations.
CROSS_ENTROPY = tf.nn.softmax_cross_entropy_with_logits(labels=ONE_HOT_Y, logits=LOGITS)
LOSS_OPERATION = tf.reduce_mean(CROSS_ENTROPY)

OPTIMIZER = tf.train.AdamOptimizer(learning_rate=RATE)
TRAINING_OPERATION = OPTIMIZER.minimize(LOSS_OPERATION)

PREDICTION = tf.argmax(LOGITS, 1, name='prediction')
CORRECT_PREDICTION = tf.equal(PREDICTION, tf.argmax(ONE_HOT_Y, 1))
ACCURACY_OPERATION = tf.reduce_mean(
    tf.cast(CORRECT_PREDICTION, tf.float32), name='accuracy_operation'
)

# Train and evaluate the feature extraction model.

def evaluate(sess, x_data, y_data, batch_size, accuracy_operation):
    """ Evaluate Accuracy """
    total_accuracy = 0
    num_examples = len(x_data)

    for offset in range(0, num_examples, batch_size):
        batch_x, batch_y = x_data[offset:offset+batch_size], y_data[offset:offset+batch_size]
        accuracy = sess.run(
            accuracy_operation,
            feed_dict={X: batch_x, Y: batch_y, KEEP_PROB: 1.0}
        )
        total_accuracy += (accuracy * len(batch_x))

    return total_accuracy / num_examples

def train():
    """ Train """
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(EPOCHS):
            x_train, y_train = shuffle(X_TRAIN, Y_TRAIN)

            print("EPOCH {} - Training (BATCH_SIZE: {})".format(i+1, BATCH_SIZE))
            for offset in range(0, N_TRAIN, BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_x, batch_y = x_train[offset:end], y_train[offset:end]
                sess.run(TRAINING_OPERATION, feed_dict={X: batch_x, Y: batch_y, KEEP_PROB: 0.5})

            training_acccuracy = evaluate(sess, x_train, y_train, BATCH_SIZE, ACCURACY_OPERATION)
            print("Training Accuracy   = {:.3f}".format(training_acccuracy))
            validation_accuracy = evaluate(sess, X_VALID, Y_VALID, BATCH_SIZE, ACCURACY_OPERATION)
            print("Validation Accuracy = {:.3f}".format(validation_accuracy))
            print()

        saver.save(sess, './alexnet.ckp')
        print("Model saved")

def test():
    """ Final test """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('alexnet.ckp.meta')
        saver.restore(sess, tf.train.latest_checkpoint('.'))

        accuracy_operation = tf.get_default_graph().get_tensor_by_name("accuracy_operation:0")

        test_accuracy = evaluate(sess, X_TEST, Y_TEST, 100, accuracy_operation)
        print("Test Accuracy = {:.3f}".format(test_accuracy))

train()
test()
