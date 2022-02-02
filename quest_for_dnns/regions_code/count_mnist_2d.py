import numpy as np
import tensorflow as tf
import pickle
import os
from tools import weight_initializer, bias_initializer, load_mnist, random_string, get_sample_plane3, count_regions_2d

TRAIN = ('../datasets/mnist/train-images-idx3-ubyte',
         '../datasets/mnist/train-labels-idx1-ubyte')
TEST = ('../datasets/mnist/t10k-images-idx3-ubyte',
        '../datasets/mnist/t10k-labels-idx1-ubyte')
NETWORKS = [[16, 16, 16, 16], [32, 32, 32, 32], [64, 64, 64, 64], [20, 20, 20], [40, 40, 40], [60, 60, 60], [80, 80, 80]]
BIAS_STD = 0.001
REPEATS = 5
LR = 1e-3
BATCH_SIZE = 128
EPOCHS = 20
NUM_SAMPLE = 5
REPORT_EPOCHS = [0, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 6, 8, 10, 15, 20]
HOME_DIR = './'
OUTPUT_BASE = 'count_2d'

assert REPORT_EPOCHS[-1] == EPOCHS, 'REPORT_EPOCHS[-1] should equal EPOCHS'
output_dir = OUTPUT_BASE + "_epochs_%d_lr_%s_batch_size_%d_bias_std_%s_num_sample_%d_report_epochs_%s" %(
    EPOCHS, str(LR), BATCH_SIZE, str(BIAS_STD), NUM_SAMPLE, str(REPORT_EPOCHS))
if output_dir not in os.listdir(HOME_DIR):
    os.makedirs(HOME_DIR + output_dir)

for network in NETWORKS:
    if str(network) not in os.listdir(HOME_DIR + output_dir):
        os.makedirs(HOME_DIR + output_dir + '/' + str(network))

X_train, Y_train = load_mnist(TRAIN)
X_test, Y_test = load_mnist(TEST)
num_train = Y_train.shape[0]
num_test = Y_test.shape[0]

num_per_epoch = int(num_train / BATCH_SIZE)
report_batches = np.floor(num_per_epoch * np.array(REPORT_EPOCHS)).astype(int)

samples1 = np.random.choice(num_train, NUM_SAMPLE, replace=False)
samples2 = np.random.choice(num_train, NUM_SAMPLE, replace=False)
points1 = [X_train[sample1, :] for sample1 in samples1]
points2 = [X_train[sample2, :] for sample2 in samples2]
input_planes = []
for point1, point2 in zip(points1, points2):
    input_plane_weight, input_plane_bias, _, _ = get_sample_plane3(point1, point2)
    input_planes.append((input_plane_weight, input_plane_bias))

def count(sess, weights, biases, input_placeholder, output_placeholder):
    [the_weights, the_biases] = sess.run([weights, biases],
                                         feed_dict={input_placeholder: X_test, output_placeholder: Y_test})
    return [count_regions_2d(the_weights, the_biases, input_plane_weight, input_plane_bias,
                             input_vertices=np.array([[-20, -20], [20, -20], [20, 20], [-20, 20]]),) for
            (input_plane_weight, input_plane_bias) in input_planes]

for network in NETWORKS:
    for repeats in range(REPEATS):
        print("Running on network with architecture: %s" %str(network))
        tf.reset_default_graph()
        input_placeholder = tf.placeholder(tf.float32, (None, X_train.shape[1]))
        output_placeholder = tf.placeholder(tf.int64, (None,))
        preacts = []
        relu = input_placeholder
        for width in network:
            dense_layer = tf.layers.dense(relu, width, activation=None, use_bias=True,
                                          kernel_initializer=weight_initializer,
                                          bias_initializer=bias_initializer(BIAS_STD))
            preacts.append(dense_layer)
            relu = tf.nn.relu(dense_layer)
        output_layer = tf.layers.dense(relu, 10, activation=None, use_bias=True,
                                       kernel_initializer=weight_initializer, bias_initializer=bias_initializer(BIAS_STD))
        loss = tf.losses.sparse_softmax_cross_entropy(output_placeholder, output_layer)
        correct = tf.equal(output_placeholder, tf.argmax(tf.nn.softmax(output_layer), axis=-1))
        acc = 100 * tf.reduce_mean(tf.cast(correct, tf.float32))
        train_op = tf.train.AdamOptimizer(LR).minimize(loss)
        relu_states = [tf.greater(preact, tf.zeros_like(preact)) for preact in preacts]
        weights = [w for w in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if w.name.endswith('kernel:0')]
        weights = weights[:-1]
        biases = [b for b in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if b.name.endswith('bias:0')]
        biases = biases[:-1]
        train_losses = []
        train_accs = []
        test_losses = []
        test_accs = []
        num_training = X_train.shape[0]
        counts = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            counts.append(count(sess, weights, biases, input_placeholder, output_placeholder))
            [train_loss, train_acc] = sess.run([loss, acc], feed_dict={input_placeholder: X_train,
                                                                       output_placeholder: Y_train})
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            [test_loss, test_acc] = sess.run([loss, acc], feed_dict={input_placeholder: X_test,
                                                                     output_placeholder: Y_test})
            test_losses.append(test_loss)
            test_accs.append(test_acc)
            batch_num = 1
            next_report = 1
            for i in range(1, EPOCHS + 1):
                perm = np.random.permutation(num_training)
                X_perm = X_train[perm, :]
                Y_perm = Y_train[perm]
                for j in range(int(num_training / BATCH_SIZE)):
                    X_batch = X_perm[(j * BATCH_SIZE):((j + 1) * BATCH_SIZE), :]
                    Y_batch = Y_perm[(j * BATCH_SIZE):((j + 1) * BATCH_SIZE)]
                    _ = sess.run([train_op], feed_dict={input_placeholder: X_batch, output_placeholder: Y_batch})
                    if batch_num == report_batches[next_report]:
                        [train_loss, train_acc] = sess.run([loss, acc], feed_dict={input_placeholder: X_train,
                                                                                   output_placeholder: Y_train})
                        train_losses.append(train_loss)
                        train_accs.append(train_acc)
                        [test_loss, test_acc] = sess.run([loss, acc], feed_dict={input_placeholder: X_test,
                                                                                 output_placeholder: Y_test})
                        test_losses.append(test_loss)
                        test_accs.append(test_acc)
                        print("%.2f epochs, train loss %.2f, train acc %.2f, test loss %.2f, test acc %.2f" %(
                            REPORT_EPOCHS[next_report], train_loss, train_acc, test_loss, test_acc))
                        counts.append(count(sess, weights, biases, input_placeholder, output_placeholder))
                        next_report += 1
                    batch_num += 1
            counts = np.array(counts)
        results = {'counts': counts,
                   'train_accs': train_accs,
                   'train_losses': train_losses,
                   'test_accs': test_accs,
                   'test_losses': test_losses}
        output = HOME_DIR + output_dir + '/' + str(network) + '/' + random_string()
        with open(output, 'wb') as f:
            pickle.dump(results, f)




###################
# PREPARE FIGURES #
###################

import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

NETWORKS = [[16, 16, 16, 16], [32, 32, 32, 32], [64, 64, 64, 64], [20, 20, 20], [40, 40, 40], [60, 60, 60], [80, 80, 80]]
BIAS_STD = 0.001
REPEATS = 10
LR = 1e-3
BATCH_SIZE = 128
EPOCHS = 20
NUM_SAMPLE = 5
REPORT_EPOCHS = [0, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 6, 8, 10, 15, 20]
HOME_DIR = './'
OUTPUT_BASE = 'count_2d'
COLORS = ['red', 'orange', 'gold', 'green', 'cyan', 'blue', 'violet', 'gray', 'pink', 'brown', 'magenta']

assert REPORT_EPOCHS[-1] == EPOCHS, 'REPORT_EPOCHS[-1] should equal EPOCHS'
output_dir = OUTPUT_BASE + "_epochs_%d_lr_%s_batch_size_%d_bias_std_%s_num_sample_%d_report_epochs_%s" %(
    EPOCHS, str(LR), BATCH_SIZE, str(BIAS_STD), NUM_SAMPLE, str(REPORT_EPOCHS))

all_results = {}
for network in NETWORKS:
    all_results[str(network)] = []
    for n in range(REPEATS):
        dir = HOME_DIR + output_dir + '/' + str(network)
        files = os.listdir(dir)
        files = [file for file in files if file != '.DS_Store']
        with open(dir + '/' + files[n], 'rb') as f:
            results = pickle.load(f)
            all_results[str(network)].append(results)


###########
# FIGURES #
###########

###########
# Plot the number of regions over the square of the number of neurons as a function of epoch.

NUM_REPORTS = len(REPORT_EPOCHS)

for n, network in enumerate(NETWORKS):
    results = all_results[str(network)][:REPEATS]
    counts = []
    for result in results:
        counts.append(np.mean(np.array(result['counts']), axis=1))
    mean = np.mean(np.array(counts), axis=0) / (np.sum(network) ** 2)
    std = np.std(np.array(counts), axis=0) / (np.sum(network) ** 2)
    plt.plot(REPORT_EPOCHS[:NUM_REPORTS], mean[:NUM_REPORTS], label=str(network),
             c=COLORS[n], marker='.')
    plt.fill_between(REPORT_EPOCHS[:NUM_REPORTS],
                     mean[:NUM_REPORTS] - std[:NUM_REPORTS],
                     mean[:NUM_REPORTS] + std[:NUM_REPORTS],
                     color=COLORS[n], alpha=0.1)

plt.legend(loc='lower right')
plt.xlabel('Epoch', size=20)
plt.ylabel('Number of regions over\nsquared number of neurons', size=20)
plt.show()

###########

###########

NUM_REPORTS = len(REPORT_EPOCHS)

for n, network in enumerate(NETWORKS):
    results = all_results[str(network)][:REPEATS]
    counts = []
    accs = []
    for result in results:
        counts.append(np.mean(np.array(result['counts']), axis=1))
        accs.append(np.array(result['train_accs']))
    counts /= (np.sum(network) ** 2)
    mean_count = np.mean(np.array(counts), axis=0)
    mean_acc = np.mean(np.array(accs), axis=0)
    plt.plot(mean_acc[:NUM_REPORTS], mean_count[:NUM_REPORTS], label=str(network), c=COLORS[n], marker='.')

plt.legend(loc='lower left')
plt.xlabel('Accuracy', size=20)
plt.ylabel('Number of regions over\nsquared number of neurons', size=20)
plt.show()

###########