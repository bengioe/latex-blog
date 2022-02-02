import numpy as np
import tensorflow as tf
import pickle
import os
from tools import weight_initializer, bias_initializer, load_mnist, random_string, calculate_distances

TRAIN = ('../datasets/mnist/train-images-idx3-ubyte',
         '../datasets/mnist/train-labels-idx1-ubyte')
TEST = ('../datasets/mnist/t10k-images-idx3-ubyte',
        '../datasets/mnist/t10k-labels-idx1-ubyte')
NETWORKS = [[16, 16, 16, 16], [32, 32, 32, 32], [64, 64, 64, 64], [128, 128, 128, 128],
            [20, 20, 20], [40, 40, 40], [60, 60, 60], [80, 80, 80],
            [25, 25, 25, 25, 25], [50, 50, 50, 50, 50], [75, 75, 75, 75, 75], [100, 100, 100, 100, 100]]
BIAS_STD = 0.1
REPEATS = 2
LR = 1e-3
BATCH_SIZE = 128
EPOCHS = 20
REPORT_EPOCHS = [0, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 6, 8, 10, 15, 20]
NUM_SAMPLE = 10000
HOME_DIR = './'
OUTPUT_BASE = 'nearest_boundary'

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

def get_distances(sess, X, Y, weights, preacts, input_placeholder, output_placeholder):
    [the_weights, the_preacts] = sess.run([weights, preacts], feed_dict={input_placeholder: X, output_placeholder: Y})
    return calculate_distances(the_weights, the_preacts)

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
        selected_sample_test = np.random.choice(num_test, NUM_SAMPLE, replace=False)
        selected_sample_train = np.random.choice(num_train, NUM_SAMPLE, replace=False)
        X_sample_test = X_test[selected_sample_test, :]
        Y_sample_test = Y_test[selected_sample_test]
        X_sample_train = X_train[selected_sample_train, :]
        Y_sample_train = Y_train[selected_sample_train]
        X_mean = np.mean(X_train, axis=0)
        X_std = np.std(X_train, axis=0)
        X_random = np.random.normal(loc=X_mean, scale=X_std, size=(NUM_SAMPLE, 784)).astype(np.float32)
        Y_random = np.zeros(NUM_SAMPLE, dtype=np.int32)
        train_losses = []
        train_accs = []
        test_losses = []
        test_accs = []
        num_training = X_train.shape[0]
        dists_test = []
        dists_train = []
        dists_random = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            dists_test.append(get_distances(sess, X_sample_test, Y_sample_test, weights, preacts,
                                            input_placeholder, output_placeholder))
            dists_train.append(get_distances(sess, X_sample_train, Y_sample_train, weights, preacts,
                                             input_placeholder, output_placeholder))
            dists_random.append(get_distances(sess, X_random, Y_random, weights, preacts,
                                              input_placeholder, output_placeholder))
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
                        dists_test.append(get_distances(sess, X_sample_test, Y_sample_test, weights, preacts,
                                                        input_placeholder, output_placeholder))
                        dists_train.append(get_distances(sess, X_sample_train, Y_sample_train, weights, preacts,
                                                         input_placeholder, output_placeholder))
                        dists_random.append(get_distances(sess, X_random, Y_random, weights, preacts,
                                                          input_placeholder, output_placeholder))
                        next_report += 1
                    batch_num += 1
            dists_test = np.array(dists_test)
            dists_train = np.array(dists_train)
            dists_random = np.array(dists_random)
        results = {'dists_test': dists_test,
                   'dists_train': dists_train,
                   'dists_random': dists_random,
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

NETWORKS = [[16, 16, 16, 16], [32, 32, 32, 32], [64, 64, 64, 64], [128, 128, 128, 128],
            [20, 20, 20], [40, 40, 40], [60, 60, 60], [80, 80, 80],
            [25, 25, 25, 25, 25], [50, 50, 50, 50, 50], [75, 75, 75, 75, 75], [100, 100, 100, 100, 100]]
BIAS_STD = 0.001
#BIAS_STD = 0.1
REPEATS = 12
LR = 1e-3
BATCH_SIZE = 128
EPOCHS = 20
REPORT_EPOCHS = [0, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 6, 8, 10, 15, 20]
NUM_SAMPLE = 10000
COLORS = ['red', 'orange', 'gold', 'green', 'lime', 'cyan', 'blue', 'purple', 'magenta', 'brown', 'gray']

HOME_DIR = './'
OUTPUT_BASE = 'nearest_boundary'
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
# Plot the distribution of a single network for a single run during training.

SAMPLE_TYPE = 'random' # 'train', 'test', or 'random'
NETWORK = [128, 128, 128, 128]
RUN_NUM = 0

assert SAMPLE_TYPE in ['train', 'test', 'random'], 'Unknown sample type'
key = 'dists_' + SAMPLE_TYPE
results = all_results[str(NETWORK)][:REPEATS]
dists = np.log(results[RUN_NUM][key] * np.sum(NETWORK))
for i, epoch in enumerate(REPORT_EPOCHS):
    plt.hist(dists[i, :].ravel(), bins=50)
    plt.xlabel('log distance to nearest boundary')
    plt.ylabel('Number of sample points')
    plt.xlim(-12, 2)
    plt.ylim(0, 1000)
    plt.title('Epoch %s for network %s, ' %(str(epoch), str(NETWORK)))
    plt.show()


plt.style.use('seaborn-deep')
SAMPLE_TYPE = 'random' # 'train', 'test', or 'random'
NETWORK = [16, 16, 16, 16]
RUN_NUM = 0
PLOT_EPOCHS = [0, 6, -1]
COLORS = ['blue', 'red', 'gold']

assert SAMPLE_TYPE in ['train', 'test', 'random'], 'Unknown sample type'
key = 'dists_' + SAMPLE_TYPE
results = all_results[str(NETWORK)][:REPEATS]
dists = np.log(results[RUN_NUM][key] * np.sum(NETWORK))
data = []
labels = []
for i, epoch in enumerate(PLOT_EPOCHS):
    data.append(dists[i, :].ravel())
    labels.append('Epoch %d'%REPORT_EPOCHS[epoch])

plt.hist(data, bins=20, color=COLORS, label=labels)
plt.xlabel('log distance to nearest boundary', size=20)
plt.ylabel('Number of sample points', size=20)
plt.legend(loc='upper left')
plt.xlim(-8, 2)
plt.show()

###########

###########
# Plot the log distances for individual neurons during training.

SAMPLE_TYPE = 'random' # 'train', 'test', or 'random'
NETWORK = [128, 128, 128, 128]
RUN_NUM = 0
NUM_SAMPLES = 1000
NUM_REPORTS = 5

assert SAMPLE_TYPE in ['train', 'test', 'random'], 'Unknown sample type'
key = 'dists_' + SAMPLE_TYPE
results = all_results[str(NETWORK)][:REPEATS]
dists = np.log(results[RUN_NUM][key] * np.sum(NETWORK))
for i in range(NUM_SAMPLES):
    plt.plot(REPORT_EPOCHS[0:NUM_REPORTS], dists[0:NUM_REPORTS, i])

plt.xlabel('Epoch')
plt.ylabel('log distance to nearest boundary')
plt.show()

###########


###########
# Plot the distance times number of neurons as a function of epoch.

SAMPLE_TYPE = 'random' # 'train', 'test', or 'random'
NUM_REPORTS = len(REPORT_EPOCHS)

assert SAMPLE_TYPE in ['train', 'test', 'random'], 'Unknown sample type'
key = 'dists_' + SAMPLE_TYPE
for n, network in enumerate(NETWORKS):
    results = all_results[str(network)][:REPEATS]
    means = []
    for result in results:
        means.append(np.mean(result[key], axis=1))
    mean = np.mean(np.array(means), axis=0)
    plt.plot(REPORT_EPOCHS[:NUM_REPORTS], mean[:NUM_REPORTS] * np.sum(network), label=str(network), c=COLORS[n], marker='.')

plt.legend(loc='upper right')
plt.xlabel('Epoch', size=20)
plt.ylabel('Distance times number of neurons', size=20)
# plt.title('For random data')
plt.show()

###########

###########
# Plot the distance times the number of neurons against the test accuracy.

SAMPLE_TYPE = 'random' # 'train', 'test', or 'random'
USE_ONLY_FINAL = False

assert SAMPLE_TYPE in ['train', 'test', 'random'], 'Unknown sample type'
key = 'dists_' + SAMPLE_TYPE
for n, network in enumerate(NETWORKS):
    results = all_results[str(network)][:REPEATS]
    means = []
    accs = []
    for result in results:
        means.append(np.mean(result[key], axis=1))
        accs.append(result['test_accs'])
    mean = np.mean(np.array(means), axis=0)
    acc = np.mean(np.array(accs), axis=0)
    if USE_ONLY_FINAL:
        mean = mean[-1]
        acc = acc[-1]
    else:
        mean = mean[:]
        acc = acc[:]
    plt.plot(acc, mean * np.sum(network), label=str(network), c=COLORS[n], marker='.')
    print(acc[-1])

plt.legend(loc='upper left')
plt.xlabel('Test accuracy', size=20)
plt.ylabel('Distance to nearest boundary\ntimes number of neurons', size=20)
plt.show()

###########

###########
# Plot the distance against the number of neurons, between two curves.

CONSTANTS = [0.4, 2]
#CONSTANTS = [0.4, 1.5]
SAMPLE_TYPE = 'random' # 'train', 'test', or 'random'

assert SAMPLE_TYPE in ['train', 'test', 'random'], 'Unknown sample type'
key = 'dists_' + SAMPLE_TYPE
num_neurons = [np.sum(network) for network in NETWORKS]
min_num = min(num_neurons)
max_num = max(num_neurons)
num_range = np.arange(min_num, max_num + 1)
plt.plot(num_range, np.divide(CONSTANTS[0], num_range), c='black')
plt.plot(num_range, np.divide(CONSTANTS[1], num_range), c='black')
for n, network in enumerate(NETWORKS):
    results = all_results[str(network)][:REPEATS]
    means = []
    for result in results:
        means.append(np.mean(result[key], axis=1))
    mean = np.mean(np.array(means), axis=0)
    plt.plot(np.repeat(np.sum(network) + 5 * np.random.random(), len(mean)), mean, label=str(network),
             c=COLORS[n], marker=".")

plt.legend(loc='upper right')
plt.xlabel('Number of neurons', size=20)
plt.ylabel('Distance to nearest boundary', size=20)
plt.show()

###########