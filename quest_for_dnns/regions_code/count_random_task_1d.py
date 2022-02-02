import numpy as np
import tensorflow as tf
import pickle
import os
from tools import weight_initializer, bias_initializer, load_mnist, random_string
from tools import count_regions_1d, get_input_line

TRAIN = ('../datasets/mnist/train-images-idx3-ubyte',
         '../datasets/mnist/train-labels-idx1-ubyte')
TEST = ('../datasets/mnist/t10k-images-idx3-ubyte',
              '../datasets/mnist/t10k-labels-idx1-ubyte')
NETWORK = [32, 32, 32]
BIAS_STD = 0.001
REPEATS = 30
LR = 1e-3
NOISE_LEVELS = [0, 0.2, 0.4, 0.6, 0.8, 1.]
NOISE_TYPE = 'Y'  # 'X' or 'Y'
BATCH_SIZE = 128
NUM_SAMPLE = 100
REPORT_EPOCHS = [0, 1, 2, 5, 10, 20, 50, 100, 200]
HOME_DIR = './'
OUTPUT_BASE = 'memorize_1d'

output_dir = OUTPUT_BASE + "_network_%s_num_sample_%d_noise_type_%s_lr_%s_batch_size_%d_bias_std_%s_report_epochs_%s" %(
    str(NETWORK), NUM_SAMPLE, NOISE_TYPE, str(LR), BATCH_SIZE, str(BIAS_STD), str(REPORT_EPOCHS))
if output_dir not in os.listdir(HOME_DIR):
    os.makedirs(HOME_DIR + output_dir)

for noise_level in NOISE_LEVELS:
    if str(noise_level) not in os.listdir(HOME_DIR + output_dir):
        os.makedirs(HOME_DIR + output_dir + '/' + str(noise_level))

X_train, Y_train = load_mnist(TRAIN)
X_test, Y_test = load_mnist(TEST)
X_mean = np.mean(X_train, axis=0)
X_std = np.std(X_train, axis=0)
num_train = Y_train.shape[0]
num_test = Y_test.shape[0]

num_per_epoch = int(num_train / BATCH_SIZE)
report_batches = np.floor(num_per_epoch * np.array(REPORT_EPOCHS)).astype(int)

indices = np.random.choice(num_test, NUM_SAMPLE, replace=False)
samples = [X_test[index, :] for index in indices]
input_lines = []
for sample in samples:
    line_weight, line_bias = get_input_line(np.zeros_like(sample), sample)
    input_lines.append((line_weight, line_bias))


for repeats in range(REPEATS):
    for noise_level in NOISE_LEVELS:
        X_noise = np.copy(X_train)
        Y_noise = np.copy(Y_train)
        num_noise = int(noise_level * num_train)
        to_corrupt = np.random.choice(num_train, num_noise, replace=False)
        if NOISE_TYPE == 'X':
            noise = np.random.normal(loc=X_mean, scale=X_std, size=(num_noise, 784)).astype(np.float32)
            X_noise[to_corrupt] = noise
        elif NOISE_TYPE == 'Y':
            noise_labels = np.random.choice(10, num_noise, replace=True)
            Y_noise[to_corrupt] = noise_labels
        else:
            raise NotImplementedError
        print("Running on noise level: %s" %str(noise_level))
        tf.reset_default_graph()
        input_placeholder = tf.placeholder(tf.float32, (None, X_noise.shape[1]))
        output_placeholder = tf.placeholder(tf.int64, (None,))
        preacts = []
        relu = input_placeholder
        for width in NETWORK:
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
        biases = [b for b in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if b.name.endswith('bias:0')]
        train_losses = []
        train_accs = []
        test_losses = []
        test_accs = []
        counts = []
        all_weights = []
        all_biases = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            [the_weights, the_biases] = sess.run([weights, biases],
                                                 feed_dict={input_placeholder: X_test, output_placeholder: Y_test})
            count = [count_regions_1d(the_weights[:-1], the_biases[:-1], the_weights[-1], the_biases[-1],
                                      input_line_weight, input_line_bias, return_regions=False)
                     for (input_line_weight, input_line_bias) in input_lines]
            counts.append(count)
            all_weights.append(the_weights)
            all_biases.append(the_biases)
            [train_loss, train_acc] = sess.run([loss, acc], feed_dict={input_placeholder: X_noise,
                                                                       output_placeholder: Y_noise})
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            [test_loss, test_acc] = sess.run([loss, acc], feed_dict={input_placeholder: X_test,
                                                                     output_placeholder: Y_test})
            test_losses.append(test_loss)
            test_accs.append(test_acc)
            print("Initialization, train loss %.2f, train acc %.2f, test loss %.2f, test acc %.2f" % (
                train_loss, train_acc, test_loss, test_acc))
            batch_num = 1
            next_report = 1
            for i in range(1, REPORT_EPOCHS[-1] + 1):
                perm = np.random.permutation(num_train)
                X_perm = X_noise[perm, :]
                Y_perm = Y_noise[perm]
                for j in range(int(num_train / BATCH_SIZE)):
                    X_batch = X_perm[(j * BATCH_SIZE):((j + 1) * BATCH_SIZE), :]
                    Y_batch = Y_perm[(j * BATCH_SIZE):((j + 1) * BATCH_SIZE)]
                    _ = sess.run([train_op], feed_dict={input_placeholder: X_batch, output_placeholder: Y_batch})
                    if batch_num == report_batches[next_report]:
                        [train_loss, train_acc] = sess.run([loss, acc], feed_dict={input_placeholder: X_noise,
                                                                                   output_placeholder: Y_noise})
                        train_losses.append(train_loss)
                        train_accs.append(train_acc)
                        [test_loss, test_acc] = sess.run([loss, acc], feed_dict={input_placeholder: X_test,
                                                                                 output_placeholder: Y_test})
                        test_losses.append(test_loss)
                        test_accs.append(test_acc)
                        print("%.2f epochs, train loss %.2f, train acc %.2f, test loss %.2f, test acc %.2f" %(
                            REPORT_EPOCHS[next_report], train_loss, train_acc, test_loss, test_acc))
                        [the_weights, the_biases] = sess.run([weights, biases],
                                                             feed_dict={input_placeholder: X_test,
                                                                        output_placeholder: Y_test})
                        count = [count_regions_1d(the_weights[:-1], the_biases[:-1], the_weights[-1], the_biases[-1],
                                                  input_line_weight, input_line_bias, return_regions=False)
                                 for (input_line_weight, input_line_bias) in input_lines]
                        counts.append(count)
                        all_weights.append(the_weights)
                        all_biases.append(the_biases)
                        next_report += 1
                    batch_num += 1
            counts = np.array(counts)
        results = {'counts': counts,
                   'weights': all_weights,
                   'biases': all_biases,
                   'train_accs': train_accs,
                   'train_losses': train_losses,
                   'test_accs': test_accs,
                   'test_losses': test_losses}
        output = HOME_DIR + output_dir + '/' + str(noise_level) + '/' + random_string()
        with open(output, 'wb') as f:
            pickle.dump(results, f)



###################
# PREPARE FIGURES #
###################

import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

NETWORK = [32, 32, 32]
BIAS_STD = 0.001
REPEATS = 40
LR = 1e-3
NOISE_LEVELS = [0, 0.2, 0.4, 0.6, 0.8, 1.]
NOISE_TYPE = 'Y'  # 'X' or 'Y'
BATCH_SIZE = 128
NUM_SAMPLE = 100
REPORT_EPOCHS = [0, 1, 2, 5, 10, 20, 50, 100, 200]
HOME_DIR = './'
OUTPUT_BASE = 'memorize_1d'
COLORS = ['red', 'orange', 'gold', 'green', 'blue', 'violet', 'gray', 'pink', 'brown', 'magenta']

output_dir = OUTPUT_BASE + "_network_%s_num_sample_%d_noise_type_%s_lr_%s_batch_size_%d_bias_std_%s_report_epochs_%s" %(
    str(NETWORK), NUM_SAMPLE, NOISE_TYPE, str(LR), BATCH_SIZE, str(BIAS_STD), str(REPORT_EPOCHS))

all_results = {}
for noise_level in NOISE_LEVELS:
    all_results[str(noise_level)] = []
    for n in range(REPEATS):
        dir = HOME_DIR + output_dir + '/' + str(noise_level)
        files = os.listdir(dir)
        files = [file for file in files if file != '.DS_Store']
        with open(dir + '/' + files[n], 'rb') as f:
            results = pickle.load(f)
            all_results[str(noise_level)].append(results)


###########
# FIGURES #
###########

###########
# Plot the number of regions as a function of epoch.

NUM_REPORTS = len(REPORT_EPOCHS)
for n, noise_level in enumerate(NOISE_LEVELS):
    results = all_results[str(noise_level)][:REPEATS]
    counts = []
    for result in results:
        counts.append(np.mean(np.array(result['counts']), axis=1))
    mean = np.mean(np.array(counts), axis=0)
    std = np.std(np.array(counts), axis=0)
    plt.plot(REPORT_EPOCHS[:NUM_REPORTS], mean[:NUM_REPORTS], label=str(noise_level), c=COLORS[n], marker='.')
    plt.fill_between(REPORT_EPOCHS[:NUM_REPORTS],
                     mean[:NUM_REPORTS] - std[:NUM_REPORTS],
                     mean[:NUM_REPORTS] + std[:NUM_REPORTS],
                     color=COLORS[n], alpha=0.1)

plt.legend(loc='lower right', title='Noise level')
plt.xlabel('Epoch', size=20)
plt.ylabel('Number of regions', size=20)
plt.show()

###########

###########

NUM_REPORTS = len(REPORT_EPOCHS)
for n, noise_level in enumerate(NOISE_LEVELS):
    results = all_results[str(noise_level)][:REPEATS]
    counts = []
    accs = []
    for result in results:
        counts.append(np.mean(np.array(result['counts']), axis=1))
        accs.append(np.array(result['train_accs']))
    mean_count = np.mean(np.array(counts), axis=0)
    mean_acc = np.mean(np.array(accs), axis=0)
    plt.plot(mean_acc[:NUM_REPORTS], mean_count[:NUM_REPORTS], label=str(noise_level), c=COLORS[n], marker='.')

plt.legend(loc='upper right', title='Noise level')
plt.xlabel('Accuracy', size=20)
plt.ylabel('Number of regions', size=20)
plt.show()

###########