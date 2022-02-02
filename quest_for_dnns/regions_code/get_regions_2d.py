import numpy as np
import tensorflow as tf
import pickle
import os
from tools import weight_initializer, bias_initializer, load_mnist, random_string
from tools import count_regions_2d_2, get_sample_plane

TRAIN = ('../datasets/mnist/train-images-idx3-ubyte',
         '../datasets/mnist/train-labels-idx1-ubyte')
TEST = ('../datasets/mnist/t10k-images-idx3-ubyte',
        '../datasets/mnist/t10k-labels-idx1-ubyte')
NETWORKS = [[8, 8, 8, 8, 8]]
BIAS_STD = 0.001
REPEATS = 5
LR = 1e-3
BATCH_SIZE = 128
REPORT_EPOCHS = [0]
HOME_DIR = './'
OUTPUT_BASE = 'get_regions_2d'

output_dir = OUTPUT_BASE + "_lr_%s_batch_size_%d_bias_std_%s_report_epochs_%s" %(
    str(LR), BATCH_SIZE, str(BIAS_STD), str(REPORT_EPOCHS))
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

fn_weight, fn_bias, _, _ = get_sample_plane(X_train, Y_train)
def get_count(sess, weights, biases, input_placeholder, output_placeholder):
    [the_weights, the_biases] = sess.run([weights, biases],
                                         feed_dict={input_placeholder: X_test, output_placeholder: Y_test})
    return count_regions_2d_2(the_weights[:-1], the_biases[:-1], the_weights[-1], the_biases[-1],
                            fn_weight, fn_bias, return_regions=True)


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
        output_layer = tf.layers.dense(relu, 10, activation=None, use_bias=True, kernel_initializer=weight_initializer,
                                       bias_initializer=bias_initializer(BIAS_STD))
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
        num_training = X_train.shape[0]
        counts = []
        all_regions = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            regions = get_count(sess, weights, biases, input_placeholder, output_placeholder)
            counts.append(len(regions))
            all_regions.append(regions)
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
            for i in range(1, REPORT_EPOCHS[-1] + 1):
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
                        regions = get_count(sess, weights, biases, input_placeholder, output_placeholder)
                        counts.append(len(regions))
                        all_regions.append(regions)
                        next_report += 1
                    batch_num += 1
            counts = np.array(counts)
        results = {'counts': counts,
                   'all_regions': all_regions,
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

NETWORKS = [[8, 8, 8, 8, 8]]
BIAS_STD = 0.001
REPEATS = 7
LR = 1e-3
BATCH_SIZE = 128
REPORT_EPOCHS = [0]
HOME_DIR = './'
OUTPUT_BASE = 'get_regions_2d'
COLORS = ['red', 'orange', 'gold', 'green', 'cyan', 'blue', 'violet', 'gray', 'pink', 'brown', 'magenta']

output_dir = OUTPUT_BASE + "_lr_%s_batch_size_%d_bias_std_%s_report_epochs_%s" %(
    str(LR), BATCH_SIZE, str(BIAS_STD), str(REPORT_EPOCHS))

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

RUN = 3
REPORT = 0

np.random.seed(10)
for network in NETWORKS:
    results = all_results[str(network)][RUN]
    all_regions = results['all_regions']
    fig, ax = plt.subplots()
    regions = all_regions[REPORT]
    print(results['counts'][REPORT])
    for region in regions:
        vertices = region.vertices
        _ = ax.fill(vertices[:, 1], -vertices[:, 0], c=np.random.rand(3, 1))
    plt.xticks([], [])
    plt.yticks([], [])
    ax.set_ylabel('Input dim 1', size=20)
    ax.set_xlabel('Input dim 2', size=20)
    ax.set_aspect('equal')
    plt.show()

###########

###########

import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
import pylab as pl
RUN = 3
REPORT = 0

np.random.seed(10)
for network in NETWORKS:
    results = all_results[str(network)][RUN]
    all_regions = results['all_regions']
    regions = all_regions[REPORT]
    print(results['counts'][REPORT])
    ax = a3.Axes3D(pl.figure())
    minimum = np.array([np.inf, np.inf, np.inf])
    maximum = np.array([-np.inf, -np.inf, -np.inf])
    for region in regions:
        vertices = region.vertices
        vertices = np.hstack((vertices, np.dot(vertices, region.fn_weight)[:, 0].reshape(-1, 1) + region.fn_bias[0]))
        print(vertices.shape)
        polygon = a3.art3d.Poly3DCollection([vertices])
        polygon.set_color(colors.rgb2hex(np.random.random(3)))
        minimum = np.minimum(np.min(vertices, axis=0), minimum)
        maximum = np.maximum(np.max(vertices, axis=0), maximum)
        ax.add_collection3d(polygon)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlim3d([minimum[0], maximum[0]])
    ax.set_ylim3d([minimum[1], maximum[1]])
    ax.set_zlim3d([minimum[2], maximum[2]])
    ax.set_xlabel('Input dim 1', size=20)
    ax.set_ylabel('Input dim 2', size=20)
    ax.set_zlabel('Function output', size=20)
    ax.view_init(elev=41., azim=18)
    plt.savefig("./high_res_3D_relu_regions.png", dpi=800, bbox_inches="tight")
    # pl.show()

###########

###########

import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
import pylab as pl
import scipy as sp

ax = a3.Axes3D(pl.figure())
for i in range(10000):
    vtx = sp.rand(3, 3)
    tri = a3.art3d.Poly3DCollection([vtx])
    tri.set_color(colors.rgb2hex(sp.rand(3)))
    tri.set_edgecolor('k')
    ax.add_collection3d(tri)


pl.show()

###########