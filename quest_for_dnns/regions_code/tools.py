import numpy as np
## from keras.initializers import _compute_fans
from tensorflow.python.ops.random_ops import random_normal
from struct import unpack
from sonnet import AbstractModule, Linear
import tensorflow as tf
import pickle

# pylint: disable=unused-argument
def list_to_str(my_list):
    output = ''
    for n, obj in enumerate(my_list):
        if isinstance(obj, int) or isinstance(obj, float):
            output = output + str(obj)
        else:
            output = output + 'Conv' + str(obj[0])
            if obj[1]:
                output = output + ',Pool'
        if n < len(my_list) - 1:
            output = output + ','
    return output

def count_neurons(network, input_shape):
    if isinstance(network[0], int):
        return np.sum(network)
    else:
        result = 0
        shape = np.array(input_shape)
        for obj in network:
            shape[0:2] -= 2
            shape[2] = obj[0]
            result += np.prod(shape)
            if obj[1]:
                shape[0] = int(shape[0] / 2)
                shape[1] = int(shape[1] / 2)
        return result

def weight_initializer(shape, dtype, partition_info):
    fan_in = _compute_fans(shape)[0]
    return random_normal(shape, stddev=(np.sqrt(2. / fan_in)))

def get_weight_initializer(scale):
    return lambda shape, dtype, partition_info: random_normal(
        shape, stddev=scale * (np.sqrt(2. / _compute_fans(shape)[0])))

def bias_initializer(bias_std):
    return lambda shape, dtype, partition_info: random_normal(shape, stddev=bias_std)

def get_all_regions(sess, input_data, output_data, relu_states, input_placeholder, output_placeholder,
                    return_hashes=True):
    max_size = 500000
    if return_hashes:
        output = np.zeros((0,))
    else:
        output = np.zeros((0, sum([relu.get_shape().as_list()[1] for relu in relu_states])), dtype=np.bool)
    for i in range(int(np.ceil(input_data.shape[0] / max_size))):
        [states] = sess.run([relu_states],
                          feed_dict={input_placeholder: input_data[(i * max_size):((i + 1) * max_size), :],
                                     output_placeholder: output_data[(i * max_size):((i + 1) * max_size)]})
        states = np.concatenate(states, axis=1)
        if return_hashes:
            output = np.hstack((output, np.apply_along_axis(lambda row: hash(tuple(row)), 1, states)))
        else:
            output = np.concatenate((output, states), axis=0)
    return output

def load_mnist(filenames, flat=True):
    with open(filenames[0], "rb") as f:
        _, _, rows, cols = unpack(">IIII", f.read(16))
        if flat:
            X = np.fromfile(f, dtype=np.uint8).reshape(-1, 784) / 255.
        else:
            X = np.fromfile(f, dtype=np.uint8).reshape(-1, 28, 28, 1) / 255.
    with open(filenames[1], "rb") as f:
        _, _ = unpack(">II", f.read(8))
        Y = np.fromfile(f, dtype=np.int8).reshape(-1)
    return X, Y

def load_cifar(paths, flat=True):
    X = np.zeros((0, 32, 32, 3))
    Y = np.zeros((0,))
    for path in paths:
        with open(path, 'rb') as f:
            dict = pickle.load(f, encoding='bytes')
        X_path = dict[b'data'].reshape(-1, 3, 32, 32).astype(np.float64)
        X_path = np.swapaxes(X_path, 1, 3)
        X_path = np.swapaxes(X_path, 1, 2)
        X_path = X_path / 255.
        X_path = 2 * X_path - 1
        Y_path = np.array(dict[b'labels'])
        X = np.concatenate((X, X_path))
        Y = np.concatenate((Y, Y_path))
    if flat:
        X = X.reshape(-1, 3072)
    return X, Y

def random_string():
    return str(np.random.random())[2:]

class MLP(AbstractModule):
    def __init__(self, bias_std, lr, widths, output_size, name='mlp'):
        super(MLP, self).__init__(name=name)
        self.bias_std = bias_std
        self.lr = lr
        self.widths = widths
        self.output_size = output_size

    def _build(self, input_placeholder):
        preacts = []
        relu = input_placeholder
        for width in self.widths:
            dense_layer = Linear(width, initializers={'w': weight_initializer,
                                                      'b': bias_initializer(self.bias_std)})(relu)
            preacts.append(dense_layer)
            relu = tf.nn.relu(dense_layer)
        output_layer = Linear(self.output_size, initializers={'w': weight_initializer,
                                                              'b': bias_initializer(self.bias_std)})(relu)
        return output_layer, preacts

    def get_ops(self, output_layer, output_placeholder):
        loss = tf.losses.sparse_softmax_cross_entropy(output_placeholder, output_layer)
        correct = tf.equal(output_placeholder, tf.argmax(tf.nn.softmax(output_layer), axis=-1))
        acc = 100 * tf.reduce_mean(tf.cast(correct, tf.float32))
        train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
        return train_op, loss, acc

def calculate_distances(the_weights, the_preacts):
    distances = []
    depth = len(the_weights)
    for n in range(the_preacts[0].shape[0]):
        all_preacts = []
        all_lengths = []
        weight = the_weights[0][:, :]
        for k in range(depth):
            preact = the_preacts[k][n, :]
            all_preacts.append(preact)
            all_lengths.append(np.linalg.norm(weight, axis=0))
            mask = np.tile(np.greater(preact, 0).astype(np.int32), (weight.shape[0], 1))
            weight = np.multiply(weight, mask)
            if k < depth - 1:
                weight = np.dot(weight, the_weights[k + 1][:, :])
        all_preacts = np.concatenate(all_preacts, axis=0)
        all_lengths = np.concatenate(all_lengths, axis=0)
        distance = np.min(np.divide(np.abs(all_preacts), all_lengths))
        distances.append(distance)
    distances = np.array(distances)
    return distances

def get_input_line(point1, point2):
    input_line_weight = point2 - point1
    input_line_bias = point1
    return input_line_weight, input_line_bias

class LinearRegion1D:
    def __init__(self, param_min, param_max, fn_weight, fn_bias, next_layer_off):
        self._min = param_min
        self._max = param_max
        self._fn_weight = fn_weight
        self._fn_bias = fn_bias
        self._next_layer_off = next_layer_off

    def get_new_regions(self, new_weight_n, new_bias_n, n):
        weight_n = np.dot(self._fn_weight, new_weight_n)
        bias_n = np.dot(self._fn_bias, new_weight_n) + new_bias_n
        if weight_n == 0:
            min_image = bias_n
            max_image = bias_n
        elif weight_n >= 0:
            min_image = weight_n * self._min + bias_n
            max_image = weight_n * self._max + bias_n
        else:
            min_image = weight_n * self._max + bias_n
            max_image = weight_n * self._min + bias_n
        if 0 < min_image:
            return [self]
        elif 0 > max_image:
            self._next_layer_off.append(n)
            return [self]
        else:
            if weight_n == 0:
                return [self]
            else:
                preimage = (-bias_n) / weight_n
                next_layer_off0 = list(np.copy(self._next_layer_off))
                next_layer_off1 = list(np.copy(self._next_layer_off))
                if weight_n >= 0:
                    next_layer_off0.append(n)
                else:
                    next_layer_off1.append(n)
                region0 = LinearRegion1D(self._min, preimage, self._fn_weight, self._fn_bias, next_layer_off0)
                region1 = LinearRegion1D(preimage, self._max, self._fn_weight, self._fn_bias, next_layer_off1)
                return [region0, region1]

    def next_layer(self, new_weight, new_bias):
        self._fn_weight = np.dot(self._fn_weight, new_weight).ravel()
        self._fn_bias = (np.dot(self._fn_bias, new_weight) + new_bias).ravel()
        self._fn_weight[self._next_layer_off] = 0
        self._fn_bias[self._next_layer_off] = 0
        self._next_layer_off = []

    @property
    def max(self):
        return self._max

    @property
    def min(self):
        return self._min

    @property
    def fn_weight(self):
        return self._fn_weight

    @property
    def fn_bias(self):
        return self._fn_bias

    @property
    def next_layer_off(self):
        return self._next_layer_off

    @property
    def dead(self):
        return np.all(np.equal(self._fn_weight, 0))


def intersect_lines_2d(line_weight, line_bias, pt1, pt2):
    t = (np.dot(pt1, line_weight) + line_bias) / np.dot(pt1 - pt2, line_weight)
    return pt1 + t * (pt2 - pt1)

class LinearRegion2D:
    def __init__(self, fn_weight, fn_bias, vertices, next_layer_off):
        self._fn_weight = fn_weight
        self._fn_bias = fn_bias
        self._vertices = vertices
        self._num_vertices = len(vertices)
        self._next_layer_off = next_layer_off

    def get_new_regions(self, new_weight_n, new_bias_n, n):
        weight_n = np.dot(self._fn_weight, new_weight_n)
        bias_n = np.dot(self._fn_bias, new_weight_n) + new_bias_n
        vertex_images = np.dot(self._vertices, weight_n) + bias_n
        is_pos = (vertex_images > 0)
        is_neg = np.logical_not(is_pos)  # assumes that distribution of bias_n has no atoms
        if np.all(is_pos):
            return [self]
        elif np.all(is_neg):
            self._next_layer_off.append(n)
            return [self]
        else:
            pos_vertices = []
            neg_vertices = []
            for i in range(self._num_vertices):
                j = np.mod(i + 1, self._num_vertices)
                vertex_i = self.vertices[i, :]
                vertex_j = self.vertices[j, :]
                if is_pos[i]:
                    pos_vertices.append(vertex_i)
                else:
                    neg_vertices.append(vertex_i)
                if is_pos[i] == ~is_pos[j]:
                    intersection = intersect_lines_2d(weight_n, bias_n, vertex_i, vertex_j)
                    pos_vertices.append(intersection)
                    neg_vertices.append(intersection)
            pos_vertices = np.array(pos_vertices)
            neg_vertices = np.array(neg_vertices)
            next_layer_off0 = list(np.copy(self._next_layer_off))
            next_layer_off1 = list(np.copy(self._next_layer_off))
            next_layer_off0.append(n)
            region0 = LinearRegion2D(self._fn_weight, self._fn_bias, neg_vertices, next_layer_off0)
            region1 = LinearRegion2D(self._fn_weight, self._fn_bias, pos_vertices, next_layer_off1)
            return [region0, region1]

    def next_layer(self, new_weight, new_bias):
        self._fn_weight = np.dot(self._fn_weight, new_weight)
        self._fn_bias = np.dot(self._fn_bias, new_weight) + new_bias
        self._fn_weight[:, self._next_layer_off] = 0
        self._fn_bias[self._next_layer_off] = 0
        self._next_layer_off = []

    @property
    def vertices(self):
        return self._vertices

    @property
    def fn_weight(self):
        return self._fn_weight

    @property
    def fn_bias(self):
        return self._fn_bias

    @property
    def dead(self):
        return np.all(np.equal(self._fn_weight, 0))


def sort_regions_1d(regions):
    mins = np.array([region.min for region in regions])
    indices = np.argsort(mins)
    return [regions[index] for index in list(indices)]


def merge_regions_1d(region1, region2):
    assert region1.next_layer_off == region2.next_layer_off, "Regions could not be merged"
    return LinearRegion1D(region1.min, region2.max, region1.fn_weight, region1.fn_bias, region1.next_layer_off)


def consolidate_regions_1d(regions):
    if np.any([region.dead for region in regions]):
        sorted_regions = sort_regions_1d(regions)
        i = 0
        merges = 0
        while i < len(sorted_regions) - 1:
            if sorted_regions[i].dead and sorted_regions[i + 1].dead:
                region1 = sorted_regions[i]
                region2 = sorted_regions.pop(i + 1)
                sorted_regions[i] = merge_regions_1d(region1, region2)
                merges += 1
            else:
                i += 1
        print("Dead regions found, merged %d" %merges)
        return sorted_regions
    else:
        return regions


def count_regions_1d(the_weights, the_biases, last_weight, last_bias, input_line_weight, input_line_bias,
                     param_min=-np.inf, param_max=np.inf, return_regions=False, consolidate_dead_regions=False):
    regions = [LinearRegion1D(param_min, param_max, input_line_weight, input_line_bias, [])]
    depth = len(the_weights)
    for k in range(depth):
        for n in range(the_biases[k].shape[0]):
            new_regions = []
            for region in regions:
                new_regions = new_regions + region.get_new_regions(the_weights[k][:, n], the_biases[k][n], n)
            regions = new_regions
        for region in regions:
            region.next_layer(the_weights[k], the_biases[k])
        if consolidate_dead_regions:
            regions = consolidate_regions_1d(regions)
    for region in regions:
        region.next_layer(last_weight, last_bias)
    if consolidate_dead_regions:
        regions = consolidate_regions_1d(regions)
    if return_regions:
        return regions
    else:
        return len(regions)

def region_pts_1d(regions):
    xs = []
    ys = []
    for region in regions:
        if region.min == -np.inf:
            pass
        else:
            xs.append(region.min)
            ys.append(region.min * region.fn_weight + region.fn_bias)
    return (xs, ys)

def gradients_1d(regions):
    lengths = []
    gradients = []
    biases = []
    for region in regions:
        lengths.append(region.max - region.min)
        gradients = gradients + list(region.fn_weight)
        biases = biases + list(region.fn_bias)
    return {'lengths': lengths, 'gradients': gradients, 'biases': biases}

def count_regions_2d(the_weights, the_biases, input_fn_weight, input_fn_bias,
                     input_vertices=np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]),
                     return_regions=False, eval_bounded=False, consolidate_dead_regions=False):
    # Note: eval_bounded requires that input_vertices is a rectangle.
    regions = [LinearRegion2D(input_fn_weight, input_fn_bias, input_vertices, [])]
    depth = len(the_weights)
    for k in range(depth):
        for n in range(the_biases[k].shape[0]):
            new_regions = []
            for region in regions:
                new_regions = new_regions + region.get_new_regions(the_weights[k][:, n], the_biases[k][n], n)
            regions = new_regions
        for region in regions:
            region.next_layer(the_weights[k], the_biases[k])
    if consolidate_dead_regions:
        raise NotImplementedError
    if eval_bounded:
        bounded_regions = []
        unbounded_regions = []
        mins = np.min(input_vertices, axis=0)
        maxs = np.max(input_vertices, axis=0)
        for region in regions:
            verts = region.vertices
            if ((mins[0] == np.min(verts, axis=0)[0]) or (mins[1] == np.min(verts, axis=0)[1])
                or (maxs[0] == np.max(verts, axis=0)[0]) or (maxs[1] == np.max(verts, axis=0)[1])):
                unbounded_regions.append(region)
            else:
                bounded_regions.append(region)
    if return_regions:
        if eval_bounded:
            return bounded_regions, unbounded_regions
        else:
            return regions
    else:
        if eval_bounded:
            return len(bounded_regions), len(unbounded_regions)
        else:
            return len(regions)


def count_regions_2d_2(the_weights, the_biases, last_weight, last_bias, input_fn_weight, input_fn_bias,
                     input_vertices=np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]),
                     return_regions=False, eval_bounded=False, consolidate_dead_regions=False):
    # Note: eval_bounded requires that input_vertices is a rectangle.
    regions = [LinearRegion2D(input_fn_weight, input_fn_bias, input_vertices, [])]
    depth = len(the_weights)
    for k in range(depth):
        for n in range(the_biases[k].shape[0]):
            new_regions = []
            for region in regions:
                new_regions = new_regions + region.get_new_regions(the_weights[k][:, n], the_biases[k][n], n)
            regions = new_regions
        for region in regions:
            region.next_layer(the_weights[k], the_biases[k])
    for region in regions:
        region.next_layer(last_weight, last_bias)
    if consolidate_dead_regions:
        raise NotImplementedError
    if eval_bounded:
        bounded_regions = []
        unbounded_regions = []
        mins = np.min(input_vertices, axis=0)
        maxs = np.max(input_vertices, axis=0)
        for region in regions:
            verts = region.vertices
            if ((mins[0] == np.min(verts, axis=0)[0]) or (mins[1] == np.min(verts, axis=0)[1])
                or (maxs[0] == np.max(verts, axis=0)[0]) or (maxs[1] == np.max(verts, axis=0)[1])):
                unbounded_regions.append(region)
            else:
                bounded_regions.append(region)
    if return_regions:
        if eval_bounded:
            return bounded_regions, unbounded_regions
        else:
            return regions
    else:
        if eval_bounded:
            return len(bounded_regions), len(unbounded_regions)
        else:
            return len(regions)


def get_sample_plane(X, Y):
    vert0 = X[np.nonzero(Y == 0)[0][0], :]
    vert1 = X[np.nonzero(Y == 1)[0][0], :]
    vert2 = X[np.nonzero(Y == 2)[0][0], :]
    side0 = vert1 - vert2
    side1 = vert2 - vert0
    side2 = vert0 - vert1
    cos0 = -np.dot(side1, side2) / (np.linalg.norm(side1) * np.linalg.norm(side2))
    cos1 = -np.dot(side2, side0) / (np.linalg.norm(side2) * np.linalg.norm(side0))
    cos2 = -np.dot(side0, side1) / (np.linalg.norm(side0) * np.linalg.norm(side1))
    sin0 = np.sqrt(1 - cos0 ** 2)
    circumradius = 0.5 * np.linalg.norm(side0) / sin0  # law of sines
    # parallelogram with radius 0 as diagonal, sides along sides 1 and 2
    proj1 = (cos2 / sin0) * circumradius * (side1 / np.linalg.norm(side1))
    proj2 = (cos1 / sin0) * circumradius * (side2 / np.linalg.norm(side2))
    circumcenter = vert0 + proj1 - proj2
    square_center = circumcenter
    # scale so that the square is slightly bigger than the circumcircle
    square_vec_1 = (proj2 - proj1) * 1.25
    unit_vec_1 = square_vec_1 / np.linalg.norm(square_vec_1)
    square_vec_2 = (side2 - np.dot(side2, unit_vec_1) * unit_vec_1)
    square_vec_2 = square_vec_2 * (circumradius * 1.25 / np.linalg.norm(square_vec_2))
    sq_norm1 = np.linalg.norm(square_vec_1) ** 2
    sq_norm2 = np.linalg.norm(square_vec_2) ** 2
    x0 = np.dot(vert0 - circumcenter, square_vec_1) / sq_norm1
    y0 = np.dot(vert0 - circumcenter, square_vec_2) / sq_norm2
    x1 = np.dot(vert1 - circumcenter, square_vec_1) / sq_norm1
    y1 = np.dot(vert1 - circumcenter, square_vec_2) / sq_norm2
    x2 = np.dot(vert2 - circumcenter, square_vec_1) / sq_norm1
    y2 = np.dot(vert2 - circumcenter, square_vec_2) / sq_norm2
    return np.array([square_vec_1, square_vec_2]), circumcenter, [x0, x1, x2], [y0, y1, y2]


def get_sample_plane2(X, Y):
    sample0 = X[np.nonzero(Y == 0)[0][0], :]
    sample1 = X[np.nonzero(Y == 1)[0][0], :]
    norm0 = np.linalg.norm(sample0)
    norm1 = np.linalg.norm(sample1)
    if norm0 > norm1:
        vert1 = sample0
        vert2 = sample1
    else:
        vert1 = sample1
        vert2 = sample0
    square_vec_1 = vert1 * 1.25
    unit_vec_1 = square_vec_1 / np.linalg.norm(square_vec_1)
    square_vec_2 = (vert2 - np.dot(vert2, unit_vec_1) * unit_vec_1)
    square_vec_2 = square_vec_2 * (np.linalg.norm(square_vec_1) / np.linalg.norm(square_vec_2))
    sq_norm1 = np.linalg.norm(square_vec_1) ** 2
    sq_norm2 = np.linalg.norm(square_vec_2) ** 2
    x1 = np.dot(vert1, square_vec_1) / sq_norm1
    y1 = np.dot(vert1, square_vec_2) / sq_norm2
    x2 = np.dot(vert2, square_vec_1) / sq_norm1
    y2 = np.dot(vert2, square_vec_2) / sq_norm2
    return np.array([square_vec_1, square_vec_2]), np.zeros_like(vert1), [x1, x2], [y1, y2]


def get_sample_plane3(sample0, sample1):
    norm0 = np.linalg.norm(sample0)
    norm1 = np.linalg.norm(sample1)
    if norm0 > norm1:
        vert1 = sample0
        vert2 = sample1
    else:
        vert1 = sample1
        vert2 = sample0
    square_vec_1 = vert1 * 1.25
    unit_vec_1 = square_vec_1 / np.linalg.norm(square_vec_1)
    square_vec_2 = (vert2 - np.dot(vert2, unit_vec_1) * unit_vec_1)
    square_vec_2 = square_vec_2 * (np.linalg.norm(square_vec_1) / np.linalg.norm(square_vec_2))
    sq_norm1 = np.linalg.norm(square_vec_1) ** 2
    sq_norm2 = np.linalg.norm(square_vec_2) ** 2
    x1 = np.dot(vert1, square_vec_1) / sq_norm1
    y1 = np.dot(vert1, square_vec_2) / sq_norm2
    x2 = np.dot(vert2, square_vec_1) / sq_norm1
    y2 = np.dot(vert2, square_vec_2) / sq_norm2
    return np.array([square_vec_1, square_vec_2]), np.zeros_like(vert1), [x1, x2], [y1, y2]
