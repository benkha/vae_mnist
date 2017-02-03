import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.misc import imsave as ims
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets


def merge(images, size):
        h, w = images.shape[1], images.shape[2]
        img = np.zeros((h * size[0], w * size[1]))

        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j*h:j*h+h, i*w:i*w+w] = image

        return img

class VAE:

    def __init__(self):
        # Hyperparameters
        self.stddev = 0.02
        self.num_reparameters = 20
        self.batch_size = 100

        self.mnist = read_data_sets("MNIST_data/", one_hot=True)
        self.n_samples = self.mnist.train.num_examples

        self.images = tf.placeholder(tf.float32, [None, 784])
        self.images_2d = tf.reshape(self.images, [-1, 28, 28, 1])

        means, stddevs = self.encoder(self.images_2d)
        samples = tf.random_normal([self.batch_size, self.num_reparameters], 0, 1, dtype=tf.float32)

        guessed_z = means + (stddevs * samples)
        self.generated_images = self.decoder(guessed_z)
        generated_flat = tf.reshape(self.generated_images, [self.batch_size, 28*28])

        self.generation_loss = -tf.reduce_sum(self.images * tf.log(1e-8 + generated_flat) + (1-self.images) * tf.log(1e-8 + 1 - generated_flat),1)

        self.latent_loss = 0.5 * tf.reduce_sum(tf.square(means) + tf.square(stddevs) - tf.log(tf.square(stddevs)) - 1,1)
        self.cost = tf.reduce_mean(self.generation_loss + self.latent_loss)
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost)


    def encoder(self, x_in):
        with tf.variable_scope("encoder"):
            h_1 = tf.nn.relu(self.conv_2d(x_in, 1, 16, "e_h_1"))
            h_2 = tf.nn.relu(self.conv_2d(h_1, 16, 32, "e_h_2"))
            h2_flat = tf.reshape(h_2, [-1, 7 * 7 * 32])

            means = self.fully_connected(h2_flat, 7 * 7 * 32, self.num_reparameters, "means")
            stddevs = self.fully_connected(h2_flat, 7 * 7 * 32, self.num_reparameters, "stddevs")

            return means, stddevs

    def decoder(self, z_in):
        with tf.variable_scope("decoder"):
            d_matrix = self.fully_connected(z_in, self.num_reparameters, 7 * 7 * 32, "d_matrix")
            d_matrix_relu = tf.nn.relu(tf.reshape(d_matrix, [self.batch_size, 7, 7, 32]))
            h_1 = tf.nn.relu(self.conv_transpose(d_matrix_relu, [self.batch_size, 14, 14, 16],
                                                "d_h1"))
            h_2 = self.conv_transpose(h_1, [self.batch_size, 28, 28, 1], "d_h2")
            return tf.nn.sigmoid(h_2)


    def conv_2d(self, x_in, input_shape, output_shape, scope):
        with tf.variable_scope(scope):
            weights = tf.get_variable("weights",
                                      [5, 5, input_shape, output_shape],
                                      initializer=
                                      tf.truncated_normal_initializer(stddev=self.stddev))
            biases = tf.get_variable("biases", [output_shape],
                                     initializer=tf.constant_initializer(0.0))
            return tf.nn.conv2d(x_in, weights, strides=[1, 2, 2, 1], padding="SAME") + biases

    def conv_transpose(self, x_in, shape, scope):
        with tf.variable_scope(scope):
            weights = tf.get_variable("weights",
                                      [5, 5, shape[-1], x_in.get_shape()[-1]],
                                      initializer=
                                      tf.truncated_normal_initializer(stddev=self.stddev))
            biases = tf.get_variable("biases", [shape[-1]],
                                     initializer=tf.constant_initializer(0.0))
            conv_t = tf.nn.conv2d_transpose(x_in, weights, output_shape=shape, strides=[1, 2, 2, 1])
            return conv_t

    def fully_connected(self, x_in, input_shape, output_shape, scope):
        with tf.variable_scope(scope):
            matrix = tf.get_variable("matrix", [input_shape, output_shape],
                                     tf.float32,
                                     tf.random_normal_initializer(stddev=self.stddev))
            bias = tf.get_variable("bias", [output_shape],
                                   initializer=tf.constant_initializer(0.0))
            return tf.matmul(x_in, matrix) + bias



    def train(self):
        visualization = self.mnist.train.next_batch(self.batch_size)[0]
        reshaped_vis = visualization.reshape(self.batch_size,28,28)
        ims("results/base.jpg",merge(reshaped_vis[:64],[8,8]))
        # train
        saver = tf.train.Saver(max_to_keep=2)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(10):
                for idx in range(int(self.n_samples / self.batch_size)):
                    batch = self.mnist.train.next_batch(self.batch_size)[0]
                    _, gen_loss, lat_loss = sess.run((self.optimizer, self.generation_loss, self.latent_loss), feed_dict={self.images: batch})
                    # dumb hack to print cost every epoch
                    if idx % (self.n_samples - 3) == 0:
                        print("epoch {}: genloss {} latloss {}".format(epoch, np.mean(gen_loss), np.mean(lat_loss)))
                        saver.save(sess, os.getcwd()+"/training/train",global_step=epoch)
                        generated_test = sess.run(self.generated_images, feed_dict={self.images: visualization})
                        generated_test = generated_test.reshape(self.batch_size,28,28)
                        ims("results/"+str(epoch)+".jpg",merge(generated_test[:64],[8,8]))

model = VAE()
model.train()
