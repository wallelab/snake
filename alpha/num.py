import numpy as np
import tensorflow as tf
#import cv2


def dataset():
	with open("/tftpboot/cv/pics/number.dat","rb") as f:
		sz = f.read()
		dz = []
		for i in sz:
			dz.append(int(i))
		d = np.array(dz)
		d = d.reshape([10,24,12])
		dset = np.zeros([10,26,14], float)
		for i in range(10):
			dset[i,1:-1,1:-1] = d[i, :, :]/255.
	return dset

def _weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev = 0.01)
	return tf.Variable(initial)

def train(dset):
	input_x = tf.placeholder(tf.float32, shape = [26, 14])
	input_y = tf.placeholder(tf.float32, shape = [10])

	w1 = _weight_variable([10,7,3])

	e = np.zeros([10],float)
	for k in range(10):
		c = []
		for i in range(20):
			for j in range(12):
				c.append(-tf.reduce_sum(tf.square(input_x[i:i+7, j:j+3] - w1[k,:,:])))

		c1 = tf.reshape(c, [1, 20, 12, 1])
		c2 = tf.nn.max_pool(c1, ksize = [1, 5, 3, 1], strides = [1, 5, 3, 1], padding = "SAME")
		c3 = tf.reshape(c2,[16])
		c4 = c3[6]
		e[k] = c4
	e1 = tf.reshape(e, [10])
	e2 = tf.exp(e1)
	cost = tf.reduce_mean(tf.square(input_y - e2))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for i in range(100000):
			label = np.zeros([10], float)
			seq = i%10
			label[seq] = 1.
			_, cost = sess.run([train_step, cost], feed_dict={input_x:dset[seq],input_y:label})
			print(cost)

if __name__ == "__main__":
	d = dataset()
	train(d)


