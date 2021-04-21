import numpy as np
import tensorflow as tf
#import cv2
#this file is used to debug why num?.py does not work.

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
	inp_x = tf.placeholder(tf.float32, shape = [None, 26, 14])
	inp_y = tf.placeholder(tf.float32, shape = [None, 10])

	w1 = _weight_variable([364, 10])

	e1 = tf.reshape(inp_x , [-1, 364])
	e5 = tf.matmul(e1, w1)

	cost = tf.reduce_sum(tf.square(inp_y - e5))

	train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for i in range(10):
			labely = np.zeros(10, float)
			labely[i] = 1.
			datx = [dset[i]]
			daty = [labely]
			_, cost = sess.run([train_step, cost], feed_dict={inp_x:datx, inp_y:daty})
			print(cost)
			#_ = sess.run(cost, feed_dict={inp_x:datx, inp_y:daty})
			#print(_.shape)

if __name__ == "__main__":
	d = dataset()
	train(d)


