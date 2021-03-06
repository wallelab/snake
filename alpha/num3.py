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

def getdata(x,ny):
	x1 =   np.array([
			[x[0:7,0:3], x[0:7,1:4], x[0:7,2:5], x[0:7,3:6], x[0:7,4:7], x[0:7,5:8],
			 x[0:7,6:9], x[0:7,7:10], x[0:7,8:11], x[0:7,9:12], x[0:7,10:13], x[0:7,11:14]],
			[x[1:8,0:3], x[1:8,1:4], x[1:8,2:5], x[1:8,3:6], x[1:8,4:7], x[1:8,5:8],
			 x[1:8,6:9], x[1:8,7:10], x[1:8,8:11], x[1:8,9:12], x[1:8,10:13], x[1:8,11:14]],
			[x[2:9,0:3], x[2:9,1:4], x[2:9,2:5], x[2:9,3:6], x[2:9,4:7], x[2:9,5:8],
			 x[2:9,6:9], x[2:9,7:10], x[2:9,8:11], x[2:9,9:12], x[2:9,10:13], x[2:9,11:14]],
			[x[3:10,0:3], x[3:10,1:4], x[3:10,2:5], x[3:10,3:6], x[3:10,4:7], x[3:10,5:8],
			 x[3:10,6:9], x[3:10,7:10], x[3:10,8:11], x[3:10,9:12], x[3:10,10:13], x[3:10,11:14]],
			[x[4:11,0:3], x[4:11,1:4], x[4:11,2:5], x[4:11,3:6], x[4:11,4:7], x[4:11,5:8],
			 x[4:11,6:9], x[4:11,7:10], x[4:11,8:11], x[4:11,9:12], x[4:11,10:13], x[4:11,11:14]],
			[x[5:12,0:3], x[5:12,1:4], x[5:12,2:5], x[5:12,3:6], x[5:12,4:7], x[5:12,5:8],
			 x[5:12,6:9], x[5:12,7:10], x[5:12,8:11], x[5:12,9:12], x[5:12,10:13], x[5:12,11:14]],
			[x[6:13,0:3], x[6:13,1:4], x[6:13,2:5], x[6:13,3:6], x[6:13,4:7], x[6:13,5:8],
			 x[6:13,6:9], x[6:13,7:10], x[6:13,8:11], x[6:13,9:12], x[6:13,10:13], x[6:13,11:14]],
			[x[7:14,0:3], x[7:14,1:4], x[7:14,2:5], x[7:14,3:6], x[7:14,4:7], x[7:14,5:8],
			 x[7:14,6:9], x[7:14,7:10], x[7:14,8:11], x[7:14,9:12], x[7:14,10:13], x[7:14,11:14]],
			[x[8:15,0:3], x[8:15,1:4], x[8:15,2:5], x[8:15,3:6], x[8:15,4:7], x[8:15,5:8],
			 x[8:15,6:9], x[8:15,7:10], x[8:15,8:11], x[8:15,9:12], x[8:15,10:13], x[8:15,11:14]],
			[x[9:16,0:3], x[9:16,1:4], x[9:16,2:5], x[9:16,3:6], x[9:16,4:7], x[9:16,5:8],
			 x[9:16,6:9], x[9:16,7:10], x[9:16,8:11], x[9:16,9:12], x[9:16,10:13], x[9:16,11:14]],
			[x[10:17,0:3], x[10:17,1:4], x[10:17,2:5], x[10:17,3:6], x[10:17,4:7], x[10:17,5:8],
			 x[10:17,6:9], x[10:17,7:10], x[10:17,8:11], x[10:17,9:12], x[10:17,10:13], x[10:17,11:14]],
			[x[11:18,0:3], x[11:18,1:4], x[11:18,2:5], x[11:18,3:6], x[11:18,4:7], x[11:18,5:8],
			 x[11:18,6:9], x[11:18,7:10], x[11:18,8:11], x[11:18,9:12], x[11:18,10:13], x[11:18,11:14]],
			[x[12:19,0:3], x[12:19,1:4], x[12:19,2:5], x[12:19,3:6], x[12:19,4:7], x[12:19,5:8],
			 x[12:19,6:9], x[12:19,7:10], x[12:19,8:11], x[12:19,9:12], x[12:19,10:13], x[12:19,11:14]],
			[x[13:20,0:3], x[13:20,1:4], x[13:20,2:5], x[13:20,3:6], x[13:20,4:7], x[13:20,5:8],
			 x[13:20,6:9], x[13:20,7:10], x[13:20,8:11], x[13:20,9:12], x[13:20,10:13], x[13:20,11:14]],
			[x[14:21,0:3], x[14:21,1:4], x[14:21,2:5], x[14:21,3:6], x[14:21,4:7], x[14:21,5:8],
			 x[14:21,6:9], x[14:21,7:10], x[14:21,8:11], x[14:21,9:12], x[14:21,10:13], x[14:21,11:14]],
			[x[15:22,0:3], x[15:22,1:4], x[15:22,2:5], x[15:22,3:6], x[15:22,4:7], x[15:22,5:8],
			 x[15:22,6:9], x[15:22,7:10], x[15:22,8:11], x[15:22,9:12], x[15:22,10:13], x[15:22,11:14]],
			[x[16:23,0:3], x[16:23,1:4], x[16:23,2:5], x[16:23,3:6], x[16:23,4:7], x[16:23,5:8],
			 x[16:23,6:9], x[16:23,7:10], x[16:23,8:11], x[16:23,9:12], x[16:23,10:13], x[16:23,11:14]],
			[x[17:24,0:3], x[17:24,1:4], x[17:24,2:5], x[17:24,3:6], x[17:24,4:7], x[17:24,5:8],
			 x[17:24,6:9], x[17:24,7:10], x[17:24,8:11], x[17:24,9:12], x[17:24,10:13], x[17:24,11:14]],
			[x[18:25,0:3], x[18:25,1:4], x[18:25,2:5], x[18:25,3:6], x[18:25,4:7], x[18:25,5:8],
			 x[18:25,6:9], x[18:25,7:10], x[18:25,8:11], x[18:25,9:12], x[18:25,10:13], x[18:25,11:14]],
			[x[19:26,0:3], x[19:26,1:4], x[19:26,2:5], x[19:26,3:6], x[19:26,4:7], x[19:26,5:8],
			 x[19:26,6:9], x[19:26,7:10], x[19:26,8:11], x[19:26,9:12], x[19:26,10:13], x[19:26,11:14]]
			])
	x2 = np.stack([x1,x1,x1,x1,x1,x1,x1,x1,x1,x1], axis=4)
	y1 = np.zeros(10, float)
	y1[ny] = 1.
	return (x2, y1)	

def _weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev = 0.01)
	return tf.Variable(initial)

def train(dset):
	inp_x = tf.placeholder(tf.float32, shape = [None, 20, 12, 7, 3, 10])
	inp_y = tf.placeholder(tf.float32, shape = [None, 10])

	w1 = _weight_variable([7, 3, 10])
	w2 = _weight_variable([160, 10])

	h_c1 = tf.abs(inp_x - w1)

	e1 = tf.reduce_sum(h_c1, axis=[3,4])
	e2 = tf.reshape(-e1, [-1, 20, 12, 10])
	e3 = tf.nn.max_pool(e2, ksize = [1, 5, 3, 1], strides = [1, 5, 3, 1], padding = "SAME")
	e4 = tf.reshape(e3, [-1, 160])
	e5 = tf.nn.relu(tf.matmul(tf.exp(e4), w2))

	cost = tf.reduce_sum(tf.square(inp_y - e5), axis = -1)

	train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

	(dat0, label0) = getdata(dset[0],0)
	(dat1, label1) = getdata(dset[1],1)
	datx = [dat0, dat1]
	daty = [label0, label1]

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for i in range(10):
			_, cost = sess.run([train_step, cost], feed_dict={inp_x:datx, inp_y:daty})
			print(cost)
			#_ = sess.run(cost, feed_dict={inp_x:datx, inp_y:daty})
			#print(_.shape)

if __name__ == "__main__":
	d = dataset()
	train(d)


