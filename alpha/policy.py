import tensorflow as tf


C_WIDTH = 306
C_HEIGHT = 178
C_ANGLE = 72
C_LAYER = 5

class Network(object):
	def __init__(self, num_int_conv_layers=12):
		self.num_input_planes = C_LAYER
		self.k = 32
		self.num_int_conv_layers = num_int_conv_layers

		self.session = tf.Session()
		self.set_up_network()

	def set_up_network(self):
		# a global_step variable allows epoch counts to persist through multiple training sessions
		global_step = tf.Variable(0, name="global_step", trainable=False)
		self.x = tf.placeholder(tf.float32, [None, C_HEIGHT, C_WIDTH, self.num_input_planes])
		self.y = tf.placeholder(tf.float32, shape=[None, C_ANGLE])

		#convenience functions for initializing weights and biases
		def _weight_variable(shape):
			initial = tf.truncated_normal(shape, stddev = 0.01)
			return tf.Variable(initial)

		def _bias_variable(shape):
			initial = tf.constant(0.01, shape = shape)
			return tf.Variable(initial)

		def _conv2d(x, W):
			return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME")

		# initial conv layer is 5x5
		W_conv_init = _weight_variable([8, 8, self.num_input_planes, self.k])
		h_conv_init = tf.nn.relu(_conv2d(self.x, W_conv_init))

		# followed by a series of 3x3 conv layers
		W_conv_intermediate = []
		h_conv_intermediate = []
		_current_h_conv = h_conv_init
		for i in range(self.num_int_conv_layers):
			W_conv_intermediate.append(_weight_variable([4, 4, self.k, self.k]))
			h_conv_intermediate.append(tf.nn.relu(_conv2d(_current_h_conv, W_conv_intermediate[-1])))
			_current_h_conv = h_conv_intermediate[-1]

		W_conv_final = _weight_variable([1, 1, self.k, 1])
		b_conv_final = tf.Variable(tf.constant(0, shape=[C_WIDTH*C_HEIGHT*self.k], dtype=tf.float32))
		h_conv_final = _conv2d(h_conv_intermediate[-1], W_conv_final)

		h_fc1 = tf.reshape(h_conv_final, [-1, C_WIDTH*C_HEIGHT*self.k]) + b_conv_final

		W_fc2 = _weight_variable([C_WIDTH*C_HEIGHT*self.k, C_ANGLE])
		b_fc2 = _bias_variable([C_ANGLE])
		logits = tf.matmul(h_fc1, W_fc2) + b_fc2

		self.log_likelihood_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.y))

		self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.log_likelihood_cost, global_step=global_step)
		was_correct = tf.equal(tf.argmax(logits, 1), tf.argmax(self.y, 1))
		self.accuracy = tf.reduce_mean(tf.cast(was_correct, tf.float32))

		#weight_summaries = tf.summary.merge([
		#tf.summary.histogram(weight_var.name, weight_var)
		#for weight_var in [W_conv_init] +  W_conv_intermediate + [W_conv_final, b_conv_final]],
		#name="weight_summaries"
		#)
		#activation_summaries = tf.summary.merge([
		#tf.summary.histogram(act_var.name, act_var)
		#for act_var in [h_conv_init] + h_conv_intermediate + [h_conv_final]],
		#	name="activation_summaries"
		#)
		#saver = tf.train.Saver()

	def initialize_variables(self, save_file=None):
		self.session.run(tf.global_variables_initializer())
		if save_file is not None:
			self.saver.restore(self.session, save_file)

	def get_global_step(self):
		return self.session.run(self.global_step)

	def train(self, training_data, step):
		batch_x, batch_y = training_data
		_, accuracy, cost = self.session.run(
			[self.train_step, self.accuracy, self.log_likelihood_cost],
			feed_dict={self.x: batch_x, self.y: batch_y})
		print(step, accuracy, cost)
		#self.training_stats.report(accuracy, cost)

		#avg_accuracy, avg_cost, accuracy_summaries = self.training_stats.collect()
		#global_step = self.get_global_step()
		#print("Step %d training data accuracy: %g; cost: %g" % (global_step, avg_accuracy, avg_cost))
		#if self.training_summary_writer is not None:
		#	activation_summaries = self.session.run(
		#		self.activation_summaries,
		#		feed_dict={self.x: batch_x, self.y: batch_y})
		#	self.training_summary_writer.add_summary(activation_summaries, global_step)
		#	self.training_summary_writer.add_summary(accuracy_summaries, global_step)

	def load_variables(self):
		checkpoint = tf.train.get_checkpoint_state("saved_networks")
		if checkpoint and checkpoint.model_checkpoint_path:
			tf.train.saver().restore(self.session, checkpoint.model_checkpoint_path)

	def save_variables(self, save_file, step = 1):
		if save_file is not None:
			print("Saving checkpoint to %s" % save_file)
			tf.train.saver().save(self.session, save_file, step)


