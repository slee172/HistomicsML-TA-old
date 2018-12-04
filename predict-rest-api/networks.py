import numpy as np
import tensorflow as tf

class Network():

	def __init__(self):

		# set variables
		self.input_units = 64
		self.hidden_units = 128
		self.output_units = 2
		self.seed = 128
		self.epochs = 10
		self.batch_size = 32
		self.test_batch_size = 1000000
		self.learning_rate = 0.01
		self.checkpointIter = 0
		# self.sess = None
		self.x = None
		self.y = None
		self.logits = None
		self.weights = None
		self.biases = None
		self.hidden_layer = None



	def init_model(self):

		self.x = tf.placeholder(tf.float32, [None, self.input_units])
		self.y = tf.placeholder(tf.float32, [None, self.output_units])
		# define weights and biases of the neural network
		self.weights = {
			'hidden': tf.Variable(tf.random_normal([self.input_units, self.hidden_units], seed=self.seed)),
			'output': tf.Variable(tf.random_normal([self.hidden_units, self.output_units], seed=self.seed))
		}
		self.biases = {
			'hidden': tf.Variable(tf.random_normal([self.hidden_units], seed=self.seed)),
			'output': tf.Variable(tf.random_normal([self.output_units], seed=self.seed))
		}
		# create our neural networks computational graph
		self.hidden_layer = tf.add(tf.matmul(self.x, self.weights['hidden']), self.biases['hidden'])
		self.hidden_layer = tf.nn.relu(self.hidden_layer)
		self.logits = tf.matmul(self.hidden_layer, self.weights['output']) + self.biases['output']
		# define cost of our neural network
		self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y))
		# set the optimizer
		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
		self.init = tf.global_variables_initializer()
		self.saver = tf.train.Saver()

	def train_model(self, features, labels, classifier):

		# # define cost of our neural network
		# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y))
		# # set the optimizer
		# optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)
		# init = tf.global_variables_initializer()
		# saver = tf.train.Saver()
		# create a session, and run our neural network in the session.
		self.classifier = classifier
		# self.checkpointIter += 1

		with tf.Session() as sess:
			# self.sess = tf.InteractiveSession()
			sess.run(self.init)
			for epoch in range(self.epochs):
				# avg_cost = 0
				total_batch = int(features.shape[0]/self.batch_size)
				for i in range(total_batch):
					batch_x, batch_y = self.train_next_batch(self.batch_size, features, labels)
					_, c = sess.run([self.optimizer, self.cost], feed_dict = {self.x: batch_x, self.y: batch_y})
					# avg_cost += c / total_batch
				# print "Epoch:", (epoch+1), "cost =", "{:.5f}".format(avg_cost)
			save_path = self.saver.save(sess, "./checkpoints/" + self.classifier + ".ckpt")


	def predict_prob(self, features):

		with tf.Session() as sess:

			try:
				# saver = tf.train.Saver()
				# saver = tf.train.import_meta_graph(self.ModelDirectory + '.meta')
				self.saver.restore(sess, "./checkpoints/" + self.classifier + ".ckpt")
			except IOError:
				print(" ...")
			softmax_ = tf.nn.softmax(self.logits)
			# probabilities = logits
			total_batch = int(features.shape[0]/self.test_batch_size)
			prob = np.zeros((len(features), 2))
			for i in range(total_batch):
				start = self.test_batch_size * i
				end = start + self.test_batch_size
				batch_x = self.predict_next_batch(start, end, features)
				prob[start: end] = softmax_.eval({self.x: batch_x})

			start = self.test_batch_size * total_batch
			end = len(features)
			batch_x = self.predict_next_batch(start, end, features)
			prob[start: end] = softmax_.eval({self.x: batch_x})
			return prob[:, 1]

	def predict(self, features):

		with tf.Session() as sess:

			try:
				# saver = tf.train.Saver()
				# saver = tf.train.import_meta_graph(self.ModelDirectory + '.meta')
				self.saver.restore(sess, "./checkpoints/" + self.classifier + ".ckpt")
			except IOError:
				print(" ...")
			# logits = init_model()
			prediction=tf.argmax(self.logits, 1)
			total_batch = int(features.shape[0]/self.test_batch_size)
			pred = np.zeros((len(features), ))
			for i in range(total_batch):
				start = self.test_batch_size * i
				end = start + self.test_batch_size
				batch_x = self.predict_next_batch(start, end, features)
				pred[start: end] = sess.run(prediction, {self.x: batch_x})
			start = self.test_batch_size * total_batch
			end = len(features)
			batch_x = self.predict_next_batch(start, end, features)
			pred[start: end] = sess.run(prediction, {self.x: batch_x})
			return pred

	def train_next_batch(self, num, features, labels):

		idx = np.arange(0 , len(features))
		np.random.shuffle(idx)
		idx = idx[:num]
		data_shuffle = features[idx]
		labels_shuffle = labels[idx]
		return data_shuffle, labels_shuffle

	def predict_next_batch(self, start, end, features):
		data_shuffle = features[start: end]
		return data_shuffle
