#coding=utf-8

import tensorflow as tf
import numpy as np

'''
Tools to build neural Q networks
'''
class Net:

	def __init__(self, opt, sess=None, name='net', optimizer=None):
		self.sess = sess
		self.params = []
		self.opt = opt
		# optimizer = tf.train.RMSPropOptimizer(learningRate, decay=0.95, epsilon=0.01, centered=True)
		self.optimizer = optimizer 
		with tf.variable_scope(name):
			self.buildForward() # set self.input and self.output
			if optimizer:
				loss = self.buildLoss()
				self.__buildGrad(loss)

	def forward(self, input_):
		return self.sess.run(self.output, feed_dict={self.input:input_})

	def trainStep(self, input_, targets, action):
		feed_dict = {self.input:input_, self.targets:targets, self.action:action}
		self.sess.run(self.applyGrads, feed_dict=feed_dict)

	def getParams(self):
		return self.sess.run(self.params)

	def setParams(self, params):
		feed_dict = {}
		for i in range(len(params)):
			feed_dict[self.newParams[i]] = params[i]
		self.sess.run(self.assignOps, feed_dict=feed_dict)

	def buildForward(self):
		self.buildStateAbstractionModule()
		self.buildOutputModule()
		self.__buildAssignOps()

	def buildStateAbstractionModule(self):
		convShape = self.opt['convShape'] # [height=84, width=84, histLen=4]
		convLayers = self.opt.get('convLayers', None)
		'''
		[
		[32, [8, 8], [4, 4]],
		[64, [4, 4], [2, 2]],
		[64, [3, 3], [1, 1]]
		]
		'''
		linearLayers = self.opt.get('linearLayers', None) # [512]
		
		inputSize = int(np.prod(convShape))
		self.input = tf.placeholder(tf.float32, [None, inputSize], name='input')
		lastOp = self.input
		if convLayers:
			lastOp = tf.reshape(lastOp ,[-1] + convShape)
			for i, layer in enumerate(convLayers):
				[outputSize, kernelSize, strides] = layer
				lastOp, W, b = Net.conv2d(lastOp, outputSize, kernelSize, strides, 
													activation=tf.nn.relu, name='conv'+str(i))
				self.params.append(W)
				self.params.append(b)
			lastOp = tf.reshape(lastOp, [-1, int(np.prod(lastOp.get_shape()[1:]))])
		if linearLayers:
			for i, outputSize in enumerate(linearLayers):
				lastOp, W, b = Net.linear(lastOp, outputSize, 
													activation=tf.nn.relu, name='lin'+str(i))
				self.params.append(W)
				self.params.append(b)
		self.output = lastOp

	def buildOutputModule(self):
		outputSize = self.opt['outputSize']
		dueling = self.opt.get('dueling', None)
		lastOp = self.output
		if dueling:
			duelA, W, b = Net.linear(lastOp, outputSize, name='duelA')
			self.params.append(W)
			self.params.append(b)
			duelV, W, b = Net.linear(lastOp, 1, name='duelV')
			self.params.append(W)
			self.params.append(b)
			lastOp = duelV + (duelA - tf.reduce_mean(duelA, 1, keep_dims=True))
		else:
			lastOp, W, b = Net.linear(lastOp, outputSize, name='actionValue')
			self.params.append(W)
			self.params.append(b)
		self.output = lastOp

	def __buildAssignOps(self):
		# 用于设置paras
		self.assignOps = []
		self.newParams = []
		for param in self.params:
			ph = tf.placeholder(tf.float32, param.get_shape().as_list())
			op = tf.assign(param, ph)
			self.newParams.append(ph)
			self.assignOps.append(op)

	def buildLoss(self):
		outputSize = self.opt['outputSize']
		clipDelta = self.opt.get('clipDelta', None)
		self.targets = tf.placeholder(tf.float32, [None], name='targets')
		self.action = tf.placeholder(tf.int32, [None], name='action')
		actionOneHot = tf.one_hot(self.action, outputSize, 1.0, 0.0)
		qOneHot = tf.reduce_sum(self.output * actionOneHot, 1)
		deltas = self.targets - qOneHot
		if clipDelta:
			deltasCliped = tf.clip_by_value(deltas, -clipDelta, clipDelta)
			loss = tf.reduce_sum(tf.square(deltasCliped) / 2 + (tf.abs(deltas) - tf.abs(deltasCliped)) * clipDelta)

		else:
			loss = tf.reduce_sum(tf.square(deltas) / 2)
		return loss

	def __buildGrad(self, loss):
		grads = self.optimizer.compute_gradients(loss, self.params)
		self.applyGrads = self.optimizer.apply_gradients(grads)

	##### STATIC UTILS #####

	@staticmethod
	def getWeights(shape, stddev=0.01, dtype=tf.float32, name='weights'):
		return tf.get_variable(name, shape, initializer=tf.random_uniform_initializer(-stddev, stddev))
		# return tf.Variable(tf.truncated_normal(shape, stddev=stddev, dtype=dtype), name=name)

	@staticmethod
	def getBias(shape, init=0.0, dtype=tf.float32, name='bias'):
		return tf.get_variable(name, shape, initializer=tf.constant_initializer(init))

	@staticmethod
	def conv2d(input_, outputSize, kernelSize, strides, stddev=None, activation=None, name='conv2d'):
		inputSize = input_.get_shape().as_list()[-1]
		kernelShape = [kernelSize[0], kernelSize[1], inputSize, outputSize]
		if stddev is None:
			stddev = 1.0 / float(np.sqrt(np.prod(kernelShape[:-1])))
		with tf.variable_scope(name):
			weight = Net.getWeights(kernelShape, stddev)
			bias = Net.getBias([outputSize]) # , stddev)
			strides = [1, strides[0], strides[1], 1]
			conv = tf.nn.conv2d(input_, weight, strides=strides, padding='SAME', data_format='NHWC')
			out = tf.nn.bias_add(conv, bias, 'NHWC')
			if activation:
				out = activation(out)
			return out, weight, bias

	@staticmethod
	def linear(input_, outputSize, stddev=None, activation=None, name='linear'):
		[batchSize, inputSize] = input_.get_shape().as_list()
		if stddev is None:
			stddev = 1.0 / float(np.sqrt(inputSize))
		with tf.variable_scope(name):
			weight = Net.getWeights([inputSize, outputSize], stddev)
			bias = Net.getBias([outputSize]) # , stddev)
			out = tf.matmul(input_, weight) + bias
			if activation:
				out = activation(out)
			return out, weight, bias


##################
#      TEST      #
##################

if __name__ == '__main__':
	from option import Option
	opt = Option('config.json')
	sess = tf.Session()
	# net
	optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-4, decay=0.95, epsilon=0.01, centered=True)
	opt['convShape'] = [opt['height'], opt['width'], opt['histLen']]
	opt['outputSize'] = 10
	opt['clipDelta'] = 1
	opt['dueling'] = True
	net = Net(opt, sess, optimizer=optimizer)
	sess.run(tf.global_variables_initializer())
	# input
	batchSize = 100
	inputSize = int(np.prod(opt['convShape']))
	input_ = np.random.randn(batchSize, inputSize)
	print 'input', input_.shape, np.mean(input_)
	target = np.random.randn(batchSize)
	action = np.random.randint(0, opt['outputSize'], [batchSize])
	# forward
	output = net.forward(input_)
	print 'output', output.shape, np.mean(output)
	# backward
	old_params = net.getParams()
	net.trainStep(input_, target, action)
	new_params = net.getParams()
	print 'params:'
	for i in range(len(new_params)):
		print np.mean(new_params[i]), np.mean(old_params[i])

