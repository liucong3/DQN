#coding=utf-8
import os
import tensorflow as tf
import numpy as np
from net import Net
from learn import RL, AtariRL
from buffer import ReplayBuffer, AtariBuffer

'''
reinforcement learning with auxiliary planing
'''

class AuxBuffer(AtariBuffer):

	'''
	Sample K actions.
	'''
	def __init__(self, opt):
		self.actionPredictions = opt['actionPredictions']
		AtariBuffer.__init__(self, opt)

	def append(self, state, prev_reward, terminal):
		AtariBuffer.append(self, state, prev_reward, terminal)
		if terminal:
			n = len(self) - 2
			curGoal = n
			while (n >= 0):
				o = self.buffer[n]
				if o['terminal']: break;
				o['goal_distance'] = curGoal - n
				if o['reward'] > 0:
					curGoal = n
				n -= 1


	def sample(self, n):
		size = len(self.buffer) - self.curEpisodeLen
		# return None if nothing to sample
		all_terminal = True
		for i in xrange(size - 1):
			if not self[i]['terminal']:
				all_terminal = False
				break
		if all_terminal: return None
		# sample
		batch = ReplayBuffer.observation([],[],[],[],[],[])
		batch['discount'] = []
		batch['goal'] = []
		for i in range(self.actionPredictions):
			batch['goal_action' + str(i)] = []
		while n > 0:
			k = np.random.randint(size - 1)
			o = self[k]
			if not o['terminal']:
				n -= 1
				batch['state'].append(o['state'])
				batch['reward'].append(o['reward'])
				batch['action'].append(o['action'])
				batch['discount'].append(self.discount)
				o2 = self[k + 1]
				batch['terminal'].append(o2['terminal'])
				batch['next_state'].append(o2['state'])
				# added
				o_goal = self[k + o['goal_distance']]
				batch['goal'].append(o_goal['state'])
				valid = True
				for i in range(self.actionPredictions):
					if not valid or i >= o['goal_distance'] or o['is_episode_step']:
						valid = False
						batch['goal_action' + str(i)].append(-1)
					else: batch['goal_action' + str(i)].append(self.buffer[k + i]['action'])
		# format data
		batch['state'] = np.array(batch['state'])
		batch['reward'] = np.array(batch['reward']).astype(np.float)
		batch['discount'] = np.array(batch['discount']).astype(np.float)
		batch['terminal'] = np.array(batch['terminal']).astype(np.float)
		batch['next_state'] = np.array(batch['next_state'])
		batch['goal'] = np.array(batch['goal'])
		return batch



class AuxNet(Net):

	def __init__(self, opt, sess=None, name='net', optimizer=None):
		self.auxiliaryTaskWeights = opt['auxiliaryTaskWeights']
		self.actionPredictions = opt['actionPredictions']
		self.batchSize = opt['batchSize']
		Net.__init__(self, opt, sess, name, optimizer)

	def getStateAbstract(self, input_):
		return self.sess.run(self.stateAbstract, feed_dict={self.input:input_})

	def buildLoss(self):
		outputSize = self.opt['outputSize']
		clipDelta = self.opt.get('clipDelta', None)
		self.targets = tf.placeholder(tf.float32, [None], name='targets')
		self.action = tf.placeholder(tf.int32, [None], name='action')
		actionOneHot = tf.one_hot(self.action, outputSize, 1.0, 0.0)
		q = tf.reduce_sum(self.output * actionOneHot, 1)
		self.deltas = self.targets - q
		loss = tf.reduce_sum(tf.square(self.deltas) / 2)
		# loss 2
		linearLayers = self.opt.get('linearLayers', None)
		output2, W2, b2 = Net.linear(self.stateAbstract, linearLayers[-1], activation=tf.nn.relu, name='output2')
		self.goal = tf.placeholder(tf.float32, [None, linearLayers[-1]], name='goal')
		self.params.append(W2)
		self.params.append(b2)
		loss2 = tf.losses.mean_squared_error(self.goal, output2, scope='loss2')
		# loss 3
		gru_forward, weightRZ, biasRZ, weightU, biasU = AuxNet.createGRU(outputSize, linearLayers[-1])
		self.params.append(weightRZ)
		self.params.append(biasRZ)
		self.params.append(weightU)
		self.params.append(biasU)
		import pdb; pdb.set_trace()
		shared_linear_forward, shared_linear_W, shared_linear_b = AuxNet.createLinear(linearLayers[-1], outputSize)
		self.params.append(shared_linear_W)
		self.params.append(shared_linear_b)

		self.goal_actions = []
		for i in range(self.actionPredictions):
			self.goal_actions.append(tf.placeholder(tf.float32, [None, outputSize], name='goal_action' + str(i)))
		x = tf.constant(0.0, shape=[self.batchSize, outputSize])
		h = self.stateAbstract
		output3 = []
		for i in range(self.actionPredictions):
			h = gru_forward(x, h)
			output3.append(shared_linear_forward(h))
			#x = tf.nn.softmax(output3[-1])
			x = self.goal_actions[i]

		loss3 = 0
		for i in range(self.actionPredictions):
			loss3 += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.goal_actions[i], logits=output3[i]))

		return loss + self.auxiliaryTaskWeights * (loss2 + loss3)

	@staticmethod
	def createGRU(inputSize, outputSize, name='gru'):
		with tf.variable_scope(name):
			stddev = 1.0 / float(np.sqrt(inputSize + outputSize))
			weightRZ = Net.getWeights([inputSize + outputSize, 2 * outputSize], stddev, name='weightRZ')
			biasRZ = Net.getBias([2 * outputSize], stddev, name='biasRZ')
			weightU = Net.getWeights([inputSize + outputSize, outputSize], stddev, name='weightU')
			biasU = Net.getBias([outputSize], stddev, name='biasU')
		def forward(x, h):
			xh = tf.concat([x, h], 1)
			rz = tf.nn.sigmoid(tf.matmul(xh, weightRZ) + biasRZ)
			r, z = tf.split(rz, [outputSize, outputSize], 1)
			u = tf.nn.tanh(tf.matmul(tf.concat([x, h * r], 1), weightU) + biasU)
			return z * h + (1 - z) * u
		return forward, weightRZ, biasRZ, weightU, biasU

	@staticmethod
	def createLinear(inputSize, outputSize, name='shared_linear'):
		with tf.variable_scope(name):
			stddev = 1.0 / float(np.sqrt(inputSize))
			W = Net.getWeights([inputSize, outputSize], stddev, name='W')
			b = Net.getBias([outputSize], stddev, name='b')
		def forward(input_):
			return tf.matmul(input_, W) + b
		return forward, W, b
		 

	def trainStep(self, input_, targets, action, goal, goal_actions):
		feed_dict = {self.input:input_, self.targets:targets, self.action:action, self.goal:goal}
		for i in range(self.actionPredictions):
			feed_dict[self.goal_actions[i]] = goal_actions[i]
		self.sess.run(self.applyGrads, feed_dict=feed_dict)

	def getDebugInfo(self, input_, targets, action, goal, goal_actions):
		feed_dict = {self.input:input_, self.targets:targets, self.action:action, self.goal:goal}
		for i in range(self.actionPredictions):
			feed_dict[self.goal_actions[i]] = goal_actions[i]
		return self.sess.run((self.deltas, self.output, self.grads), feed_dict=feed_dict)


class AuxAtariRL(AtariRL):

	def __init__(self, opt):
		self.actionPredictions = opt['actionPredictions']
		AtariRL.__init__(self, opt, NetType=AuxNet, BufferType=AuxBuffer)


	def getStateAbstract(self, state):
		state = state.reshape([-1, self.inputSize])
		return self.qTarget.getStateAbstract(state)

	@staticmethod
	def one_hot(x, size):
		batch_size = x.shape[0]
		y = np.zeros([batch_size, size])
		for i in range(batch_size):
			if x[i] != -1:
				y[i, x[i]] = 1.0
		return y

	def __computeTargets(self, batch):
		q2Max = self.computeTarget(batch)
		target = batch['reward'] + q2Max * batch['discount'] * (1 - batch['terminal'])
		goal = self.getStateAbstract(batch['goal'])
		goal_actions = []
		for i in range(self.actionPredictions):
			goal_actions.append(AuxAtariRL.one_hot(batch['goal_action' + str(i)], self.outputSize))
		return batch['state'], target, batch['action'], goal, goal_actions

	def trainStep(self):
		batch = self.replayBuffer.sample(self.batchSize)
		if batch:
			#print 'trainStep -- ' + time.ctime()
			state, target, action, goal, goal_actions = self.__computeTargets(batch)
			self.qNetwork.trainStep(state, target, action, goal, goal_actions)

	def printDebugInfo(self):
		if not self.evalBatchSize: return
		batch = self.replayBuffer.sample(self.evalBatchSize)
		if not batch: return
		state, target, action, goal, goal_actions = self.__computeTargets(batch)
		deltas, output, grads = self.qNetwork.getDebugInfo(state, target, action, goal, goal_actions)
		params = self.qNetwork.getParams()
		RL.printDebugInfo4(params, deltas, output, grads, self.evalBatchSize)



if __name__ == '__main__':
	from option import Option
	import numpy as np
	opt = Option('config.json')
	'''
	opt = opt.copy()
	opt['bufSize'] = 100
	opt['histLen'] = 2
	buf = FastBuffer(opt)
	for i in range(20):
		buf.append(np.ones([1,1])*0.05*(i+1)*255, (i+1), i, False)
	print buf.sample(20,3)
	'''
	from env import AtariEnv
	AtariEnv.create(opt)
	env = opt['env']
	if not opt.get('savePath', None):
		opt['savePath'] = 'save_' + env
	# net
	trainer = AuxAtariRL(opt)
	if os.path.exists(opt['savePath']):
		trainer.load()
	else:
		os.makedirs(opt['savePath'])
	trainer.train(opt['trainSteps'])


