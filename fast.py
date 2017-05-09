#coding=utf-8
import os
import tensorflow as tf
import numpy as np
from net import Net
from learn import RL, AtariRL
from buffer import ReplayBuffer, AtariBuffer

'''
LEARNING TO PLAY IN A DAY: FASTER DEEP REINFORCEMENT LEARNING BY OPTIMALITY TIGHTENING
This implementation has bugs to fix
'''

class FastBuffer(AtariBuffer):

	'''
	Sample K (boundSteps) states before and after each randomly selected state in order to calcuate the 
	upper and lower bound for the selected state.
	'''
	def __init__(self, opt):
		self.boundSteps = opt['boundSteps']
		AtariBuffer.__init__(self, opt)

	def append(self, state, prev_reward, terminal):
		AtariBuffer.append(self, state, prev_reward, terminal)
		if terminal:
			n = len(self) - 2
			cumulativeReward = 0
			while (n >= 0):
				o = self.buffer[n]
				if o['terminal']: break;
				cumulativeReward = o['reward'] + self.discount * cumulativeReward
				o['reward'] = cumulativeReward
				n -= 1

	def sample(self, n):
		size = len(self.buffer) - self.curEpisodeLen
		# return None if nothing to sample
		all_terminal = True
		for i in xrange(size - 1):
			if not self.buffer[i]['terminal']:
				all_terminal = False
				break
		if all_terminal: return None
		# sample
		batch = ReplayBuffer.observation([],[],[],[],[],[])
		batch['discount'] = []
		while n > 0:
			k = np.random.randint(size - 1)
			o = self[k]
			if not o['terminal']:
				n -= 1
				batch['state'].append(o['state'])
				batch['action'].append(o['action'])
				valid = True
				for i in range(self.boundSteps):
					valid = valid and (k - i - 1 > 0)
					if valid:
						o2 = self[k - 1 - i]
						valid =  not o2['terminal']
					batch['terminal'].append(valid)
					# thing appended is meaningless is the state is invalid
					batch['next_state'].append(o2['state'] if valid else o['state'])	
					batch['reward'].append(o2['reward'] - o['reward'] if valid else 0)
					batch['discount'].append(self.discount)
				valid = not self[k]['terminal']
				for i in range(self.boundSteps + 1):
					valid = valid and (k + i + 1 < len(self)) 
					if valid:
						o2 = self[k + i + 1]
						valid =  not o2['terminal']
					batch['terminal'].append(valid)
					# thing appended is meaningless is the state is invalid
					batch['next_state'].append(o2['state'] if valid else o['state'])	
					batch['reward'].append(o['reward'] - o2['reward'] if valid else 0) 
					batch['discount'].append(self.discount)
		# format data
		batch['state'] = np.array(batch['state'])
		batch['reward'] = np.array(batch['reward']).astype(np.float)
		batch['discount'] = np.array(batch['discount']).astype(np.float)
		batch['terminal'] = np.array(batch['terminal']).astype(np.float)
		batch['next_state'] = np.array(batch['next_state'])
		return batch


class FastNet(Net):

	def __init__(self, opt, sess=None, name='net', optimizer=None):
		self.penaltyForBounds = opt['penaltyForBounds']
		Net.__init__(self, opt, sess, name, optimizer)

	def buildLoss(self):
		outputSize = self.opt['outputSize']
		clipDelta = self.opt.get('clipDelta', None)
		self.targets = tf.placeholder(tf.float, [None], name='targets')
		self.action = tf.placeholder(tf.int32, [None], name='action')
		self.L = tf.placeholder(tf.float, [None], name='L') # lower bound
		self.U = tf.placeholder(tf.float, [None], name='U') # upper bound
		actionOneHot = tf.one_hot(self.action, outputSize, 1.0, 0.0)
		q_sa = tf.reduce_sum(self.output * actionOneHot, 1)
		deltasQ = self.targets - q_sa
		deltasL = tf.nn.relu(self.L - q_sa)
		deltasU = tf.nn.relu(q_sa - self.U)
		self.deltas = deltasQ + deltasL + deltasU
		# if clipDelta:
		# 	deltasCliped = tf.clip_by_value(self.deltas, -clipDelta, clipDelta)
		# 	deltasRatio = (deltasCliped / self.deltas)
		# 	deltasQCliped = deltasQ * deltasRatio
		# 	deltasLCliped = deltasL * deltasRatio
		# 	deltasUCliped = deltasU * deltasRatio
		# 	loss = tf.reduce_sum(tf.square(deltasQCliped) / 2 +  (deltasQ - deltasQCliped) * deltasQCliped + \
		# 						tf.square(deltasQCliped) / 2 +  (deltasQ - deltasQCliped) * deltasQCliped + \
		# 						tf.square(deltasUCliped) / 2 +  (deltasU - deltasUCliped) * deltasUCliped)
		# else:
		# 	loss = tf.reduce_sum(tf.square(self.deltas) / 2 + self.penaltyForBounds / 2 * (tf.square(deltasL) + tf.square(deltasU))) 
		loss = tf.reduce_sum(tf.square(self.deltas) / 2 + self.penaltyForBounds / 2 * (tf.square(deltasL) + tf.square(deltasU))) 
		return loss

	def trainStep(self, input_, targets, action, L, U):
		feed_dict = {self.input:input_, self.targets:targets, self.action:action, self.L:L, self.U:U}
		self.sess.run(self.applyGrads, feed_dict=feed_dict)

	def getDebugInfo(self, input_, targets, action, L, U):
		feed_dict = {self.input:input_, self.targets:targets, self.action:action, self.L:L, self.U:U}
		return self.sess.run((self.deltas, self.output, self.grads), feed_dict=feed_dict)


class FastAtariRL(AtariRL):

	def __init__(self, opt):
		self.boundSteps = opt['boundSteps']
		AtariRL.__init__(self, opt, NetType=FastNet, BufferType=FastBuffer)

	def __computeTargets(self, batch):
		valid = batch['terminal']
		valid = valid.reshape([-1, 2 * self.boundSteps])
		validU = valid[:,:self.boundSteps]
		validL = valid[:,self.boundSteps+1:]

		q2Max = self.computeTarget(batch)
		q2Max = q2Max.reshape([-1, 2 * self.boundSteps])
		U = q2Max[:,:self.boundSteps]
		L = q2Max[:,self.boundSteps+1:]
		discount = self.discount
		for i in range(self.boundSteps):
			L[:,i] *= discount
			discount *= self.discount
		rewards = batch['reward'].reshape([-1, 2 * self.boundSteps])
		U = U - rewards[:,:self.boundSteps]
		L = L + rewards[:,self.boundSteps:]
		discount = self.discount
		for i in range(self.boundSteps):
			U[:,-(i+1)] /= discount
			discount *= self.discount
		maxU = U.max()
		U = U * validU + maxU * (1 - validU)
		L = L * validL
		U = U.min(axis=1)
		L = L.max(axis=1)

		rewards = rewards[:,self.boundSteps]
		q2Max = q2Max[:,self.boundSteps].reshape(-1)
		valid = valid[:,self.boundSteps]
		target = rewards + q2Max * self.discount * (1 - valid)
		return batch['state'], target, batch['action'], L, U

	def trainStep(self):
		batch = self.replayBuffer.sample(self.batchSize)
		if batch:
			state, target, action, L, U = self.__computeTargets(batch)
			self.qNetwork.trainStep(state, target, action, L, U)


	def printDebugInfo(self):
		if not self.evalBatchSize: return
		batch = self.replayBuffer.sample(self.evalBatchSize)
		if not batch: return
		state, target, action, L, U = self.__computeTargets(batch)
		deltas, output, grads = self.qNetwork.getDebugInfo(state, target, action, L, U)
		params = self.qNetwork.getParams()
		RL.printDebugInfo4(self.debug, params, deltas, output, grads, self.evalBatchSize)



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
	trainer = FastAtariRL(opt)
	if os.path.exists(opt['savePath']):
		trainer.load()
	else:
		os.makedirs(opt['savePath'])
	trainer.train(opt['trainSteps'])
