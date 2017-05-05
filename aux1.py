#coding=utf-8
import os
import tensorflow as tf
import numpy as np
from net import Net
from learn import RL, AtariRL
from buffer import ReplayBuffer, AtariBuffer

'''
reinforcement learning with iterative predictive auxiliary tasks  
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
		batch['actions_to_goal'] = []
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
				actions_to_goal = [0] * self.actionPredictions
				for i in range(min(o['goal_distance'], self.actionPredictions)):
					actions_to_goal[i] = self.buffer[k + i]['action']
				batch['actions_to_goal'].append(actions_to_goal)
		# format data
		batch['state'] = np.array(batch['state'])
		batch['reward'] = np.array(batch['reward']).astype(np.float)
		batch['discount'] = np.array(batch['discount']).astype(np.float)
		batch['terminal'] = np.array(batch['terminal']).astype(np.float)
		batch['next_state'] = np.array(batch['next_state'])
		batch['goal'] = np.array(batch['goal'])
		batch['actions_to_goal'] = np.array(batch['actions_to_goal'])
		return batch



class AuxNet(Net):

	def __init__(self, opt, sess=None, name='net', optimizer=None):
		self.penaltyForBounds = opt['penaltyForBounds']
		Net.__init__(self, opt, sess, name, optimizer)

	def buildLoss(self):
		outputSize = self.opt['outputSize']
		clipDelta = self.opt.get('clipDelta', None)
		self.targets = tf.placeholder(tf.float32, [None], name='targets')
		self.action = tf.placeholder(tf.int32, [None], name='action')
		actionOneHot = tf.one_hot(self.action, outputSize, 1.0, 0.0)
		q = tf.reduce_sum(self.output * actionOneHot, 1)
		self.deltas = self.targets - q
		if clipDelta:
			deltasCliped = tf.clip_by_value(self.deltas, -clipDelta, clipDelta)
			#loss = tf.reduce_sum(tf.square(deltasCliped) / 2 + (tf.abs(self.deltas) - tf.abs(deltasCliped)) * clipDelta)
			loss = tf.reduce_sum(tf.square(deltasCliped) / 2 + (self.deltas - deltasCliped) * clipDelta)
		else:
			loss = tf.reduce_sum(tf.square(self.deltas) / 2)
		return loss

	def trainStep(self, input_, targets, action, L, U):
		feed_dict = {self.input:input_, self.targets:targets, self.action:action, self.L:L, self.U:U}
		self.sess.run(self.applyGrads, feed_dict=feed_dict)

	def getDebugInfo(self, input_, targets, action, L, U):
		feed_dict = {self.input:input_, self.targets:targets, self.action:action, self.L:L, self.U:U}
		return self.sess.run((self.deltas, self.output, self.grads), feed_dict=feed_dict)


class AuxAtariRL(AtariRL):

	def __init__(self, opt):
		self.boundSteps = opt['boundSteps']
		AtariRL.__init__(self, opt, NetType=AuxNet, BufferType=AuxBuffer)

	def __computeTargets(self, batch):
		valid = batch['terminal']
		valid = valid.reshape([-1, 2 * self.boundSteps])
		validU = valid[:,:self.boundSteps]
		validL = valid[:,self.boundSteps:]

		q2Max = self.computeTarget(batch)
		q2Max = q2Max.reshape([-1, 2 * self.boundSteps])
		U = q2Max[:,:self.boundSteps]
		L = q2Max[:,self.boundSteps:]
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


