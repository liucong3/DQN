#coding=utf-8
import os
import tensorflow as tf
import numpy as np
from net import Net
from learn import RL, AtariRL
from buffer import ReplayBuffer, AtariBuffer

'''
1. use n-step, n is as large as possible, constrained by that the n-step span of the experience 
	contains only greedy step, no exploration step is allowed. n-step backup is only used as an
	upper bound, when n-step backup is no longer useful (when the Q value to backup is larger 
	than the n-step target), n is replaced by 1
2. Make the replay buffer smaller to allow more training to run in parallel, 
	decay the Q values of the states that are no seen for a long time.
'''

class FastBuffer(AtariBuffer):

	def __init__(self, opt):
		self.greedy_n_step = opt['greedy_n_step']
		AtariBuffer.__init__(self, opt)

	def append(self, state, prev_reward, terminal):
		AtariBuffer.append(self, state, prev_reward, terminal)
		if terminal:
			n = len(self) - 2
			cumulativeReward = 0
			prev_epsilon_state = len(self) - 1
			discount = self.discount
			while (n >= 0):
				o = self.buffer[n]
				if o['terminal']: break;
				cumulativeReward = o['reward'] + self.discount * cumulativeReward
				o['reward'] = cumulativeReward
				if self.greedy_n_step:
					o['next_state'] = prev_epsilon_state - n
				else:
					o['next_state'] = 1
				o['discount'] = discount
				discount *= self.discount
				if o['is_episode_step']:
					cumulativeReward = 0
					prev_epsilon_state = n
					discount = self.discount
					terminal = False
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
		batch['index'] = []
		while n > 0:
			k = np.random.randint(size - 1)
			o = self[k]
			if not o['terminal']:
				n -= 1
				batch['index'].append(k)
				batch['state'].append(o['state'])
				batch['reward'].append(o['reward'])
				batch['action'].append(o['action'])
				batch['discount'].append(o['discount'])
				o2 = self[k + o['next_state']]
				batch['terminal'].append(o2['terminal'])
				batch['next_state'].append(o2['state'])
		# format data
		batch['state'] = np.array(batch['state'])
		batch['reward'] = np.array(batch['reward']).astype(np.float)
		batch['discount'] = np.array(batch['discount']).astype(np.float)
		batch['terminal'] = np.array(batch['terminal']).astype(np.float)
		batch['next_state'] = np.array(batch['next_state'])
		return batch


class FastNet(Net):

	def __init__(self, opt, sess=None, name='net', optimizer=None):
		self.qDecay = opt['qDecay']
		Net.__init__(self, opt, sess=sess, name=name, optimizer=optimizer)

	def buildLoss(self):
		outputSize = self.opt['outputSize']
		clipDelta = self.opt.get('clipDelta', None)
		self.targets = tf.placeholder(tf.float32, [None], name='targets')
		self.action = tf.placeholder(tf.int32, [None], name='action')
		actionOneHot = tf.one_hot(self.action, outputSize, 1.0, 0.0)
		target = tf.tile(tf.reshape(self.targets, [-1,1]), [1, outputSize])
		self.deltas = (self.output - target) * actionOneHot + self.output * self.qDecay
		if clipDelta:
			deltasCliped = tf.clip_by_value(self.deltas, -clipDelta, clipDelta)
			loss = tf.reduce_sum(tf.square(deltasCliped) / 2 + (self.deltas - deltasCliped) * deltasCliped)
		else:
			loss = tf.reduce_sum(tf.square(self.deltas) / 2)
		return loss


class FastAtariRL(AtariRL):

	def __init__(self, opt, NetType=FastNet, BufferType=FastBuffer):
		self.greedy_n_step = opt['greedy_n_step']
		AtariRL.__init__(self, opt, NetType, BufferType)

	def trainStep(self):
		batch = self.replayBuffer.sample(self.batchSize)
		if batch:
			#print 'trainStep -- ' + time.ctime()
			q2Max = self.computeTarget(batch)
			target = batch['reward'] + q2Max * batch['discount'] * (1 - batch['terminal'])
			self.qNetwork.trainStep(batch['state'], target, batch['action'])
			if self.greedy_n_step:
				q = self.qNetwork.forward(batch['state'])
				for i in range(q.shape[0]):
					a = batch['action'][i]
					if q[i, a] > target[i]:
						k = batch['index'][i]
						self.replayBuffer[k]['next_state'] = 1
						self.replayBuffer[k]['discount'] = self.discount


if __name__ == '__main__':
	'''
	from option import Option
	import numpy as np
	opt = Option('config.json')
	opt = opt.copy()
	opt['bufSize'] = 10
	opt['histLen'] = 2
	buf = FastBuffer(opt)
	n = 0
	for i in range(4):
		n += 1
		buf.append(np.ones([1,1])*0.05*n*255, n, n-1, False, False)
	n += 1
	buf.append(np.ones([1,1])*0.05*n*255, n, n-1, False, True)
	for i in range(4):
		n += 1
		buf.append(np.ones([1,1])*0.05*n*255, n, n-1, False, False)
	n += 1
	buf.append(np.ones([1,1])*0.05*n*255, n, n-1, True, False)
	for i in range(len(buf)):
		print buf[i]

	print buf.sample(3)

	'''

	from option import Option
	import numpy as np
	opt = Option('config.json')
	from env import AtariEnv
	AtariEnv.create(opt)
	env = opt['env']
	if opt['savePath'] is None:
		opt['savePath'] = 'save_' + env
	# net
	trainer = FastAtariRL(opt, NetType=FastNet, BufferType=FastBuffer)
	if os.path.exists(opt['savePath']):
		trainer.load()
	else:
		os.makedirs(opt['savePath'])
	trainer.train(opt['trainSteps'])


