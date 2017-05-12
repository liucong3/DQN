#coding=utf-8
import os
import tensorflow as tf
import numpy as np
from net import Net
from learn import RL, AtariRL
from buffer import Queue, ReplayBuffer, AtariBuffer

'''
1. use n-step, n is as large as possible, constrained by that the n-step span of the experience 
	contains only greedy step, no exploration step is allowed. n-step backup is only used as an
	upper bound, when n-step backup is no longer useful (when the Q value to backup is larger 
	than the n-step target), n is replaced by 1
2. Make the replay buffer smaller to allow more training to run in parallel, 
	decay the Q values of the states that are no seen for a long time.
'''

class EssentialBuffer:

	def __init__(self, maxSize):
		self.maxSize = int(maxSize)
		self.buffer = Queue(self.maxSize)

	def add(self, item):
		if len(self.buffer) == self.maxSize:
			self.buffer.dequeue()
		self.buffer.enqueue(item)

	def removeAt(self, indexes):
		if isinstance(indexes, (set, list, tuple)):
			indexes = list(indexes)
			indexes.sort(reverse=True)
			for index in indexes:
				self.removeAt(index)
		else:
			self.buffer[indexes] = self.buffer[-1]
			self.buffer.dequeue(-1)

	def __len__(self):
		return len(self.buffer)

	@staticmethod
	def holdStateInfo(buffer, index, histLen):
		info = [None] * histLen
		for i in range(histLen):
			if index - i < 0: break
			if buffer[index - i]['terminal']: break
			info[histLen - i - 1] = buffer[index - i]['state']
		return info

	@staticmethod
	def __getState(info, shape):
		histLen = len(info)
		state = np.zeros(shape)
		for i in range(histLen):
			if info[i] is None: continue
			state[:, i] = info[i].astype(np.float) / 255.0
		return state.reshape(-1)

	def getState(self, index, shape):
		item = self.buffer[index]
		o = item.copy()
		o['state'] = EssentialBuffer.__getState(item['state'], shape)
		o['next_state'] = EssentialBuffer.__getState(item['next_state'], shape)
		return o


class FastBuffer(AtariBuffer):

	def __init__(self, opt):
		self.greedy_n_step = opt['greedy_n_step']
		self.essential = EssentialBuffer(opt['bufSize']) if opt['essential'] else None
		AtariBuffer.__init__(self, opt)

	def __buildEssentialItem(self, i):
		eo = {}
		eo['state'] = EssentialBuffer.holdStateInfo(self.buffer, i, self.histLen)
		o = self.buffer[i]
		eo['reward'] = o['reward']
		eo['action'] = o['action']
		o2 = self.buffer[i + 1]
		eo['terminal'] = o2['terminal']
		eo['next_state'] = EssentialBuffer.holdStateInfo(self.buffer, i+1, self.histLen)
		return eo

	def _add_items_into_essential_buffer(self):
		episodeLen = self.episodeInfo[0]['episodeLen']
		for i in xrange(episodeLen - 1):
			o = self.buffer[i]
			assert not o['terminal']
			if o['is_episode_step']:
				eo = self.__buildEssentialItem(i)
				self.essential.add(eo)

	def append(self, state, prev_reward, terminal):
		if not self.essential is None:
			if len(self.buffer) + 1 >= self.size and len(self.episodeInfo) > 0:
				self._add_items_into_essential_buffer()
		AtariBuffer.append(self, state, prev_reward, terminal)
		if terminal:
			n = len(self) - 2
			cumulativeReward = 0
			prev_epsilon_state = len(self) - 1
			discount = self.discount
			greedy_n_step_count = 0
			while (n >= 0):
				o = self.buffer[n]
				if o['terminal']: break;
				cumulativeReward = o['reward'] + self.discount * cumulativeReward
				o['cumulative_reward'] = cumulativeReward
				if self.greedy_n_step:
					o['next_state'] = prev_epsilon_state - n
				else:
					o['next_state'] = 1
				if o['next_state'] > 1:
					greedy_n_step_count += 1
				o['discount'] = discount
				discount *= self.discount
				if o['is_episode_step']:
					cumulativeReward = 0
					prev_epsilon_state = n
					discount = self.discount
					terminal = False
				o['episodeInfo'] = self.episodeInfo[-1]
				n -= 1
			self.episodeInfo[-1]['greedy_n_step_count'] = greedy_n_step_count

	def printInfo(self):
		text = 'buf: [ '
		n = size = len(self.episodeInfo)
		ignoreMiddle = False
		def episodeInfo(i):
			return str(self.episodeInfo[i]['greedy_n_step_count']) + '/' + str(self.episodeInfo[i]['episodeLen'])
		if n > 10:
			ignoreMiddle = True
			n = 5
		for i in range(n):
			text += episodeInfo(i) + ' '
		if ignoreMiddle:
			text += '... '
			for i in range(n):
				text += episodeInfo(size - 1 - i) + ' '
		text += ']'
		if not self.essential is None:
			text += '   ebuf:' + str(len(self.essential))
		print text

	def sampleBuffer(self, batch, n):
		size = len(self.buffer) - self.curEpisodeLen
		while n > 0:
			k = np.random.randint(size - 1)
			o = self[k]
			if not o['terminal']:
				n -= 1
				batch['index'].append(k)
				batch['state'].append(o['state'])
				batch['reward'].append(o['cumulative_reward'])
				batch['action'].append(o['action'])
				batch['discount'].append(o['discount'])
				o2 = self[k + o['next_state']]
				batch['terminal'].append(o2['terminal'])
				batch['next_state'].append(o2['state'])

	def sampleEssential(self, batch, n):
		size = len(self.essential)
		shape = list(self.buffer[0]['state'].shape) + [self.histLen]
		while n > 0:
			k = np.random.randint(size)
			o = self.essential.getState(k, shape)
			n -= 1
			batch['index'].append(k - size) # negativa number for index in essential buffer
			batch['state'].append(o['state'])
			batch['reward'].append(o['reward'])
			batch['action'].append(o['action'])
			batch['discount'].append(self.discount)
			batch['terminal'].append(o['terminal'])
			batch['next_state'].append(o['next_state'])

	def getSampleNums(self, n):
		if self.essential is None:
			return n, 0
		n2 = n / 2
		if n2 > len(self.essential):
			n2 = len(self.essential)
		n1 = n - n2
		return n1, n2

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
		n1, n2 = self.getSampleNums(n)
		batch = ReplayBuffer.observation([],[],[],[],[],[])
		batch['discount'] = []
		batch['index'] = []
		self.sampleBuffer(batch, n1) 
		self.sampleEssential(batch, n2) 
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

	def report(self):
		RL.report(self)
		self.replayBuffer.printInfo()

	def trainStep(self):
		batch = self.replayBuffer.sample(self.batchSize)
		if batch:
			#print 'trainStep -- ' + time.ctime()
			qMax = self.computeTarget(batch['next_state'])
			target = batch['reward'] + qMax * batch['discount'] * (1 - batch['terminal'])
			self.qNetwork.trainStep(batch['state'], target, batch['action'])
			if self.greedy_n_step:
				qMax, targetQs, qs, action = self.computeTarget(batch['state'], getAll=True)
				indexesToRemove = set()
				for i in range(qMax.shape[0]):
					a = batch['action'][i]
					k = batch['index'][i]
					if k >= 0: # index in replay buffer
						q = min(targetQs[i, a], qs[i, a]) # to mitigate over-estimation
						if self.replayBuffer[k]['next_state'] > 1 and q > target[i]:						
							self.replayBuffer[k]['next_state'] = 1
							self.replayBuffer[k]['discount'] = self.discount
							self.replayBuffer[k]['cumulative_reward'] = self.replayBuffer[k]['reward']
							self.replayBuffer[k]['episodeInfo']['greedy_n_step_count'] -= 1
					else: # index in essential buffer
						# check and possibly remove an item from essential buffer
						if action[i] != a and qMax[i] > target[i]:
							k += len(self.replayBuffer.essential)
							indexesToRemove.add(k)
				if len(indexesToRemove) > 0:
					self.replayBuffer.essential.removeAt(indexesToRemove)


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


