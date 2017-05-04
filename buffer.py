#coding=utf-8

import numpy as np

class Queue:

	def __init__(self, size=1):
		self.data = [0] * size
		self.size = self.head = self.tail = 0

	def __len__(self):
		return self.size

	def __getitem__(self, index):
		if index < 0: index = self.size + index
		assert index < self.size, 'index:%d < self.size:%d' % (index, self.size)
		index = self.head + index
		if index >= len(self.data):
			index -= len(self.data)
		return self.data[index]

	def __str__(self):
		data = self.as_list()
		return str(data)

	# enqueue one or more data
	def enqueue(self, *data):
		self._ensureCapacity(self.size + len(data))
		self.size += len(data)
		for d in data:			
			self.data[self.tail] = d
			self.tail += 1
			if self.tail == len(self.data):
				self.tail = 0

	# pop a number (num) of elements from the front
	def dequeue(self, num=1):
		assert self.size >= num
		self.size -= num
		self.head += num
		if self.head >= len(self.data):
			self.head -= len(self.data)

	def clear(self):
		self.size = self.head = self.tail = 0

	def as_list(self, new_data=None):
		if not new_data:
			new_data = [0] * self.size
		section1 = len(self.data) - self.head
		if section1 > self.size:
			new_data[:self.size] = self.data[self.head:self.head+self.size]
		else:
			section2 = self.size - section1
			new_data[:section1] = self.data[self.head:]
			new_data[section1:self.size] = self.data[:section2]
		return new_data

	def _ensureCapacity(self, new_size):
		while new_size > len(self.data):
			self.data = self.as_list([0] * (2 * len(self.data)))
			self.head = 0
			self.tail = self.size
			#print 'new capacity:', len(self.data)


class ReplayBuffer:
	
	def __init__(self, opt):
		self.size = int(opt['bufSize'])
		self.discount = opt['discount']
		self.reset()

	def reset(self, size=None):
		self.size = size or self.size
		self.buffer = Queue(self.size)
		self.curEpisodeLen = 0
		self.episodeLens = Queue()

	@staticmethod
	def observation(state, action, reward, terminal, next_state, is_episode_step):
		return {'state':state, 'action':action, 'reward':reward, 'terminal':terminal, 'next_state':next_state, 'is_episode_step':is_episode_step}

	def append(self, state, prev_reward, terminal):
		if self.curEpisodeLen > 0:
			self.buffer[-1]['reward'] = prev_reward
		self.buffer.enqueue(ReplayBuffer.observation(state, None, None, terminal, None, None))
		if len(self.buffer) >= self.size and len(self.episodeLens) > 0:
			episodeLen = self.episodeLens[0]
			self.episodeLens.dequeue()
			self.buffer.dequeue(episodeLen)
		self.curEpisodeLen += 1
		if terminal:
			self.episodeLens.enqueue(self.curEpisodeLen)
			self.curEpisodeLen = 0

	def appendAction(self, action, is_episode_step):
		self.buffer[-1]['action'] = action
		self.buffer[-1]['is_episode_step'] = is_episode_step

	def __getitem__(self, index):
		return self.buffer[index]

	def sample(self, n, excludeCurEpisode=False):
		if  excludeCurEpisode:
			size = len(self.buffer) - self.curEpisodeLen
		else:
			size = len(self.buffer)
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
				batch['reward'].append(o['reward'])
				batch['action'].append(o['action'])
				batch['discount'].append(self.discount)
				o2 = self[k + 1]
				batch['terminal'].append(o2['terminal'])
				batch['next_state'].append(o2['state'])
		# format data
		batch['state'] = np.array(batch['state'])
		batch['reward'] = np.array(batch['reward']).astype(np.float)
		batch['discount'] = np.array(batch['discount']).astype(np.float)
		batch['terminal'] = np.array(batch['terminal']).astype(np.float)
		batch['next_state'] = np.array(batch['next_state'])
		return batch

	def __len__(self):
		return len(self.buffer)


class AtariBuffer(ReplayBuffer):
	
	def __init__(self, opt):
		self.histLen = int(opt['histLen'])
		ReplayBuffer.__init__(self, opt)

	def append(self, state, prev_reward, terminal):
		state = state.reshape(-1).copy().astype(np.uint8)
		ReplayBuffer.append(self, state, prev_reward, terminal)

	def _getState(self, index):
		if index < 0: index += len(self.buffer) 
		shape = list(self.buffer[0]['state'].shape) + [self.histLen]
		state = np.zeros(shape)
		for i in range(self.histLen):
			if index - i < 0: break
			if self.buffer[index - i]['terminal']: break
			state[:, self.histLen - i - 1] = self.buffer[index - i]['state'].astype(np.float) / 255.0
		return state.reshape(-1)

	def __getitem__(self, index):
		o = self.buffer[index].copy()
		o['state'] = self._getState(index)
		return o


##################
#      TEST      #
##################

if __name__ == '__main__':
	# Test Queue
	q = Queue(3)
	q.enqueue(*range(10))
	print q
	q.dequeue(7)
	print q
	q.enqueue(*range(9))
	print q
	for i in range(0, len(q)):
		print(q[i], q[-i-1])
	q.enqueue(*range(9,12))
	print q
	q.dequeue(5)
	print q
	while len(q):
		print q[0]
		q.dequeue()

	# Test AtariBuffer
	from option import Option
	opt = Option('config.json')
	opt['bufSize'] = 9
	opt['histLen'] = 2
	buffer = AtariBuffer(opt)
	for i in range(1,3):
		buffer.append(np.ones([1,1])*255/i, i, i-1, False)
		buffer.appendAction(0, True)
	buffer.append(np.ones([1,1])*255/3, 3, 3-1, True)
	buffer.appendAction(0, False)
	for i in range(4,10):
		buffer.append(np.ones([1,1])*255/i, i, i-1, False)
		buffer.appendAction(0, True)
	buffer.append(np.ones([1,1])*255/10, 10, 10-1, True)
	buffer.appendAction(0, False)
	print 'len:', len(buffer)
	for i in range(len(buffer)):
		print buffer[i]
	print buffer.sample(5)

