#coding=utf-8
import os
import tensorflow as tf
import numpy as np
from net import Net
from learn import RL, AtariRL
from buffer import ReplayBuffer, AtariBuffer

'''
LEARNING TO PLAY IN A FEW HOURS: FAST DEEP REINFORCEMENT LEARNING WITH DYNAMIC LEARNING STEPS
'''

class FastBuffer(AtariBuffer):

	def append(self, state, action, prev_reward, terminal, is_episode_step):
		AtariBuffer.append(self, state, action, prev_reward, terminal, is_episode_step)
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
				o['next_state'] = prev_epsilon_state - n
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
		while n > 0:
			k = np.random.randint(size - 1)
			o = self[k]
			if not o['terminal']:
				n -= 1
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
	opt['savePath'] = 'save_' + env
	# net
	trainer = AtariRL(opt, BufferType=FastBuffer)
	if os.path.exists(opt['savePath']):
		trainer.load()
	else:
		os.makedirs(opt['savePath'])
	trainer.train(opt['trainSteps'])


