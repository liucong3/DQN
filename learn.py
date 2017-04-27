#coding=utf-8

import tensorflow as tf
import numpy as np
import os, time, datetime, json
from PIL import Image
from net import Net
from env import AtariEnv
from buffer import AtariBuffer

'''
Implementation of reinforcement Learning, including the following techniques:
 - target Q
 - experience replay
 - double Q learning
 - dueling network
'''
class RL():

	def __init__(self, opt, gameEnv, qNetwork, qTarget, params, replayBuffer):
		self.gameEnv = gameEnv
		self.qNetwork = qNetwork
		self.qTarget = qTarget
		self.replayBuffer = replayBuffer
		self.syncTarget()
		self.inputSize, self.outputSize = opt['inputSize', 'outputSize']
		self.epsEndT, self.epsEnd, self.learnStart, self.discount, self.batchSize, self.doubleDQN, self.randomStarts = \
				opt['epsEndT', 'epsEnd', 'learnStart', 'discount', 'batchSize', 'doubleDQN', 'randomStarts']
		self.trainFreq, self.targetFreq, self.reportFreq, self.evalFreq, self.saveFreq, self.savePath = \
				opt['trainFreq', 'targetFreq', 'reportFreq', 'evalFreq', 'saveFreq', 'savePath']
		self.saver = tf.train.Saver(params)
		self.terminal = True
		self.step =  self.episode = int(0)
		self.averageEpisodeReward = self.episodeReward = 0.0
		self.startTime = time.time()
		self.evalInfo = []
		self.lastReportTime = 0

	def train(self, maxSteps=None, maxEpisode=None):
		assert maxSteps or maxEpisode
		if len(self.evalInfo) > 0:
			self.step = self.evalInfo[-1]['step']
		print 'Start training from step %d ...' % self.step
		while True:
			self.step += 1
			# epsilon greedy
			self.epsilonGreedyStep()
			self.replayBuffer.append(self.state, self.action, self.prev_reward, self.terminal)
			self.episodeReward += self.prev_reward
			if self.terminal:
				self.averageEpisodeReward = 0.9 * self.averageEpisodeReward+ 0.1 * self.episodeReward
				self.episodeReward = 0
				self.episode += 1
			# train
			if self.step > self.learnStart:
				if (self.step + 1) % self.trainFreq == 0:
					self.trainStep()
				if (self.step + 1) % self.targetFreq == 0:
					self.syncTarget()
			# report, save, eval
			if (self.step + 1) % self.reportFreq == 0 and self.episode:
				self.report()
			if self.step > self.learnStart:
				if (self.step + 1) % self.evalFreq == 0:
					self.eval()
				if (self.step + 1) % self.saveFreq == 0:
					self.save()
			# terminate
			if maxSteps and self.step >= maxSteps: break
			if maxEpisode and self.episode >= maxEpisode: break
		self.endTime = time.time()

	def syncTarget(self):
		params = self.qNetwork.getParams()
		self.qTarget.setParams(params)

	def q(self, state, useTarget=False):
		state = state.reshape([-1, self.inputSize])
		if useTarget:
			return self.qTarget.forward(state)
		else:
			return self.qNetwork.forward(state)

	def curEpsilon(self):
		epsilon = 1.0
		if self.step >= self.epsEndT:
			epsilon = self.epsEnd
		elif self.step > self.learnStart:
			epsilon = self.epsEnd + (self.epsEndT - self.step) * (1 - self.epsEnd) / (self.epsEndT - self.learnStart)
		return epsilon

	def epsilonGreedyStep(self):
		if self.terminal:
			if self.randomStarts:
				self.state, self.prev_reward, self.terminal, _ = self.gameEnv.nextRandomGame(training=True)
			else:
				self.state, self.prev_reward, self.terminal, _ = self.gameEnv.newGame()
			self.action = 0
		else:
			# epsilon-greedy
			if np.random.rand() < self.curEpsilon():
				self.action = np.random.randint(self.outputSize)
			else:
				state = self.replayBuffer[-1]['state'] # contains a number (histLen) of screens
				q = self.q(state)
				self.action = np.argmax(q.reshape(-1))
			self.state, self.prev_reward, self.terminal, _ = self.gameEnv.step(self.action, training=True)


	def trainStep(self):
		batch = self.replayBuffer.sample(self.batchSize)
		if batch:
			target = self.computTarget(batch)
			self.qNetwork.trainStep(batch['state'], target, batch['action'])

	def computTarget(self, batch):
		next_state = batch['next_state']
		if self.doubleDQN:
			q2 = self.q(next_state, useTarget=True)
			q2a = self.q(next_state).argmax(1)
			q2Max = q2[:, q2a][:, 0]
		else:
			q2 = self.q(next_state, useTarget=True)
			q2Max = q2.max(1)
		return batch['reward'] + q2Max * self.discount * (1 - batch['terminal'])

	def save(self):
		path = self.savePath + '/params'
		self.saver.save(self.sess, path)
		print 'Agent is saved to:', path

		path = self.savePath + '/evalInfo.json'
		text = json.dumps(self.evalInfo, indent=4, sort_keys=False, ensure_ascii=False)
		f = open(path, 'w')
		f.write(text)
		f.close()

	def load(self):
		path = self.savePath + '/params'
		if os.path.exists(path + '.index'):
			self.saver.restore(self.sess, path)
			self.syncTarget()
			print 'Agent is loaded from:', path

		path = self.savePath + '/evalInfo.json'
		try:
			self.evalInfo = Option.loadJSON(path)
		except IOError:
			self.evalInfo = []

	@staticmethod
	def duration(dt):
		return str(datetime.timedelta(dt / 24 / 3600))

	def report(self):
		self.lastReportTime = self.lastReportTime or self.startTime
		curTime = time.time()
		t = RL.duration(curTime - self.startTime)
		dt = RL.duration(curTime - self.lastReportTime)
		self.lastReportTime = curTime
		print 'step:%d/e:%d t:%s [dt:%s] reward:%f' % (self.step, self.episode, t, dt, self.averageEpisodeReward)

	def eval(self):
		pass



class AtariRL(RL):

	def __init__(self, opt):
		gameEnv = AtariEnv.create(opt) # initialize the game environment
		self.optimizer = tf.train.RMSPropOptimizer(learning_rate=opt['learningRate'], decay=0.95, epsilon=0.01, centered=True)
		# initialize session
		config = tf.ConfigProto()
		# config.log_device_placement = True
		config.gpu_options.allow_growth = True
		self.sess = tf.Session(config=config)
		# initialize neural networks
		opt['convShape'] = [opt['height'], opt['width'], opt['histLen']]
		opt['outputSize'] = gameEnv.getActions()
		with tf.device(opt['device']):
			qNetwork = Net(opt, self.sess, name='qNetwork', optimizer=self.optimizer)
			self.sess.run(tf.global_variables_initializer())
			if opt['targetFreq']:
				qTarget = Net(opt, self.sess, name='qTarget')
				
			else:
				qTarget = qNetwork
		# initialize replay buffer
		replayBuffer = AtariBuffer(int(opt['bufSize']), int(opt['histLen']))
		# initializer
		opt['inputSize'] = int(np.prod(opt['convShape']))
		RL.__init__(self, opt, gameEnv, qNetwork, qTarget, qNetwork.params, replayBuffer)
		# other data
		self.maxReward = opt.get('maxReward', None)
		self.imageSize = (opt['height'], opt['width'])

	@staticmethod
	def preprocessState(state, imageSize):
		screen = Image.fromarray(state)
		screen = screen.convert('L')
		screen = screen.resize(imageSize)
		screen = np.asarray(screen)
		return screen

	def epsilonGreedyStep(self):
		RL.epsilonGreedyStep(self)
		self.state = AtariRL.preprocessState(self.state, self.imageSize)
		if self.maxReward:
			self.prev_reward = min(self.prev_reward, self.maxReward)
			self.prev_reward = max(self.prev_reward, -self.maxReward)



##################
#      TEST      #
##################

if __name__ == '__main__':
	from option import Option
	opt = Option('config.json')
	AtariEnv.create(opt)
	env = opt['env']
	opt['savePath'] = 'save_' + env
	# net
	trainer = AtariRL(opt)
	if os.path.exists(opt['savePath']):
		trainer.load()
	else:
		os.makedirs(opt['savePath'])
	trainer.train(opt['trainSteps'])

