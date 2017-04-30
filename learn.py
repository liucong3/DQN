#coding=utf-8

import tensorflow as tf
import numpy as np
import os, time, datetime, json
from PIL import Image
from net import Net
from env import AtariEnv
from buffer import AtariBuffer
from option import Option

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
		self.inputSize, self.outputSize, self.batchSize = opt['inputSize', 'outputSize', 'batchSize']
		self.epsEndT, self.epsEnd, self.learnStart, self.discount, self.doubleDQN, self.randomStarts = opt['epsEndT', 'epsEnd', 'learnStart', 'discount', 'doubleDQN', 'randomStarts']
		self.trainFreq, self.targetFreq, self.reportFreq, self.evalFreq, self.savePath = opt['trainFreq', 'targetFreq', 'reportFreq', 'evalFreq', 'savePath']
		if params: self.saver = tf.train.Saver(params)
		self.terminal = True
		# report
		self.step = self.episode = int(0)
		self.totalReward = self.episodeReward = 0.0
		self.startTime = time.time()
		self.prevReportTime = self.prevStep = self.prevTotalReward = 0
		# eval
		self.evalInfo = []
		self.bestScore = -1


	def train(self, maxSteps=None, maxEpisode=None):
		assert maxSteps or maxEpisode
		if len(self.evalInfo) > 0:
			self.step = self.evalInfo[-1]['step']
		print 'Start training from step %d ...' % self.step
		while True:
			self.step += 1
			# epsilon greedy
			self.epsilonGreedyStep()
			self.replayBuffer.append(self.state, self.action, self.prev_reward, self.terminal, self.is_episode_step)
			self.episodeReward += self.prev_reward
			if self.terminal: 
				self.totalReward += self.episodeReward
				self.episodeReward = 0
				self.episode += 1
			# train
			if self.step > self.learnStart:
				if self.step % self.trainFreq == 0:
					self.trainStep()
				if self.step % self.targetFreq == 0:
					self.syncTarget()
			# report, save, eval
			if self.step == 1 or self.step % self.reportFreq == 0:
				self.report()
			if self.step > self.learnStart and self.step % self.evalFreq == 0:
				score = self.eval()
				self.save(self.bestScore < score)
				if self.bestScore < score: self.bestScore = score				
			# terminate
			if maxSteps and self.step >= maxSteps: break
			if maxEpisode and self.episode >= maxEpisode: break
		self.endTime = time.time()

	def syncTarget(self):
		if self.qTarget:
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
		self.is_episode_step = False
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
				self.is_episode_step = True
			else:
				state = self.replayBuffer[-1]['state'] # contains a number (histLen) of screens
				q = self.q(state)
				self.action = np.argmax(q.reshape(-1))
			self.state, self.prev_reward, self.terminal, _ = self.gameEnv.step(self.action, training=True)

	def trainStep(self):
		batch = self.replayBuffer.sample(self.batchSize)
		if batch:
			q2Max = self.computTarget(batch)
			target = batch['reward'] + q2Max * batch['discount'] * (1 - batch['terminal'])
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
		return q2Max

	def save(self, saveModel=True):
		if saveParams:
			path = self.savePath + '/model'
			self.saver.save(self.sess, path)
			print 'Model is saved to:', path
		path = self.savePath + '/evalInfo.json'
		Option.saveJSON(path, self.evalInfo)

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
		# http://www.cnblogs.com/lhj588/archive/2012/04/23/2466653.html
		dt = float(int(dt))
		return str(datetime.timedelta(dt / 24 / 3600))

	def report(self):
		curTime = time.time()
		if self.episode and self.prevStep: # to prevent dividing by zero episode
			step = self.step - self.prevStep
			episode = self.episode - self.prevEpisod
			time1 = RL.duration(curTime - self.prevReportTime)
			time2 = RL.duration(curTime - self.startTime)
			totalReward = self.totalReward - self.prevTotalReward
			print 'S:%d E:%d T|%s, s:%d e:%d t|%s, s/e:%.2f r/e:%.2f' % (self.step, self.episode, time2, step, episode, time1, (step / episode), (totalReward / episode))
		self.prevStep = self.step
		self.prevEpisod = self.episode
		self.prevReportTime = curTime
		self.prevTotalReward = self.totalReward

	def eval(self, evalInfo={}):
		curTime = time.time()
		evalInfo['step'] = self.step
		evalInfo['time'] = curTime - self.startTime	
		evalInfo['time_eval'] = curTime - self.prevReportTime
		self.evalInfo.append(evalInfo)
		self.prevReportTime = curTime
		textInfo = 'Eval'
		for key, value in evalInfo.iteritems():
			if key.startswith('time'): textInfo += ' ' + key + '|' +  RL.duration(value)
			else: textInfo += ' ' + key + ':' +  str(value)
		print textInfo
		return -1

class AtariPlayer(RL):

	@staticmethod
	def preprocessState(state, imageSize):
		screen = Image.fromarray(state)
		screen = screen.convert('L')
		screen = screen.resize(imageSize)
		screen = np.asarray(screen)
		return screen

	@staticmethod
	def initOptions(opt, gameEnv):
		opt['convShape'] = [opt['height'], opt['width'], opt['histLen']]
		opt['outputSize'] = gameEnv.getActions()
		opt['inputSize'] = int(np.prod(opt['convShape']))

	def __init__(self, opt, sess, qNetwork):
		gameEnv = AtariEnv.create(opt) # initialize the game environment
		AtariPlayer.initOptions(opt, gameEnv)
		self.sess = sess
		# initialize replay buffer
		opt = opt.copy()
		opt['bufSize'] = 1
		replayBuffer = AtariBuffer(opt) # small buffer always clean up obsolete spaces
		# initializer
		RL.__init__(self, opt, gameEnv, qNetwork, None, None, replayBuffer)
		# other data
		self.imageSize = (opt['height'], opt['width'])
		self.render, self.epsTest = opt['render', 'epsTest']
		self.randomStarts = None

	def curEpsilon(self):
		return self.epsTest

	def play(self, maxSteps=None, maxEpisode=None):
		assert maxSteps or maxEpisode
		self.step = 0
		self.episode = 0
		self.episodeReward = 0.0
		self.totalReward = 0.0
		while True:
			self.step += 1
			# epsilon greedy
			self.epsilonGreedyStep()
			self.state = AtariPlayer.preprocessState(self.state, self.imageSize)
			self.replayBuffer.append(self.state, self.action, self.prev_reward, self.terminal)
			# accumulate rewards
			self.episodeReward += self.prev_reward
			if self.terminal:
				self.totalReward += self.episodeReward
				self.episodeReward = 0
				self.episode += 1
			# render
			if self.render:
				self.gameEnv.render()
			# terminate
			if maxSteps and self.step >= maxSteps: break
			if maxEpisode and self.episode >= maxEpisode: break
		self.endTime = time.time()
		avgTotalReward = self.totalReward / self.episode if self.episode else 0
		return {'totalReward':avgTotalReward, 'step_eval':self.step, 'episode_eval':self.episode}

class AtariRL(RL):

	def __init__(self, opt, NetType=Net, BufferType=AtariBuffer):
		gameEnv = AtariEnv.create(opt) # initialize the game environment
		AtariPlayer.initOptions(opt, gameEnv)
		self.optimizer = tf.train.RMSPropOptimizer(learning_rate=opt['learningRate'], decay=0.95, epsilon=0.01, centered=True)
		# initialize session
		config = tf.ConfigProto()
		# config.log_device_placement = True
		config.gpu_options.allow_growth = True
		self.sess = tf.Session(config=config)
		# initialize neural networks
		with tf.device(opt['device']):
			qNetwork = NetType(opt, self.sess, name='qNetwork', optimizer=self.optimizer)
			self.sess.run(tf.global_variables_initializer())
			qTarget = NetType(opt, self.sess, name='qTarget') if opt['targetFreq'] else None
		# initialize replay buffer
		replayBuffer = BufferType(opt)
		# initializer
		RL.__init__(self, opt, gameEnv, qNetwork, qTarget, qNetwork.params, replayBuffer)
		# other data
		self.maxReward = opt.get('maxReward', None)
		self.imageSize = (opt['height'], opt['width'])
		# evaluator
		self.evaluator = AtariPlayer(opt, self.sess, self.qNetwork)
		self.evalMaxSteps, self.evalMaxEpisode = opt['evalMaxSteps', 'evalMaxEpisode']


	def epsilonGreedyStep(self):
		RL.epsilonGreedyStep(self)
		self.state = AtariPlayer.preprocessState(self.state, self.imageSize)
		if self.maxReward:
			self.prev_reward = min(self.prev_reward, self.maxReward)
			self.prev_reward = max(self.prev_reward, -self.maxReward)

	def eval(self, evalInfo={}):
		evalInfo = self.evaluator.play(self.evalMaxSteps, self.evalMaxEpisode)
		RL.eval(self, evalInfo)
		return evalInfo['totalReward']


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

