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
		self.step = self.episode = 0
		self.totalReward = self.episodeReward = self.prevTotalReward= 0.0
		self.startTime = time.time()
		self.prevReportTime = self.prevStep = self.prevEpisode = 0
		# eval
		self.evalInfo = []
		self.bestScore = -1
		self.evalBatchSize = opt.get('evalBatchSize', None)
		self.debug = opt['debug']


	def train(self, maxSteps=None, maxEpisode=None):
		assert maxSteps or maxEpisode
		if len(self.evalInfo) > 0:
			self.step = self.evalInfo[-1]['step']
		print 'Start training from step %d ...' % self.step
		while True:
			self.step += 1
			self.gameEnvStep()
			if maxSteps and self.step >= maxSteps: break
			if maxEpisode and self.episode >= maxEpisode: break
			# epsilon greedy step
			self.replayBuffer.append(self.state, self.prev_reward, self.terminal)
			self.epsilonGreedyStep()
			self.replayBuffer.appendAction(self.action, self.is_episode_step)
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
		self.endTime = time.time()

	def syncTarget(self):
		if self.qTarget:
			print 'syncTarget -- ' + time.ctime()
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
		# epsilon-greedy
		if np.random.rand() < self.curEpsilon():
			self.action = np.random.randint(self.outputSize)
			self.is_episode_step = True
		else:
			state = self.replayBuffer[-1]['state'] # contains a number (histLen) of screens
			q = self.q(state)
			self.action = np.argmax(q.reshape(-1))

	def gameEnvStep(self, training=True):
		if self.terminal:
			if self.randomStarts:
				self.state, self.prev_reward, self.terminal, _ = self.gameEnv.nextRandomGame(training=training)
			else:
				self.state, self.prev_reward, self.terminal, _ = self.gameEnv.newGame()
		else:
			self.state, self.prev_reward, self.terminal, _ = self.gameEnv.step(self.action, training=training)
		# accumulate rewards
		self.episodeReward += self.prev_reward
		if self.terminal: 
			self.totalReward += self.episodeReward
			self.episodeReward = 0
			self.episode += 1

	def trainStep(self):
		batch = self.replayBuffer.sample(self.batchSize)
		if batch:
			#print 'trainStep -- ' + time.ctime()
			qMax = self.computeTarget(batch['next_state'])
			target = batch['reward'] + qMax * batch['discount'] * (1 - batch['terminal'])
			self.qNetwork.trainStep(batch['state'], target, batch['action'])

	def computeTarget(self, state, getAll=False):
		if self.doubleDQN:
			targetQs = self.q(state, useTarget=True)
			qs = self.q(state)
		else:
			targetQs = self.q(state, useTarget=True)
			qs = targetQs
		action = qs.argmax(1)
		qMax = targetQs[:, action][:, 0]
		if getAll:
			return qMax, targetQs, qs, action
		else:
			return qMax

	def save(self, saveModel=True):
		if saveModel:
			path = self.savePath + '/model'
			self.saver.save(self.sess, path)
			print 'Model is saved to:', path
		path = self.savePath + '/evalInfo.json'
		Option.saveJSON(path, self.evalInfo)

	def load(self):
		path = self.savePath + '/model'
		if os.path.exists(path + '.index'):
			self.saver.restore(self.sess, path)
			print 'Agent is loaded from:', path
			self.syncTarget()

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
		print time.ctime()
		curTime = time.time()
		episode = self.episode - self.prevEpisode
		if episode > 0: # to prevent dividing by zero episode
			step = self.step - self.prevStep
			time1 = RL.duration(curTime - self.prevReportTime)
			time2 = RL.duration(curTime - self.startTime)
			totalReward = self.totalReward - self.prevTotalReward
			bufferSize = len(self.replayBuffer)
			episodeSize = len(self.replayBuffer.episodeInfo)
			print 'S:%d E:%d T|%s, s:%d e:%d t|%s, s/e:%.2f r/e:%.2f B:%d/%d' % (self.step, self.episode, time2, step, episode, time1, (step / episode), (totalReward / episode), bufferSize, episodeSize)
		self.prevStep = self.step
		self.prevEpisode = self.episode
		self.prevReportTime = curTime
		self.prevTotalReward = self.totalReward
		self.printDebugInfo()

	@staticmethod
	def printInfo(info):
		keys = info.keys()
		keys.sort()
		for key in keys:
			value = info[key]
			if key.startswith('time'): print '\t' + key + '|' +  RL.duration(value)
			elif type(value).__name__.find('float') != -1: print '\t' + key + ': %.6f' % value
			else: print '\t' + key + ': ' +  str(value)

	def eval(self, evalInfo={}):
		curTime = time.time()
		evalInfo['step'] = self.step
		evalInfo['time'] = curTime - self.startTime	
		evalInfo['time_eval'] = curTime - self.prevReportTime
		print 'Evaluation:'
		RL.printInfo(evalInfo)
		self.evalInfo.append(evalInfo)
		self.prevReportTime = curTime
		return -1

	@staticmethod
	def printDebugInfo4(debug, params, deltas, output, grads, batchSize):
		info = {}
		info['TD'] = np.abs(deltas).mean()
		info['deltas mean'] = deltas.mean()
		info['deltas std'] = deltas.std()
		info['Q mean'] = output.mean()
		info['Q std'] = output.std()
		if debug > 1:
			norms = []
			maxs = []
			for param in params:
				norms.append(np.abs(param).mean())
				maxs.append(np.abs(param).max())
			info['param norm'] = RL.debugListRepr(norms)
			info['param max'] = RL.debugListRepr(maxs)
			norms = []
			maxs = []
			for grad in grads:
				norms.append(np.abs(grad).mean() / batchSize)
				maxs.append(np.abs(grad).max() / batchSize)
			info['grads norm'] = RL.debugListRepr(norms)
			info['grads max'] = RL.debugListRepr(maxs)
		print 'Debug info:'
		RL.printInfo(info)

	def printDebugInfo(self):
		if not self.debug or not self.evalBatchSize: return
		batch = self.replayBuffer.sample(self.evalBatchSize)
		if not batch: return
		qMax = self.computeTarget(batch['next_state'])
		state = batch['state']
		action = batch['action']
		targets = batch['reward'] + qMax * batch['discount'] * (1 - batch['terminal'])
		deltas, output, grads = self.qNetwork.getDebugInfo(state, targets, action)
		params = self.qNetwork.getParams() if self.debug > 1 else None
		RL.printDebugInfo4(self.debug, params, deltas, output, grads, self.evalBatchSize)

	@staticmethod
	def debugListRepr(li):
		repr = '['
		for i in range(len(li)):
			if i == 0:
				repr += '%.6f' % li[i]
			else:
				repr += ', %.6f' % li[i]
		repr += ']'
		return repr


class AtariControl(RL):

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
		AtariControl.initOptions(opt, gameEnv)
		self.sess = sess
		# initialize replay buffer
		opt = opt.copy()
		opt['bufSize'] = 1000
		replayBuffer = AtariBuffer(opt) # small buffer always clean up obsolete spaces
		# initializer
		RL.__init__(self, opt, gameEnv, qNetwork, None, None, replayBuffer)
		# other data
		self.imageSize = (opt['height'], opt['width'])
		self.epsTest = opt['epsTest']
		self.randomStarts = None

	def curEpsilon(self):
		return self.epsTest

	def getAction(self, state):
		state = AtariControl.preprocessState(state, self.imageSize)
		self.replayBuffer.append(state, None, False)
		self.epsilonGreedyStep()
		return self.action

	def eval(self, maxSteps=None, maxEpisode=None):
		assert maxSteps or maxEpisode
		self.step = 0
		self.episode = 0
		self.episodeReward = 0.0
		self.totalReward = 0.0
		self.terminal = True
		while True:
			self.step += 1
			self.gameEnvStep(training=False)
			if maxSteps and self.step >= maxSteps: break
			if maxEpisode and self.episode >= maxEpisode: break
			# epsilon greedy
			self.state = AtariControl.preprocessState(self.state, self.imageSize)
			self.replayBuffer.append(self.state, self.prev_reward, self.terminal)
			self.epsilonGreedyStep()
		self.endTime = time.time()
		avgTotalReward = float(self.totalReward) / self.episode if self.episode else 0
		return {'total_reward':avgTotalReward, 'step_eval':self.step, 'episode_eval':self.episode}


class AtariRL(RL):

	def __init__(self, opt, NetType=Net, BufferType=AtariBuffer):
		gameEnv = AtariEnv.create(opt) # initialize the game environment
		AtariControl.initOptions(opt, gameEnv)
		self.optimizer = tf.train.RMSPropOptimizer(learning_rate=opt['learningRate'], decay=0.95, epsilon=0.01, centered=True)
		# initialize session
		config = tf.ConfigProto()
		# config.log_device_placement = True
		config.gpu_options.allow_growth = True
		self.sess = tf.Session(config=config)
		# initialize neural networks
		with tf.device(opt['device']):
			qNetwork = NetType(opt, self.sess, name='QNetwork', optimizer=self.optimizer)
			self.sess.run(tf.global_variables_initializer())
			qTarget = NetType(opt, self.sess, name='QTarget') if opt['targetFreq'] else None
		# initialize replay buffer
		replayBuffer = BufferType(opt)
		# initializer
		RL.__init__(self, opt, gameEnv, qNetwork, qTarget, qNetwork.params, replayBuffer)
		# other data
		self.maxReward = opt.get('maxReward', None)
		self.imageSize = (opt['height'], opt['width'])
		# evaluator
		self.evaluator = AtariControl(opt, self.sess, self.qNetwork)
		self.evalMaxSteps, self.evalMaxEpisode = opt['evalMaxSteps', 'evalMaxEpisode']

	def gameEnvStep(self):
		RL.gameEnvStep(self)
		self.state = AtariControl.preprocessState(self.state, self.imageSize)
		if self.maxReward:
			self.prev_reward = min(self.prev_reward, self.maxReward)
			self.prev_reward = max(self.prev_reward, -self.maxReward)

	def eval(self, evalInfo={}):
		evalInfo = self.evaluator.eval(self.evalMaxSteps, self.evalMaxEpisode)
		RL.eval(self, evalInfo)
		return evalInfo['total_reward']


##################
#      TEST      #
##################

if __name__ == '__main__':
	from option import Option
	opt = Option('config.json')
	AtariEnv.create(opt)
	env = opt['env']
	if not opt.get('savePath', None):
		opt['savePath'] = 'save_' + env
	# net
	trainer = AtariRL(opt)
	if os.path.exists(opt['savePath']):
		trainer.load()
	else:
		os.makedirs(opt['savePath'])
	trainer.train(opt['trainSteps'])

