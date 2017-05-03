#coding=utf-8

import gym
import numpy as np

def envFormat(env):
	'''
	envFormat("abc_edfg_123") returns
	AbcEdfg123NoFrameskip-v3
	'''
	name = ''.join([g.capitalize() for g in env.split('_')])
	name = '{}NoFrameskip-v3'.format(name)
	return name

class AtariEnv:

	def __init__(self, game_name, actrep=1, randomStarts=1):
		'''
		Arguments:
			* actrep: number of times that each action repeats.
			* game_name: name of the game to start. Available games are given by AtariEnv.games
			* actrep: action repeating in each AtariEnv.step(action)
			* randomStarts: number of random AtariEnv.step(action=)
		'''
		self.env = game_name;
		self.actrep = actrep
		self.randomStarts = randomStarts
		self._reset()

	def _reset(self):
		self.game = gym.make(envFormat(self.env))
		assert type(self.game.env) == gym.envs.atari.AtariEnv, "只支持atari游戏"
		self.actionSpace = self.game.action_space
		self.game.reset()
		self.prev_lives = -1

	'''
	Name of the available game, which can be used to call atari_env=AtariEnv(game_name)
	'''
	games = ['Atlantis', 
			'Double_Dunk',
			'Krull',
			'Video_Pinball',
			'Enduro',
			'Jamesbond',
			'Frostbite',
			'Kangaroo',
			'Up_n_Down',
			'Zaxxon',
			'Road_Runner',
			'Assault',
			'Wizard_of_Wor', # This game have problem: screen constantly flicker
			'Tutankham',
			'Bank_Heist',
			'Kung_Fu_Master',
			'Beam_Rider',
			'QBert',
			'Boxing',
			'Battle_Zone',
			'Tennis',
			'Fishing_Derby',
			'Bowling',
			'Freeway',
			'Gravitar',
			'Pong',
			'Name_This_Game',
			'Montezuma_Revenge',
			'HERO', # An error occurs: could not initialize the game environment
			'Crazy_Climber',
			'Asteroids', # display problem: black screen
			'Private_Eye',
			'Riverraid',
			'Ms_Pacman',
			'Chopper_Command',
			'Seaquest',
			'Asterix',
			'Venture',
			'Amidar',
			'Ice_Hockey',
			'Alien',
			'Time_Pilot',
			'Robotank',
			'Centipede',
			'Demon_Attack',
			'Breakout',
			'Gopher',
			'Space_Invaders',
			'Star_Gunner']

	def render(self):
		self.game.render()

	def sample(self):
		return self.actionSpace.sample()

	def randomStep(self):
		return self.game.step(self.sample())

	def step(self, action, training=False):
		reward = 0
		for _ in range(self.actrep):
			observation, reward1, terminal, info = self.game.step(action)
			reward += reward1
			lives = info.get('ale.lives')
			terminal = terminal or (training and lives and self.prev_lives > 0 and lives < self.prev_lives)
			self.prev_lives = lives
			if terminal: break
		return observation, reward, terminal, lives

	def newGame(self):
		self.game.reset()
		observation, reward, terminal, info = self.game.step(0)
		lives = info.has_key('ale.lives') and info['ale.lives'] or None
		return observation, reward, terminal, lives

	def nextRandomGame(self, k=None, training=False):
		k = k or np.random.randint(0, self.randomStarts)
		if training:
			_, _, terminal, _ = self.game.step(0)
			if terminal: self.game.reset()
		else:
			self.newGame()
		for i in range(k):
			_, _, terminal, _ = self.game.step(0)
			if terminal:
				self.game.reset()
				break
		return self.game.step(0)

	def getActions(self):
		return self.game.action_space.n

	def get_action_meanings(self):
		# https://github.com/openai/gym/blob/master/gym/envs/atari/atari_env.py
		return self.game.env.get_action_meanings()

	@staticmethod
	def create(opt):
		if not opt.get('env', None):
			print 'Please try one of the following', str(len(AtariEnv.games)), 'commands:'
			for game in AtariEnv.games:
				print 'python', __file__ ,'--env', game
			exit(0)
		try:
			id = int(opt['env'])
			opt['env'] = AtariEnv.games[id]
		except ValueError:
			pass
		#return AtariEnv(opt['env'], opt['actrep'], opt['randomStarts'])
		return AtariEnv(*opt['env', 'actrep', 'randomStarts'])


