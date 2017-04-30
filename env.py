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



################################
#           GAME UI            #
################################

if __name__ == '__main__':
	from option import Option
	opt = Option('config.json')
	gameEnv = AtariEnv.create(opt)

	actions = gameEnv.get_action_meanings()

	import sys
	displayEnlarged = 4
	sys.argv = sys.argv[:1] + ['--size=%dx%d' % (160 * displayEnlarged, 210 * displayEnlarged)]# to disable Kivy's option parser
	import kivy
	#kivy.require('1.9.0')

	from kivy.app import App
	from kivy.uix.widget import Widget
	from kivy.core.window import Window
	from kivy.clock import Clock

	#from kivy.core.image import Image as CoreImage
	from kivy.graphics.texture import Texture
	from kivy.uix.image import Image

	class AtariUI(App):

		def build(self):
			self.updateEvent = Clock.schedule_interval(self.update, 0.1)
			self.title = opt['env'] + ' ' + str(gameEnv.get_action_meanings())
			self.keyHelper = AtariKeyHelper(gameEnv.get_action_meanings())

			self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
			self._keyboard.bind(on_key_down=self._on_keyboard_down)
			self._keyboard.bind(on_key_up=self._on_keyboard_up)
			
			observation, reward, terminal, self.lives = gameEnv.step(0)
			shape = observation.shape
			self.image = Image(texture=self._get_texture(observation))
			return self.image

		def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
			self.keyHelper.on_keyboard_down(keycode[1])

		def _on_keyboard_up(self, keyboard, keycode):
			self.keyHelper.on_keyboard_up(keycode[1])

		def _keyboard_closed(self):
			self._keyboard.unbind(on_key_down=self._on_keyboard_down)
			self._keyboard = None

		def update(self, dt):
			action = self.keyHelper.getAction()
			observation, reward, terminal, lives = gameEnv.step(action)
			self.image.texture = self._get_texture(observation)
			info = ''
			if reward != 0:
				info += ' reward:' + str(reward)
			if self.lives != lives:
				info += ' lives:' + str(self.lives) + '->' + str(lives)
				self.lives = lives
			if terminal:
				info += ' terminal'
			if len(info) > 0:
				print '*' + info
			if terminal:
				self.updateEvent.cancel()
				Clock.schedule_once(lambda dt: exit(0), 2)
			#print reward, terminal and 'terminal' or ''

		def _get_texture(self, observation):
			observation = np.repeat(observation, displayEnlarged, axis=0)
			observation = np.repeat(observation, displayEnlarged, axis=1)
			shape = observation.shape
			texture = Texture.create(size=(shape[1], shape[0]))
			texture.blit_buffer(observation.tostring(), colorfmt='rgb', bufferfmt='ubyte')
			texture.flip_vertical()
			return texture

	class AtariKeyHelper:

		all_action_meanings = {
				0 : "NOOP",
				1 : "FIRE",
				2 : "UP",
				3 : "RIGHT",
				4 : "LEFT",
				5 : "DOWN",
				6 : "UPRIGHT",
				7 : "UPLEFT",
				8 : "DOWNRIGHT",
				9 : "DOWNLEFT",
				10 : "UPFIRE",
				11 : "RIGHTFIRE",
				12 : "LEFTFIRE",
				13 : "DOWNFIRE",
				14 : "UPRIGHTFIRE",
				15 : "UPLEFTFIRE",
				16 : "DOWNRIGHTFIRE",
				17 : "DOWNLEFTFIRE"}

		def __init__(self, action_meanings):
			print 'action_meanings:', action_meanings
			self.keys = set()
			self.action_meanings = action_meanings
			self.meaning_actions = {}
			for i, m in enumerate(action_meanings):
				self.meaning_actions[m] = i

		def on_keyboard_down(self, key):
			if key == 'spacebar':
				self.keys.add(key)
			elif key == 'up':
				self.keys.add(key)
				if 'down' in self.keys:
					self.keys.remove('down')
			elif key == 'down':
				self.keys.add(key)				
				if 'up' in self.keys:
					self.keys.remove('up')
			elif key == 'left':
				self.keys.add(key)
				if 'right' in self.keys:
					self.keys.remove('right')
			elif key == 'right':
				self.keys.add(key)
				if 'left' in self.keys:
					self.keys.remove('left')

		def on_keyboard_up(self, key):
			if key in self.keys:
				self.keys.remove(key)

		def getAction(self):
			action = 0
			if 'spacebar' in self.keys:
				if 'down' in self.keys:
					if 'left' in self.keys:
						action = 17
					elif 'right' in self.keys:
						action = 16
					else:
						action = 13
				elif 'up' in self.keys:
					if 'left' in self.keys:
						action = 15
					elif 'right' in self.keys:
						action = 14
					else:
						action = 10
				else:
					if 'left' in self.keys:
						action = 12
					elif 'right' in self.keys:
						action = 11
					else:
						action = 1
			else:
				if 'down' in self.keys:
					if 'left' in self.keys:
						action = 9
					elif 'right' in self.keys:
						action = 8
					else:
						action = 5
				elif 'up' in self.keys:
					if 'left' in self.keys:
						action = 7
					elif 'right' in self.keys:
						action = 6
					else:
						action = 2
				else:
					if 'left' in self.keys:
						action = 4
					elif 'right' in self.keys:
						action = 3
					else:
						action = 0
			actionText = AtariKeyHelper.all_action_meanings[action]
			action2 = 0;
			if self.meaning_actions.has_key(actionText):
				action2 = self.meaning_actions[actionText]
			#print actionText, self.action_meanings[action2]
			return action2

	AtariUI().run()
