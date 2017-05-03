#coding=utf-8

import sys
import numpy as np
from env import AtariEnv

argv = sys.argv 
sys.argv = argv[:1] # to disable Kivy's option parser
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

	def __init__(self, gameEnv, gameControl, displayEnlarged=3):
		self.gameEnv = gameEnv
		self.gameControl = gameControl
		self.displayEnlarged = displayEnlarged
		App.__init__(self)

	def build(self):
		self.updateEvent = Clock.schedule_interval(self.update, 0.1)
		self.title = opt['env'] + ' ' + str(self.gameEnv.get_action_meanings())

		self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
		self._keyboard.bind(on_key_down=self._on_keyboard_down)
		self._keyboard.bind(on_key_up=self._on_keyboard_up)
		
		self.observation, _, _, self.lives = self.gameEnv.step(0)
		shape = self.observation.shape
		self.image = Image(texture=self._get_texture(self.observation))
		self.y = 0
		Window.size = (160 * self.displayEnlarged, 210 * self.displayEnlarged)
		return self.image

	def to_window(self, x, y):
		return (x, y)

	def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
		self.gameControl.on_keyboard_down(keycode[1])

	def _on_keyboard_up(self, keyboard, keycode):
		self.gameControl.on_keyboard_up(keycode[1])

	def _keyboard_closed(self):
		self._keyboard.unbind(on_key_down=self._on_keyboard_down)
		self._keyboard = None

	def update(self, dt):
		prev_lives = self.lives
		action = self.gameControl.getAction(self.observation)
		print 'action:', gameEnv.get_action_meanings()[action]
		self.observation, reward, terminal, self.lives = self.gameEnv.step(action)
		#print 'self.observation.mean()', self.observation.mean()
		self.image.texture = self._get_texture(self.observation)
		info = ''
		if reward != 0:
			info += ' reward:' + str(reward)
		if self.lives != prev_lives:
			info += ' lives:' + str(prev_lives) + '->' + str(self.lives)
		if terminal:
			info += ' terminal'
		if len(info) > 0:
			print '*' + info
		if terminal:
			self.updateEvent.cancel()
			Clock.schedule_once(lambda dt: exit(0), 2)
		#print reward, terminal and 'terminal' or ''

	def _get_texture(self, observation):
		observation = np.repeat(observation, self.displayEnlarged * 3, axis=0)
		observation = np.repeat(observation, self.displayEnlarged * 3, axis=1)
		shape = observation.shape
		texture = Texture.create(size=(shape[1], shape[0]))
		texture.blit_buffer(observation.tostring(), colorfmt='rgb', bufferfmt='ubyte')
		texture.flip_vertical()
		return texture



class AtariKeyControl:

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

	# required by all Control classes
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

	# required by all Control classes
	def getAction(self, observation):
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
		actionText = AtariKeyControl.all_action_meanings[action]
		action2 = 0;
		if self.meaning_actions.has_key(actionText):
			action2 = self.meaning_actions[actionText]
		#print actionText, self.action_meanings[action2]
		return action2



if __name__ == '__main__':
	from option import Option
	sys.argv = argv
	opt = Option('config.json')
	gameEnv = AtariEnv.create(opt)
	if not opt.get('savePath', None):
		gameControl = AtariKeyControl(gameEnv.get_action_meanings())
	else:
		from learn import AtariControl
		AtariControl.initOptions(opt, gameEnv)
		import tensorflow as tf
		sess = tf.Session()
		from net import Net
		#qNetwork = Net(opt, sess, name='QNetwork')
		from netComm import buildNet
		qNetwork = buildNet(opt, gameEnv, sess)
		path = opt['savePath'] # save/agent-best
		saver = tf.train.Saver(qNetwork.params)
		saver.restore(sess, path)
		gameControl = AtariControl(opt, sess, qNetwork)
	AtariUI(gameEnv, gameControl).run()
