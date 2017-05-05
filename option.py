#coding=utf-8
import sys
from optparse import OptionParser
import json
import string

class Option:

	def __init__(self, configurationFile):
		'''
		This option parser must be initialized with a configuration file.
		Keys in the configuration file becomes long option keys
		Short option keys are automatically assigned.
		Use the -h option to view the assigned short option keys.
		'''
		parser = OptionParser()
		self.options = {}
		self.load(configurationFile)
		short = self._shortKeys(self.options.keys())
		for key, value in self.options.iteritems():
			#print '-' + short[key], '--' + key
			parser.add_option('-' + short[key], '--' + key, 
							action="store", 
							dest=key, 
							default=value, 
							help = 'e.g. ' + str(value))
		(options, args) = parser.parse_args(sys.argv[1:])
		for key, value in options.__dict__.iteritems():
			self.options[key] = value
			if value is None: continue
			if not isinstance(value, str): continue
			try:
				self.options[key] = int(value)
			except ValueError:
				try:
					self.options[key] = float(value)
				except ValueError:
					if value == 'True' or value == 'true': self.options[key] = True
					if value == 'False' or value == 'false': self.options[key] = False

	def _shortKeys(self, keys):
		'''
		This function assigns short option keys.
		arguments
			* keys: list
		return
			* short: map of {long-option-key : short-option-key}
		'''
		short = {'help': 'h'}
		free = set(string.ascii_letters + string.digits)
		free.remove('h')
		for key in keys:
			if key[0] in free:
				short[key] = key[0];
				free.remove(key[0])
		for key in keys:
			if not short.has_key(key):
				if key[0].upper() in free:
					short[key] = key[0].upper()
					free.remove(key[0].upper())
				if key[0].lower() in free:
					short[key] = key[0].lower()
					free.remove(key[0].lower())
		for key in keys:
			if not short.has_key(key):
				k0 = list(free)[0]
				short[key] = k0
				free.remove(k0)
		return short

	def save(self, file):
		Option.saveJSON(file, self.options)

	def load(self, file):
		self.options = Option.loadJSON(file)

	@staticmethod
	def saveJSON(file, obj):
		text = json.dumps(obj, indent=4, sort_keys=False, ensure_ascii=False)
		f = open(file, 'w')
		f.write(text)
		f.close()

	@staticmethod
	def loadJSON(file):
		f = open(file, 'r')
		obj = json.load(f)
		obj = Option._byteify(obj)
		f.close()
		return obj

	@staticmethod
	def _byteify(input):
		if isinstance(input, dict):
			return {Option._byteify(key): Option._byteify(value)
					for key, value in input.iteritems()}
		elif isinstance(input, list):
			return [Option._byteify(element) for element in input]
		elif isinstance(input, unicode):
			return input.encode('utf-8')
		else:
			return input

	def get(self, key, defaultValue):
		'''
		Return the value if there is a key in self.options
		Otherwise return and add the default value into self.options
		'''
		if self.options.has_key(key):
			value = self.options[key]
			if value != None:
				return value
		if defaultValue:
			self.options[key] = defaultValue
			return defaultValue
		return None

	def __getitem__(self, keys):
		values = []
		if not isinstance(keys, tuple):
			keys = (keys,)
		for key in keys:
			value = self.get(key, None)
			assert not value is None, key + ' is None'
			values.append(value)
		if len(values) == 1: return values[0]
		return values

	def __setitem__(self, keys, values):
		'''
		Override values in the self.options
		'''
		if not isinstance(keys, tuple):
			assert not isinstance(values, tuple)
			keys = (keys,)
			values = (values, )
		for i in range(len(keys)):
			self.options[keys[i]] = values[i]

	def __str__(self):
		return json.dumps(self.options, indent=4,
				sort_keys=False, ensure_ascii=False)

	def copy(self):
		class OptionCopy(Option):
			def __init__(self, opt):
				self.options = opt.options
		return OptionCopy(self)


##################
#      TEST      #
##################

if __name__ == '__main__':
	opt = Option('config.json')
	# print opt['device']
	# print opt['device', 'learningRate']
	# opt['device'] = 'ABC'
	# opt['device', 'learningRate'] = 'ABC', 123
	print opt
	#opt.save('1.txt')

