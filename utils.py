import os
import logging

def getLogger(path):
	logger = logging.getLogger()
	logger.setLevel(logging.DEBUG)
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
	fh = logging.FileHandler(os.path.join(path, 'logs', 'log.txt'))
	fh.setLevel(logging.INFO)
	fh.setFormatter(formatter)
	ch = logging.StreamHandler()
	ch.setLevel(logging.INFO)
	ch.setFormatter(formatter)
	logger.addHandler(ch)
	logger.addHandler(fh)
	return logger

class AvgMeter(object):
	def __init__(self):
		self.clear()

	def add(self, value):
		self.value += value
		self.n += 1

	def get(self):
		if self.n == 0:
			return 0
		return self.value/self.n

	def clear(self):
		self.n = 0
		self.value = 0.

class AvgMeters(object):
	def __init__(self):
		self.clear()

	def add(self, name, value):
		if name in self.meters:
			self.meters[name].add(value)
		else:
			self.meters[name] = AvgMeter()
			self.meters[name].add(value)

	def get(self, name):
		return self.meters[name].get()

	def clear(self):
		self.meters = {}
