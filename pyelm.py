import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import theano.tensor.nlinalg
import numpy as np


class ELM:
	""" 
		Visible - input data as pair (x,y)
		num_hidd - number of hidden neurons
	"""
	def __init__(self, visible, num_hidd):
		self.inpdata = list(map(lambda x: x[0], visible))
		self.outdata = list(map(lambda x: x[1], visible))
		stream = RandomStreams()
		num_vis = len(self.inpdata)
		num_out = len(self.outdata)
		self.vis = T.dmatrix('vis')
		self.outd = T.dmatrix('out')
		#init as weights vector connected from input layer to hidden
		self.weights_inphid = theano.shared(
			np.array(
				np.random.random((num_vis, num_hidd))
				), name='Wih')

		#init as weights vector connected from hidden layer to output
		self.weights_hidout = theano.shared(
			np.array(
				np.random.random((num_hidd, num_out))
				), name='Who')
		self.bias = theano.shared(np.ones(num_hidd, dtype=theano.config.floatX))

	def train(self, iters):
		""" TODO: write function with scan and updates """
		hidden_pre = T.dot(self.vis, self.weights_inphid) + self.bias
		hidden = T.dot(self.weights_hidout, T.nnet.sigmoid(T.dot(hidden_pre, self.weights_hidout)))
		result = self._inverse(hidden, self.outd)

	def _inverse(self, value, outdata):
		"""
			Moore-Penrose pseudo-inverse of a matrix
		"""
		#pinv_result = theano.shared(np.linalg.pinv(value))
		return T.dot(outdata, MatrixPinv(value))