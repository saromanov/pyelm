import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano.tensor.nlinalg import MatrixPinv
import numpy as np



class HiddenLayer:
	def __init__(self, inp, num_visible, num_hidd, bias, weights_inphid=None, weights_hidout=None):
		self.weights_inphid = weights_inphid
		self.weights_hidout = weights_hidout
		self.bias = bias
		self.vis = inp
		if self.weights_inphid == None:
			self.weights_inphid = theano.shared(
			np.array(
				np.random.random((num_vis, num_hidd))
				), name='Wih')
		if self.weights_hidout == None:
			self.weights_hidout = theano.shared(
			np.array(
				np.random.random((num_hidd, num_out))
				), name='Who')

		hidden_pre = T.nnet.sigmoid(T.dot(self.vis, self.weights_inphid) + self.bias)
		self.result = self._inverse(self.weights_hidout, hidden_pre)

	def _inverse(self, value, outdata):
		return T.dot(outdata, MatrixPinv()(value))

class ELM:
	""" 
		Visible - input data as pair (x,y)
		num_hidd - number of hidden neurons
	"""
	def __init__(self, visible, num_hidd):
		self.inpdata = visible[0]
		self.outdata = visible[1]
		stream = RandomStreams()
		self.num_vis = len(self.inpdata)
		self.num_out = len(self.outdata)
		self.num_hidd = num_hidd
		self.vis = T.dmatrix('vis')
		self.outd = T.dmatrix('out')
		#init as weights vector connected from input layer to hidden
		self.weights_inphid = theano.shared(
			np.array(
				np.random.random((self.num_vis, self.num_hidd))
				), name='Wih')
		#main training param
		self.weights_hidout = theano.shared(
			np.array(
				np.random.random((self.num_hidd, self.num_out))
				), name='Who')
		self.bias = theano.shared(np.ones(self.num_hidd, dtype=theano.config.floatX))

	def train(self, iters):
		""" TODO: write function with scan and updates """
		hidden = HiddenLayer(self.vis, self.num_vis, self.num_hidd, self.bias, weights_inphid=self.weights_inphid,\
			weights_hidout=self.weights_hidout)
		diff = hidden.result - self.outd.T
		#cost = T.sum(T.dot(diff, diff))
		func = theano.function([self.vis, self.outd], diff)
		res = func(self.inpdata, self.outdata)
