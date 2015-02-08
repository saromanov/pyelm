import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano.tensor.nlinalg import MatrixPinv
import numpy as np



class HiddenLayer:
	def __init__(self, inp, num_visible, num_hidd, bias, weights_inphid=None):
		self.weights_inphid = weights_inphid
		self.bias = bias
		self.vis = inp
		if self.weights_inphid == None:
			self.weights_inphid = theano.shared(
			np.array(
				np.random.random((num_visible, num_hidd))
				), name='Wih')
		self.hidden_pre = T.nnet.sigmoid(T.dot(self.vis, self.weights_inphid) + self.bias)
		self.result = MatrixPinv()(self.hidden_pre)

class ELM:
	""" 
		Visible - input data as pair (x,y)
		num_hidd - number of hidden neurons
	"""
	def __init__(self, visible, num_vis, num_hidd):
		self.inpdata = visible[0]
		self.outdata = visible[1]
		stream = RandomStreams()
		self.num_vis = num_vis
		self.num_out = len(self.outdata)
		self.num_hidd = num_hidd
		self.vis = T.dmatrix('vis')
		self.outd = T.dmatrix('out')
		#init as weights vector connected from input layer to hidden
		self.weights_inphid = theano.shared(
			np.array(
				np.random.random((self.num_vis, self.num_hidd))
				), name='Wih')
		self.bias = theano.shared(np.ones(self.num_hidd, dtype=theano.config.floatX))

	def _cost(self, x,y, method='lse'):
		return T.sum(T.sqrt((x - y)**2))/x.shape[0]

	def _trainInner(self):
		hidden = HiddenLayer(self.vis, self.num_vis, self.num_hidd, self.bias, weights_inphid=None)
		self.weights_hidout = T.dot(hidden.result, self.outd)
		value = T.dot(hidden.hidden_pre, self.weights_hidout)
		loss = self._cost(self.outd, value)
		return loss

	def train(self):
		""" TODO: write function with scan and updates """
		loss = self._trainInner()
		func = theano.function([self.vis, self.outd], loss)
		return loss

	def predict(self, X):
		x = T.matrix('x')
		layer = HiddenLayer(x, X.shape[1], 0, self.bias, weights_inphid=self.weights_inphid)
		result = T.dot(layer.result.T, self.weights_hidout)
		func = theano.function([x, self.vis, self.outd], result)
		return func(X, self.inpdata, self.outdata)

