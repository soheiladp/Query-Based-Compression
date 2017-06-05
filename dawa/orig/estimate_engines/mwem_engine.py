"""Classes of MWEM engines"""

import math
import numpy

import estimate_engine

@estimate_engine.register('mwem_simple', 'transformQ')
class mwem_engine(estimate_engine.estimate_engine):
	"""Basic Multiplictive weight mechanism engine."""

	def __init__(self, nrounds = 10, ratio = 0.5):
		"""Set up basic parameters for MWEM.
		
		nrounds(10) - how many rounds are MWEM run.
		ratio(0.5) - the ratio of privacy budget used for query selection.
		"""
		self._nrounds = nrounds
		self._ratio = ratio
		if ratio <= 0 or ratio >= 1:
			raise ValueError('ratio must in range (0, 1)')
	
	@staticmethod	
	def _exponentialMechanism(x, hatx, Q, epsilon):
		"""Choose the worst estimated query (set) using the exponential mechanism.

		x - true data vector
		hatx - estimated data vector
		Q - the queries to be chosen from
		epsilon - private parameter
		"""
		diffx = x - hatx
		# compute the error of each query 
		error = numpy.array([ abs(sum([diffx[lb:rb+1].sum() * wt
							  for wt, lb, rb in q ])) for q in Q ])
		# compute the sampling probability
		merr = max(error)
		# cheating case
		if epsilon == -1:
			return error.argmax()
	
		prob = numpy.exp( epsilon* (error - merr) / 2.0 )

		sample = numpy.random.random() * sum(prob)
		for c in range(len(prob)):
			sample -= prob[c]
			if sample <= 0:
				return c
		
		return len(prob)-1

	@staticmethod
	def _update(hatx, q, est):
		"""basic multiplicative weight update."""
		total = sum(hatx)
	
		error = est - sum([hatx[lb:rb+1].sum() * wt for wt, lb, rb in q ])
		q1 = numpy.zeros(len(hatx))
		for wt, lb, rb in q:
			q1[lb:rb+1] = wt
		
		hatx = hatx * numpy.exp( q1 * error / (2.0 * total) )
		hatx *= total / sum(hatx)
	
		return hatx


	def Run(self, Q, x, epsilon):
		Rowrem = range(len(Q))
		n = len(x)
		Q1 = list(Q)

		# here we assume the total count is known
		hatx = numpy.array( [sum(x) / (1.0 * len(x))] * len(x) )

		selepsilon = epsilon * self._ratio
		queryepsilon = epsilon - selepsilon

		for c in range(self._nrounds):
			rid = self._exponentialMechanism(x, hatx, numpy.array(Q1)[Rowrem], selepsilon / self._nrounds )
			q = list(Q1[Rowrem[rid]])
			sens = max(zip(*q)[0])
			est = sum([sum(x[lb:rb+1]) * wt for wt, lb, rb in q ]) \
				  + numpy.random.laplace(0.0, sens * self._nrounds / queryepsilon, 1)
			del Rowrem[rid]	
			hatx = self._update(hatx, q, est)

		return hatx


@estimate_engine.register('mwem', 'transformQ')
class mwem_engine_frank(mwem_engine):
	"""MWEM engine with Frank's enhanced update method."""

	def __init__(self, nrounds = 10, ratio = 0.5, updateround = 100):
		"""Set up parameters for MWEM.

		nrounds(10) - how many rounds are MWEM run.
		ratio(0.5) - the ratio of privacy budget used for query selection.
		updateround(100) - the number of iterations in each update.
		"""
		self._updateround = updateround
		super(type(self), self).__init__(nrounds, ratio)

	def _update(self, hatx, estlist):
		"""Update using all historical results for multiple rounds"""
		total = sum(hatx)
		for c in range(self._updateround):
			for q, est in estlist:
				q1 = numpy.zeros(len(hatx))
				for wt, lb, rb in q:
					q1[lb:rb+1] = wt

				error = est - sum([hatx[lb:rb+1].sum() * wt for wt, lb, rb in q ])
				hatx = hatx * numpy.exp( q1 * error / (2.0 * total) )
				hatx *= total / sum(hatx)

		return hatx

	def Run(self, Q, x, epsilon):
		estlist = []
		Rowrem = range(len(Q))
		n = len(x)
		Q1 = list(Q)

		# here we assume the total count is known
		hatx = numpy.array( [sum(x) / (1.0 * len(x))] * len(x) )

		selepsilon = epsilon * self._ratio
		queryepsilon = epsilon - selepsilon

		for c in range(self._nrounds):
			rid = self._exponentialMechanism(x, hatx, numpy.array(Q1)[Rowrem], selepsilon / self._nrounds )
			q = list(Q1[Rowrem[rid]])
			sens = max(zip(*q)[0])
			est = sum([sum(x[lb:rb+1]) * wt for wt, lb, rb in q ]) \
				  + numpy.random.laplace(0.0, sens * self._nrounds / queryepsilon, 1)
			del Rowrem[rid]	
			estlist.append([q, est])
			hatx = self._update(hatx, estlist)

		return hatx

