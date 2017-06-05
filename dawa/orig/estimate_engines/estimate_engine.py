"""Classes of template and simple estimate engines."""
import math
import numpy
import os
import sys

from dawa.routine_engines import routine_engine

registry = {}


class estimate_engine_info(object):
	def __init__(self, estimate_addr, engine_addr):
		self.estimate = estimate_addr
		self.engine = engine_addr


def register(name, engine_name='default'):
	def wrap(cls):
		force_bound = False
		if '__init__' in cls.__dict__:
			cls.__init__.func_globals[name] = cls
			force_bound = True
		try:
			registry[name] = estimate_engine_info(cls,
                                                  routine_engine.registry[engine_name])
		finally:
			if force_bound:
				del cls.__init__.func_globals[name]
		return cls
	return wrap

class estimate_engine(object):
	"""The template class for query engine."""

	@staticmethod
	def Run(Q, x, epsilon):
		"""Return an estimate dataset of x
		to answer Q with privacy budget epsilon.
		
		Q - the query workload
		x - the underlying dataset.
		epsilon - privacy budget.
		
		Generally speaking, the query engine can be any
		differentially privacy algorithm.
		"""
		raise NotImplementedError('A Run method must be implemented'
								  ' for a query engine.')


######################################################################
#
# Simple query engines
#
######################################################################

@register('identity')
class identity_engine(estimate_engine):
	"""Estimate a dataset by asking each of its entry with laplace mechanism."""

	@staticmethod
	def Run(Q, x, epsilon):
		return x + numpy.random.laplace(0.0, 1.0 / epsilon, len(x))


@register('privelet')
class privelet_engine(estimate_engine):
	"""Estimate a dataset by asking its wavelet parameters."""

	@staticmethod
	def _wave(x, m):
		"""Compute the wavelet parameters of dataset x with
		size 2^m.
		"""
		y = numpy.array(x)
		n = len(x)
		for c in range(m):
			y[:n] = numpy.hstack([y[:n][0::2] + y[:n][1::2],
								  y[:n][0::2] - y[:n][1::2]])
			n /= 2		
		return y
		
	@staticmethod
	def _dewave(y, m):
		"""Compute the original dataset from a set of wavelet parameters
		y with size 2^m.
		"""
		x = numpy.array(y)
		n = 2
		half_n = 1
		for c in range(m):
			x[:n:2], x[1:n:2] = (x[:half_n] + x[half_n:n])/2.0, \
								(x[:half_n] - x[half_n:n])/2.0
			n *= 2
			half_n *= 2

		return x
	
	@staticmethod
	def Run(Q, x, epsilon):
		n = len(x)
		if n <= 16:
			# don't convert to wavelet parameters for small domains
			return x + numpy.random.laplace(0.0, 1.0 / epsilon, len(x))
		else:
			m = int(math.ceil(math.log(n, 2)))
			x1 = numpy.zeros(2**m)
			x1[:n] = x
			y1 = privelet_engine._wave(x1, m) + \
				 numpy.random.laplace(0.0, (m+1.0) / epsilon, len(x1))
			return privelet_engine._dewave(y1, m)[:n]
