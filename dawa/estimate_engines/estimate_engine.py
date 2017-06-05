"""Classes of template and simple estimate engines."""
import math
import numpy
import os
import sys
from multiprocessing import Pool

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
	def Run(Q, x, epsilon, ratio, seed):
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


from dawa.thirdparty.Acs12.lib import EFPA

@register('my', 'transformQ')
class my_engine(estimate_engine):
    """My engine with Frank's enhanced update method."""
    
    @staticmethod
    def _exponentialMechanism(ans, hatx, Q, epsilon):
        error = []
        for q in range(len(Q)):
            appAns = sum([sum(hatx[int(lb):int(rb)+1]) * wt for wt, lb, rb in list(Q)[q]])
            error.append(abs(ans[q] - appAns))
         
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
    def _KLD(x, hatx):
        sx = sum(x)
        shatx = sum(hatx)
        # print('len x = %d' %(len(x)))
        probs = numpy.zeros(len(x))
        for i in range(len(x)):
            pi = ((x[i] + 1) * 1.0 / sx) / (1 + len(x) / sx)
            qi = ((hatx[i] + 1) * 1.0 / shatx) / (1 + len(hatx) / shatx)
            probs[i] = pi * numpy.log(pi / qi)
        return sum(probs)
    
    @staticmethod
    def _Relativeerror(ans, hatx, Q):
        error = 0
        for ind in range(len(Q)):
            q = list(Q[ind])
            appAns = sum([sum(hatx[int(lb):int(rb)+1]) * wt for wt, lb, rb in q])
            error += min(abs(ans[ind] - appAns), abs(ans[ind])) / max(1, abs(ans[ind]))
                    
        return 1 - (error / len(Q))
    
    import copy_reg
    import types
    def _pickle_method(m):
        if m.im_self is None:
            return getattr, (m.im_class, m.im_func.func_name)
        else:
            return getattr, (m.im_self, m.im_func.func_name)
    
    copy_reg.pickle(types.MethodType, _pickle_method)
    
    
    def _solve(self, x, hatx, Q, e, noisyAns, realAns, tr):
        print('trial = %d' %(tr))
        for c in range(200000):
            rid = self._exponentialMechanism(noisyAns, hatx, Q, e / len(Q) )
            q = list(Q[rid])
    
            ####### Updata #######
            # Retrive the query
            q1 = numpy.zeros(len(hatx))
            for wt, lb, rb in q:
                q1[int(lb):int(rb)+1] = wt
        
            # Query Error
            total = sum(hatx)
            qerror = (noisyAns[rid] - sum([sum(hatx[int(lb):int(rb)+1]) * wt for wt, lb, rb in q])) / total
            # print('size noisyAns = %d and size Q = %d' %(len(noisyAns), len(Q)))
            # Global Error
            globalerror = 0
            for ind in range(len(Q)):
                q = list(Q[ind])
                temp = abs((noisyAns[ind] - sum([sum(hatx[int(lb):int(rb)+1]) * wt for wt, lb, rb in q])))
                globalerror += temp / sum(hatx)
            globalerror = min(1, globalerror / len(Q))
            hatx = hatx * numpy.exp( q1 * qerror * globalerror)
            hatx *= total / sum(hatx)
        
        return [self._KLD(x, hatx), self._Relativeerror(noisyAns, hatx, list(Q)), self._Relativeerror(realAns, hatx, list(Q))]
    
    def Run(self, Q, x, epsilon):
        realAns = [sum([sum(x[int(lb):int(rb)+1]) * wt for wt, lb, rb in q]) for q in Q]
        hatx = numpy.array( [sum(x) / (1.0 * len(x))] * len(x) )
        
        # Measurements
        mykld = []
        myacc = []
        myut = []
        eps = [0.01, 0.05, 0.1, 0.5]
        trial = 3
        for e in eps:
            print('epsilon = %f' %(e))
            #noisyAns = EFPA.EFPA(realAns, 1, e)
            noisyAns = [ans + numpy.random.laplace(0.0, max(zip(*q)[0])/e) for ans in realAns for q in Q] 
            pool = Pool()
            result1 = pool.apply_async(self._solve, args = (x, hatx, Q, e, noisyAns, realAns, 1))
            result2 = pool.apply_async(self._solve, (x, hatx, Q, e, noisyAns, realAns, 2))
            result3 = pool.apply_async(self._solve, (x, hatx, Q, e, noisyAns, realAns, 3))
            answer1 = result1.get()
            answer2 = result2.get()
            answer3 = result3.get()
            
            mykld.append(float(answer1[0] + answer2[0] + answer3[0])/trial)
            myacc.append(float(answer1[1] + answer2[1] + answer3[1])/trial)
            myut.append(float(answer1[2] + answer2[2] + answer3[2])/trial)

        numpy.savetxt('myQuality.txt', numpy.array([myacc, myut, mykld]))
        print('done!')
        return hatx














