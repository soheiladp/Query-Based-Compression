"""Classes of MWEM engines"""

import math
import numpy
from multiprocessing import pool

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


	def Run(self, Q, x, epsilon, ratio):
		Rowrem = range(len(Q))
		n = len(x)
		Q1 = list(Q)

		# here we assume the total count is known
		hatx = numpy.array( [sum(x) / (1.0 * len(x))] * len(x) )

		selepsilon = epsilon * ratio

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
    def __init__(self, nrounds = 90, ratio = 0.5, updateround = 100):
        self._updateround = updateround
        super(type(self), self).__init__(nrounds, ratio)
    
    def _update(self, hatx, estlist):
        total = sum(hatx)
        for c in range(self._updateround):
            for q, est in estlist:
                q1 = numpy.zeros(len(hatx))
                for wt, lb, rb in q:
                    q1[lb:rb+1] = wt
                
                error = est - sum(sum([hatx[int(lb):int(rb+1)]]) * wt for wt, lb, rb in q)
                hatx = hatx * numpy.exp( q1 * error / (2.0 * total) )
                hatx *= total / sum(hatx)
        return hatx
    
    @staticmethod
    def _KLD(x, hatx):
        sx = sum(x)
        shatx = sum(hatx)
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
    
    def Run(self, Q, x, epsilon):
        estlist = []
        Rowrem = range(len(Q))
        n = len(x)
        Q1 = list(Q)
        hatx = numpy.array( [sum(x) / (1.0 * len(x))] * len(x) )
        realAns = [sum([sum(x[int(lb):int(rb)+1]) * wt for wt, lb, rb in q]) for q in Q]

        trial = 3 
        eps = [0.01, 0.05, 0.1, 0.5]
        T = [a*10 for a in range(1,21)]
        
        for e in eps:
            noise = [numpy.random.laplace(0.0, max(zip(*q)[0])/e) for q in Q]
            for i in range(len(Q)):
                sens = max(zip(*q)[0])
                noisyAns = [sum([sum(x[int(lb):int(rb)+1]) * wt for wt, lb, rb in q]) + noise[i]]
                
            MWEM_kld = []
            MWEM_ut = []
            MWEM_acc = []
        
            for t in range(trial):
                kld = []
                acc = []
                ut = []
                for itera in T:
                    ac = 100
                    u = 0
                    kl = 0
                    for c in range(itera):
                        rid = self._exponentialMechanism(x, hatx, numpy.array(Q1)[Rowrem], e / itera )
                        q = list(Q1[Rowrem[rid]])
                        sens = max(zip(*q)[0])
                        est = sum([sum(x[int(lb):int(rb+1)]) * wt for wt, lb, rb in q ]) + noise[Rowrem[rid]]
                        del Rowrem[rid]	
                        estlist.append([q, est])
                        hatx = self._update(hatx, estlist)
                    a = self._Relativeerror(noisyAns, hatx, list(Q))
                    if (a < acc):
                        ac = a
                        u = self._Relativeerror(realAns, hatx, list(Q))
                        kl = self._KLD(x, hatx)
                kld.append(kl)
                acc.append(ac)
                ut.append(u)
        
            MWEM_kld.append(float(sum(kld))/trial)
            MWEM_ut.append(float(sum(tu))/trial)
            MWEM.acc.append(float(sum(acc))/trial)
        mwemQuality = numpy.array([MWEM_acc, MWEM_ut, MWEM_kld])
        numpy.savetxt('mwemQuality.txt', mwemQuality)
        
        return hatx
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

