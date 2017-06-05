"""Classes of hierarchical engines"""
import math
import numpy

import estimate_engine

@estimate_engine.register('geometricH')
class geometric_hierarchy_engine(estimate_engine.estimate_engine):
	"""Geometric hierarchy engine with the privacy budget decayed with depth."""
	
	def __init__(self, decay = math.pow(2, 1.0/3.0)):
		"""Set up the ratio of privacy budget decayed with depth.

		decay(2^(1/3)) - the ratio of privacy budget decayed with depth.
		"""
		self._decay = decay

	def	Run(self, QtQ, x, epsilon):
		n = len(x)
		m = int(math.ceil(math.log(n, 2)))
		n1 = 2**m
		x1 = numpy.zeros(n1)
		x1[:n] = x

		curepsilon = epsilon / sum([math.pow(self._decay, int((m-c)/2))
									for c in range(m+1)])
		tree = []
		tree.append([[ (0, n1),  curepsilon, sum(x), [] ]])
		Eh = [2**m * curepsilon]
		for c in range(1, m+1):
			treelv = []
			if ( (m-c) % 2 == 0 ):
				curepsilon *= self._decay

			Eh.append(2**(m-c) * curepsilon**2)
			for rng, _, _, _ in tree[-1]:
				lb, rb = rng
				mid = (lb+rb)/2
				treelv.append([(lb, mid), curepsilon,
							   numpy.random.laplace(scale=1.0/curepsilon)+sum(x1[lb:mid]),
							   []])
				treelv.append([(mid, rb), curepsilon,
							   numpy.random.laplace(scale=1.0/curepsilon)+sum(x1[mid:rb]),
							   []])

			tree.append(treelv)
		for _, curepsilon, val, dev in tree[0]:
			dev.append(curepsilon**2  * val)

		for c in range(1, m+1):
			for c1 in range(len(tree[c])):
				_, curepsilon, val, dev = tree[c][c1]
				dev.append(curepsilon**2*val + tree[c-1][c1/2][3][0])

		# dev: [alpha, Z]
		for _, _, _, dev in tree[-1]:
			dev.append(dev[0])

		for c in range(m-1, -1, -1):
			for c1 in range(len(tree[c])):
				_, _, _, dev = tree[c][c1]
				dev.append(tree[c+1][c1*2][3][1] + tree[c+1][c1*2+1][3][1])

		Eh.reverse()
		Eh = numpy.cumsum(Eh).tolist()
		Eh.reverse()

		# dev: [alpha, Z, F, beta]
		_, _, _, dev = tree[0][0]
		dev.extend([0.0, dev[1]/float(Eh[0])])
		for c in range(1, m+1):
			for c1 in range(len(tree[c])):
				_, _, _, dev = tree[c][c1]
				_, pareps, _, pardev = tree[c-1][c1/2]
				Fv = pardev[2] + pardev[3]*pareps**2
				dev.extend([Fv, (dev[1]-Fv*2**(m-c))/float(Eh[c])])

		res = numpy.zeros(n1)
		for rng, _, _, dev in tree[-1]:
			lb, _ = rng
			res[lb] = dev[-1]

		return res[:n]

@estimate_engine.register('uniformH')
class uniform_hierarchy_engine(geometric_hierarchy_engine):
	"""The hierarchy engine with no decay"""

	def __init__(self):
		super(type(self), self).__init__(1.0)

