import numpy
import os
import sys
sys.path.append(os.path.join(os.pardir))
from dawa.cutils import cutil


def L1partition(x, epsilon, seed, weights, ratio=0.5, gethist=False):
	"""Compute the noisy L1 histogram using all interval buckets

	Args:
		x - list of numeric values. The input data vector
		epsilon - double. Total private budget
		ratio - double in (0, 1). use ratio*epsilon for partition computation and (1-ratio)*epsilon for querying
				the count in each partition
		gethist - boolean. If set to truth, return the partition directly (the privacy budget used is still ratio*epsilon)

	Return:
		if gethist == False, return an estimated data vector. Otherwise, return the partition
	"""

	numpy.random.seed(seed)
	weights = [int(w) for w in weights]

	n = len(x)
	hist = cutil.L1partition(n+1, x, epsilon, weights, ratio, numpy.random.randint(500000))
	hatx = numpy.zeros(n)
	rb = n
	if gethist:
		bucks = []
		for lb in hist[1:]:
			bucks.insert(0, [lb, rb-1])
			rb = lb
			if lb == 0:
				break
		return bucks
	else:
		for lb in hist[1:]:
			hatx[lb:rb] = max(0, sum(x[lb:rb]) + numpy.random.laplace(0, 1.0/(epsilon*(1-ratio)), 1)) / float(rb - lb)
			rb = lb
			if lb == 0:
				break

		return hatx


def L1partition_approx(x, epsilon, seed, weights, ratio=0.5, gethist=False):
	"""Compute the noisy L1 histogram using interval buckets of size 2^k

	Args:
		x - list of numeric values. The input data vector
		epsilon - double. Total private budget
		ratio - double in (0, 1) the use ratio*epsilon for partition computation and (1-ratio)*epsilon for querying
				the count in each partition
		gethist - boolean. If set to truth, return the partition directly (the privacy budget used is still ratio*epsilon)

	Return:
		if gethist == False, return an estimated data vector. Otherwise, return the partition
	"""
	numpy.random.seed(seed)
	weights = [int(w) for w in weights]

	n = len(x)
	hist = cutil.L1partition_approx(n+1, x, epsilon, weights, ratio, numpy.random.randint(500000))
	hatx = numpy.zeros(n)
	rb = n
	if gethist:
		bucks = []
		for lb in hist[1:]:
			bucks.insert(0, [lb, rb-1])
			rb = lb
			if lb == 0:
				break
		return bucks
	else:
		for lb in hist[1:]:
			hatx[lb:rb] = max(0, sum(x[lb:rb]) + numpy.random.laplace(0, 1.0/(epsilon*(1-ratio)), 1)) / float(rb - lb)
			rb = lb
			if lb == 0:
				break

		return hatx
