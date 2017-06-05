"""Interface APIs"""
import glob
import math
import numpy
import os

if os.__name__ == 'dawa':
	from dawa.partition_engines import partition_engine
	from dawa.estimate_engines import estimate_engine
else:
	from partition_engines import partition_engine
	from estimate_engines import estimate_engine
	
# The dict predefined algorithms: name: (p_name, p_argv, e_name, e_argv, ratio)
# definition of each parameters are same as AlgorithmBuild (below)
# the names should be in lower case letters
algorithms = {
				'dawa': ('l1approx', [], 'greedyH', [], 0.25),
				'efpa': (None, None, 'EFPA', [], None),
				'greedyh': (None, None, 'greedyH', [], None),
				'identity': (None, None, 'identity', [], None),
				'l1partition': ('l1partition', None, 'identity', [], 0.5),
				'l1approx': ('l1approx', None, 'identity', [], 0.5),
				'mwem': (None, None, 'mwem', [], None),
				'p-hp': (None, None, 'P-HP', [], None),
				'privelet': (None, None, 'privelet', [], None),
				'structurefirst': (None, None, 'structurefirst', [], None),
			 }


def AlgorithmBuilder(p_name, p_argv, e_name, e_argv, ratio=0.5):
    """Build an algorithm with given partition and estimation engine.
    
    p_name - name of a partition engine
    p_argv - initialization parameters for the partition engine
    e_name - name of a estimation engine
    e_argv - initialization parameters for the estimation engine
    ratio - ratio of privacy budget to be used for the partition engine
    """
    e_engine = estimate_engine.registry[e_name]
    if p_name is None:
		return e_engine.engine(None, e_engine.estimate(*e_argv), ratio)
    else:
        p_engine = partition_engine.registry[p_name]
        return e_engine.engine(p_engine(*p_argv),
                               e_engine.estimate(*e_argv), ratio)


def Algorithm(name):
    """Build an algorithm with predefined configuations in dict `algorithm'."""
    return AlgorithmBuilder(*algorithms[name.lower()])


def Test(Q, x, epsilon, algs, ntest=1, seed=None):
    """Test a set of algorithms against a given queryset and dataset. Return 
    L0(max), L1(absolute sum), and L2(root of sum of square) distance between
    the estimated answer and the true answer.
    
    Q - the query workload (a list of queries)
        Each query is a list of triplets: 
        [ [w1, lb1, rb1], ... [wk, lbk, rb    k] ],
        which represents the query w1 ( x[lb1] + ... + x[rb1] ) 
        + ... + wk ( x[lbk] + ... + x[rbk] )
    x - the underlying data vector (a list of numbers)
    algs - a list of algorithms to be use.
           Each entry of the list is either a string (name of a predefined
           algorithm in dict `algorithm') or a tuple of size 4 or 5
		   (parameters for AlgorithmBuilder)
    ntest(1) - the total number of test runs
    seed(None) - the random seed

    Return:
    result - dictionary with three keys: l0, l1, l2. Each key associates with 
             a 2d array of size ntest x len(algs). Each row of the array stores
             results of all input algorithms of one test run.
    """
    if seed is not None:
        numpy.random.seed(seed)
    nalgs = len(algs)
    result = {'l0': numpy.zeros([ntest, nalgs]),
              'l1': numpy.zeros([ntest, nalgs]),
              'l2': numpy.zeros([ntest, nalgs])}
    true_ans = numpy.array([sum([sum(x[lb:rb+1])*wt for wt,lb,rb in q ]) 
                            for q in Q])
    for c in range(ntest):
        for c1 in range(nalgs):
            alg = algs[c1]
            if type(alg) == str:
                hatx = Algorithm(alg).Run(Q, x, epsilon)
            elif type(alg) == tuple and ( len(alg) == 5 or len(alg) == 4 ):
                hatx = AlgorithmBuilder(*alg).Run(Q, x, epsilon)
            else:
                raise ValueError('Expect a string or a tuple of size 4 or 5'
                                 ' but received %s' % str(alg))

            est_ans = numpy.array([sum([sum(hatx[lb:rb+1])*wt for 
                                   wt,lb,rb in q ]) for q in Q])
            diff = abs(true_ans - est_ans)
            result['l0'][c, c1] = max(diff)
            result['l1'][c, c1] = sum(diff)
            result['l2'][c, c1] = math.sqrt(sum(diff**2))

    return result

############################################################
# Interactive Help Functions
############################################################


def PartitionEngine():
    """List names and docstrings of all partition engines."""
    for key, val in partition_engine.registry.iteritems():
        print key, '\n\t', val.__doc__


def EstimateEngine():
    """List names, docstrings  and initialization docstrings \
    of all estimation engines.
    """
    for key, val in estimate_engine.registry.iteritems():
        print key, '\n\t', val.estimate.__doc__
        if '__init__' in val.estimate.__dict__:
            print '\tinitialization:', val.estimate.__init__.__doc__

