'''Query generating functions.'''

import numpy


def RandomRange(n, m):
    '''Uniformly randomly generate m range queries in domain n.'''
    numpy.random.seed(5)
    Q = []
    for c in range(m):
        lb, rb = sorted(numpy.random.randint(0, n, 2))
        Q.append([[1, lb, rb]])

    return numpy.array(Q)


def RandomCenter(n, m, k=1, stdev=32):
    '''Uniformly randomly pick k centers in domain n,
    generate m queries around each center. For each query,
    the distance left and right boundary to the center follows
    Gaussian distribution with the condition of x>=0, respectively.
    '''
    numpy.random.seed(5)
    Q = []
    for center in numpy.random.randint(0, n, k):
        for c in range(m):
            lb = max(0, center - abs(int(numpy.random.normal(0, stdev))))
            rb = min(n-1, center + abs(int(numpy.random.normal(0, stdev))))
            Q.append([[1, lb, rb]])

    return numpy.array(Q)


def FixSize(n, m, length):
    '''Uniformly randomly generate m random range queries with fix size'''
    Q = []
    for c in range(m):
        lb = numpy.random.randint(0, n-length+1, 1)
        Q.append([[1, lb, lb+length-1]])

    return numpy.array(Q)


def Identity(n):
    '''Return the workload of all identity queries.'''
    return numpy.array([[[1, c, c]] for c in range(n)])

