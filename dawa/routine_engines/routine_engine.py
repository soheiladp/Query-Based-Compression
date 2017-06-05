"""General engine for Data- and Workload-Aware algorithms."""
import bisect
import numpy

registry = {}

def register(name):
	def wrap(cls):
		force_bound = False
		if '__init__' in cls.__dict__:
			cls.__init__.func_globals[name] = cls
			force_bound = True
		try:
			registry[name] = cls
		finally:
			if force_bound:
				del cls.__init__.func_globals[name]
		return cls
	return wrap


@register('default')
class engine(object):
	"""General two-step engine without workload transform."""

	def __init__(self, alg, p_engine = None, e_engine = None, ratio = 0.5, weights = [],
				 seed = 5, stdb=0.5, split_flag=True):
		"""Setting up engines for each step.
		
		partition_engine - the class for data aware paritioning
		query_enging - the class for counts querying
        ratio - the ratio of privacy budget used for the partition engine
		"""
		self.stdb = stdb
		self.split_flag = split_flag
		self.alg = alg
		self._seed = seed
		self._weights = weights
		self._partition_engine = p_engine
		self._estimate_engine = e_engine
		self._ratio = ratio

		numpy.random.seed(seed)
	
	def _DirectRun(self, Q, x, epsilon):
		"""Run a estimate engine without a partition engine"""
		return self._estimate_engine.Run(Q, x, epsilon)

	def Run(self, Q, x, epsilon):
		"""Run three engines in order with given epsilons to estimate a
		dataset x to answer query set Q
		
		Q - the query workload
		x - the underlying dataset
		epsilon - the total privacy budget
		"""
		n = len(x)

		if self._partition_engine is None:
			# ignore ratio when partition_engine is omitted
			return self._DirectRun(Q, x, epsilon)
		else:
			if self._ratio < 0 or self._ratio >= 1:
				raise ValueError('ratio must in range [0, 1)')

			partition = self._partition(x, epsilon)
			if self.alg == 'pdawa':
				partition = self._compress_partitions (partition, Q, x)
				# partition = self._revisePartitions(partition, x)
			# return self._rebuild_step1(partition, x, self._weights)
			counts = self._estimate_engine.Run(
						self._workload_reform(Q, partition, n),
						self._dataset_reform(x, partition),
						epsilon*(1-self._ratio), self._seed)
			return self._rebuild(partition, counts, n, self._weights)

	def Get_Partition(self):
		"""Get the data dependent partition"""
		return self._partition1

	def _partition(self, x, epsilon):
		"""Compute the data dependent partition."""
		if self._ratio == 0:
			# use identity partition if no privacy budget is
			# reserved for partitioning
			self._partition1 = [[c, c] for c in range(len(x))]
		else:
			self._partition1 = self._partition_engine.Run(x, epsilon,
														  self._ratio, self._weights, self._seed)

		return self._partition1

	def Partition_error(self, x):
		error = 0
		partition = self._partition1
		n2 = len(partition)
		bucket_avg = [sum(x[lb:(rb+1)])/float(rb-lb+1) for lb, rb in partition]

		for c in range(n2):
			lb, rb = partition[c]
			bavg = bucket_avg[c]
			error = error + sum([abs(i - bavg) for i in x[lb:(rb+1)]])

		return [n2, error/len(x)]

	def _compress_partitions(self, partition, Q, x):
		Q_split = self._split_overlap(self._sortQuerySet(Q), len(x))
		if self.split_flag:
			Q_split = self._reviseSplittedQ(Q_split, x)
		part_revised = []

		for q in Q_split:				  # no overlap in queries
			w, lbq, rbq = q[0]
			tobe_removed = []
			for p in partition:
				lbp, rbp = p
				# if rbp < lbq:         # no intersection
				# 	part_revised.append(p)
				# 	partition.remove(p)
				# elif rbq <= rbp:	  # intersection and q in p
				# 	part_revised.append([lbp, lbq-1])
				# 	part_revised.append([lbq, rbq])
				# 	if rbq+1 <= rbp:
				# 		part_revised.append([rbq+1, rbp])
				# 	partition.remove(p)
				# 	break
                #
				# else:				  # intersection and q larger than p (includes several p)
				# 	if lbp < lbq:
				# 		part_revised.append([lbp, lbq - 1])
				# 	part_revised.append([lbq, rbq])
                #
				# 	intersection_Flag = False
				# 	partition.remove(p)
				# 	pp = []
				# 	for pt in partition:
				# 		lbpt, rbpt = pt
				# 		if rbpt <= rbq:
				# 			partition.remove(pt)
				# 		elif lbpt <= rbq:
				# 			pp = pt
				# 			intersection_Flag = True ### ??? more than one intersection
				# 									 ### wont happen since there is no intersection in partitions
                #
				# 	if intersection_Flag:
				# 		lbpp, rbpp = pp
				# 		part_revised.append([rbq + 1, rbpp])
				# 		partition.remove(pp)
                #
				# 	break

				### Since it is sorted, we check in order
				### 3 cases:
				### 1, no intersection -> just copy to part_revised
				### 2, intersection in first part of query -> split [lbp, lbq-1][q]
				### 3, intersection in last part of query -> split [q][rbq+1, rbp]
				### if where p < q, just remove from partitions (bc 2 or 3 would also happen)
				### if q <= p, just [p]
				if rbp < lbq:                    # no intersection
					part_revised.append(p)
					tobe_removed.append(p)
				elif (lbq>=lbp and rbq==rbp):    # q <= p (q in p and right side mach)
					part_revised.append(p)
					partition.remove(p)
					break
				elif (lbq>=lbp and rbq<rbp):
					break
				elif (lbp>=lbq and rbp<=rbq):	 # p < q
					tobe_removed.append(p)
				else:
					if (lbp < lbq and rbp < rbq): # left intersection
						part_revised.append([lbp, lbq-1])
						tobe_removed.append(p)
					elif (lbp > lbq and rbp > rbq): # right inersection
						part_revised.append([lbq, rbq])
						partition.remove(p)
						partition.insert(0, [rbq+1, rbp])
						break

			for tbr in tobe_removed:
				partition.remove(tbr)
		# for p in partition:
		# 	part_revised.append(p)
		return part_revised

	def _reviseSplittedQ(self, Q, x):
		import math
		revisedQ = []
		for q in Q:
			wt, lb, rb = q[0]
			rb = rb+1
			cc = x[lb:rb]
			if not cc:
				print(q[0])
				continue
			stdbu = numpy.std(cc)

			if (rb - lb) > 1 and not math.isnan(float(stdbu)) and stdbu > 0.0:
				newlb = lb
				# newrb = lb
				for i in range(rb - lb):
					newrb = lb + i
					sl = numpy.std(x[newlb : newrb + 1])
					if (sl/stdbu) <= self.stdb : continue #
					revisedQ.append([[1, newlb, newrb - 1]])
					newlb = newrb
			else:
				revisedQ.append([[1, lb, rb - 1]])
			# if (rb - lb) > 1 and not math.isnan(float(stdb)) and stdb > 0.0:
			# 	m = []
			# 	for i in range(rb - lb - 1):
			# 		aa = x[lb: (lb + i + 1)]
			# 		bb = x[(lb + i + 1): rb]
			# 		if not aa or not bb: continue
			# 		stdl = numpy.std(aa)
			# 		stdr = numpy.std(bb)
			# 		# print("%s  %s  %s"%(stdb, stdl, stdr))
			# 		if math.isnan(float(stdl)) or math.isnan(float(stdr)): continue
			# 		m.append(1.0 - (stdl + stdr) / min(-1.0, (2.0 * stdb)))
            #
			# 	if len(m)==0:
			# 		revisedQ.append([[1, lb, rb - 1]])
			# 		break
			# 	bestSvalue = max(m)
			# 	bestInd = m.index(bestSvalue) + 1
			# 	if bestSvalue > 0.5 and bestInd > 0:
			# 		revisedQ.append([[1, lb, lb + bestInd-1]])
			# 		revisedQ.append([[1, lb + bestInd, rb-1]])
			# 	else:
			# 		revisedQ.append([[1, lb, rb-1]])
			# else:
			# 	revisedQ.append([[1,lb,rb-1]])
		return revisedQ

	@staticmethod
	def _split_overlap(Q, n):
		splitQ = []
		lbounds = [0]

		for q in Q:
			wt, lb, rb = q[0]
			if lb not in lbounds:
				lbounds.append(lb)
			if rb + 1 not in lbounds and rb + 1 < n:
				lbounds.append(rb + 1)
		lbounds = list(numpy.sort(lbounds))
		# # make new query set
		# for i in range(len(lbounds)-1):
		# 	splitQ.append([[1, lbounds[i], lbounds[i+1]-1]])
		rlbounds = lbounds[::-1]
		rb = n-1
		for lb in rlbounds:
			splitQ.insert(0, [[1, lb, rb]])
			rb = lb-1

		return splitQ

	@staticmethod
	def _sortQuerySet(Q):
		'''Sort Query set based on lb and rb. lb higher priority.'''
		from intervaltree import IntervalTree
		t = IntervalTree()

		for q in Q:  # Building the tree of intervals
			w, lb, rb = q[0]
			t[lb:rb + 1] = [w]

		t = sorted(t)
		Qs = []


		for v in t:
			q = [[(v.data)[0], v.begin, v.end - 1]]
			Qs.append(q)

		return Qs

	@staticmethod
	def _workload_reform(Q, partition, n):
		pass

	@staticmethod
	def _dataset_reform(x, partition):
		"""Reform a dataset x0 into x with a given parition."""
		return [sum(x[lb:(rb+1)]) for lb, rb in partition]

	@staticmethod
	def _rebuild_step1(partition, x, w):
		"""Rebuild an estimated data using uniform expansion."""
		weights = w
		estx = numpy.zeros(len(x))
		n2 = len(partition)

		for c in range(n2):
			lb, rb = partition[c]
			avg_partition = sum(x[lb:(rb+1)]) / sum(weights[lb:(rb + 1)])
			estx[lb:(rb + 1)] = numpy.multiply(avg_partition, weights[lb:(rb + 1)])

		return estx

	@staticmethod
	def _rebuild(partition, counts, n, w):
		"""Rebuild an estimated data using uniform expansion."""
		weights = w
		estx = numpy.zeros(n)
		n2 = len(counts)
		if not weights:
			for c in range(n2):
				lb, rb = partition[c]
				estx[lb:(rb + 1)] = counts[c] / float(rb - lb + 1)
		else:
			for c in range(n2):
				lb, rb = partition[c]
				counts[c] = counts[c] / sum(weights[lb:(rb + 1)])
				estx[lb:(rb + 1)] = numpy.multiply(counts[c], weights[lb:(rb + 1)])

		# for c in range(n2):
		# 	lb, rb = partition[c]
		# 	estx[lb:(rb + 1)] = counts[c] / float(rb - lb + 1)

		return estx


@register('transformQ')
class transform_engine_q(engine):
	"""The engine with workload reform implemented."""

	@staticmethod
	def _workload_reform(Q0, partition, n):
		"""Reform a workload Q0 into Q with a given parition,

		Q0 - the workload to be reformed
		partition - the given partition
		n - the size of the original domain

		An example of query reform: 
		Give a dataset with size 4, and partition is [[0], [1, 2, 3]],
		Then query x1+x2+x3+x4 will be converted to y1+y2
		     query x1+x2 will be converted y1+(1/3)y2
		"""
		partition_lb, partition_rb = zip*(partition)
		Q = []
		for q in Q0:
			q1 = []
			for wt, lb, rb in q:
				lpos = bisect.bisect_left(partition_rb, lb)
				rpos = bisect.bisect_left(partition_lb, rb)
				if lpos == rpos:
					q1.append([wt*(rb-lb+1.0)
							   /(partition_rb[lpos]-partition_lb[lpos]+1.0),
							   lpos, rpos])
				else:
					q1.append([wt*(partition_rb[lpos]-lb+1.0)
							   /(partition_rb[lpos]-partition_lb[lpos]+1.0),
							   lpos, lpos])
					q1.append([wt*(rb-partition_lb[rpos]+1.0)
							   /(partition_rb[rpos]-partition_lb[rpos]+1.0),
							   rpos, rpos])
					if lpos + 1 < rpos:
						q1.append([wt, lpos+1, rpos-1])

			Q.append(q1)

		return Q


@register('transformQtQ')
class transform_engine_qtqmatrix(engine):
	"""The engine that outputs the matrix form of Q^TQ in workload reform."""

	def __init__(self, alg, p_engine = None, e_engine = None, \
				 ratio = 0.5, max_block_size = None, weights = [], seed=50, stdb=0.5, split_flag=True):
		"""Setting up engines for each step.
		
		partition_engine - the class for data aware paritioning
		query_enging - the class for counts querying
		ratio - the ratio of privacy budget for partitioning
		max_block_size - parameter for workload_reform, see below for details.
		"""
		self._max_block_size = max_block_size
		super(type(self), self).__init__(alg, p_engine, e_engine, ratio, weights, seed, stdb, split_flag)

	def _DirectRun(self, Q, x, epsilon):
		"""Run a estimate engine without a partition engine"""
		n = len(x)
		partition = [[c,c] for c in range(n)]
		return self._estimate_engine.Run(
			self._workload_reform(Q, partition, n), x, epsilon)

	def _workload_reform(self, Q0, partition, n):
		"""Reform a workload Q0 into Q with a given parition,
		and output Q^TQ

		max_block_size - the max number of rows to be materialized
						 when computing Q^TQ. Set to n if omitted.
		"""
		n = partition[-1][-1] + 1
		n2 = len(partition)
		QtQ = numpy.zeros([n2, n2])
		if self._max_block_size is None:
			max_block_size = n
		else:
			max_block_size = self._max_block_size

		cnum = range(0, len(Q0), max_block_size)

		for c0 in cnum:
			nrow = min(len(Q0)-c0, max_block_size)
			Q0mat = numpy.zeros([nrow, n])
			for c in range(nrow):              # assign wt to corresponding items of each query
				for wt, lb, rb in Q0[c+c0]:
					Q0mat[c, lb:rb+1] = wt

			Qmat = numpy.zeros([nrow, n2])
			for c in range(n2):                # compute the weight of each partition for each query
				lb, rb = partition[c]
				Qmat[:, c] = Q0mat[:, lb:(rb+1)].mean(axis=1)

			QtQ += numpy.dot(Qmat.T, Qmat)

		return QtQ


@register('transformMyQtQ')
class transform_engine_myqtqmatrix(engine):
	"""The engine that outputs the matrix form of Q^TQ in workload reform."""

	def __init__(self, alg, p_engine=None, e_engine=None, \
				 ratio=0.5, max_block_size=None, weights = [], seed=5, stdb=0.5, split_flag=True):
		"""Setting up engines for each step.

		partition_engine - the class for data aware paritioning
		query_enging - the class for counts querying
		ratio - the ratio of privacy budget for partitioning
		max_block_size - parameter for workload_reform, see below for details.
		"""

		self._max_block_size = max_block_size
		super(type(self), self).__init__(alg, p_engine, e_engine, ratio, weights, seed, stdb, split_flag)

	# def _DirectRun(self, Q, x, epsilon):
	# 	"""Run a estimate engine without a partition engine"""
	# 	n = len(x)
	# 	partition = [[c, c] for c in range(n)]
	# 	return self._estimate_engine.Run(
	# 		self._workload_reform(Q, partition, n), x, epsilon)

	def _workload_reform(self, Q0, partition, n):
		"""Reform a workload Q0 into Q with a given parition,
		and output Q^TQ

		W - weight of each data element after preprocessing (vector size n)

		max_block_size - the max number of rows to be materialized
						 when computing Q^TQ. Set to n if omitted.
		"""
		Weights = self._weights
		# n = partition[-1][-1] + 1
		n2 = len(partition)
		QtQ = numpy.zeros([n2, n2])
		if self._max_block_size is None:
			max_block_size = n
		else:
			max_block_size = self._max_block_size

		cnum = range(0, len(Q0), max_block_size)

		for c0 in cnum:
			nrow = min(len(Q0) - c0, max_block_size)
			Q0mat = numpy.zeros([nrow, n])
			for c in range(nrow):  # assign wt to corresponding items of each query
				for wt, lb, rb in Q0[c + c0]:
					Q0mat[c, lb:rb + 1] = wt

			Q0mat = numpy.multiply(Weights, Q0mat) # multiply each element with its weight
			Wpart = numpy.ones(n)
			for c in range(n2):
				lb, rb = partition[c]
				Wpart[lb:(rb+1)] = sum(Weights[lb:(rb + 1)])

			Q0mat = numpy.divide(Q0mat, Wpart)

			Qmat = numpy.zeros([nrow, n2])
			for c in range(n2):  # compute the weight of each partition for each query
				lb, rb = partition[c]
				Qmat[:, c] = Q0mat[:, lb:(rb + 1)].sum(axis=1)

			QtQ += numpy.dot(Qmat.T, Qmat)

		return QtQ