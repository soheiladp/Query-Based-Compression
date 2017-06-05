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

	def __init__(self, p_engine = None, e_engine = None, ratio = 0.5):
		"""Setting up engines for each step.
		
		partition_engine - the class for data aware paritioning
		query_enging - the class for counts querying
        ratio - the ratio of privacy budget used for the partition engine
		"""
		self._partition_engine = p_engine
		self._estimate_engine = e_engine
		self._ratio = ratio
	
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
			counts = self._estimate_engine.Run(
						self._workload_reform(Q, partition, n),
						self._dataset_reform(x, partition),
						epsilon*(1-self._ratio))
			return self._rebuild(partition, counts, n)

	def Get_Partition(self):
		"""Get the data dependent partition"""
		return self._partition

	def _partition(self, x, epsilon):
		"""Compute the data dependent partition."""
		if self._ratio == 0:
			# use identity partition if no privacy budget is
			# reserved for partitioning
			self._partition = [[c, c] for c in range(n)]
		else:
			self._partition = self._partition_engine.Run(x, epsilon,
                                                         self._ratio)

		return self._partition

	@staticmethod
	def _workload_reform(Q, partition, n):
		pass

	@staticmethod
	def _dataset_reform(x, partition):
		"""Reform a dataset x0 into x with a given parition."""
		return [sum(x[lb:(rb+1)]) for lb, rb in partition]
		
	@staticmethod
	def _rebuild(partition, counts, n):
		"""Rebuild an estimated data using uniform expansion."""
		estx = numpy.zeros(n)
		n2 = len(counts)
		for c in range(n2):
			lb, rb = partition[c]
			estx[lb:(rb+1)] = counts[c] / float(rb - lb + 1)

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

	def __init__(self, p_engine = None, e_engine = None, \
				 ratio = 0.5, max_block_size = None):
		"""Setting up engines for each step.
		
		partition_engine - the class for data aware paritioning
		query_enging - the class for counts querying
		ratio - the ratio of privacy budget for partitioning
		max_block_size - parameter for workload_reform, see below for details.
		"""
		self._max_block_size = max_block_size
		super(type(self), self).__init__(p_engine, e_engine, ratio)

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
			for c in range(nrow):
				for wt, lb, rb in Q0[c+c0]:
					Q0mat[c, lb:rb+1] = wt

			Qmat = numpy.zeros([nrow, n2])
			for c in range(n2):
				lb, rb = partition[c]
				Qmat[:, c] = Q0mat[:, lb:(rb+1)].mean(axis=1)

			QtQ += numpy.dot(Qmat.T, Qmat)

		return QtQ

