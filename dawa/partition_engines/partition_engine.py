"""Classes of partition engines."""
import l1partition

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

class partition_engine(object):
	"""The template class for partition engines."""

	@staticmethod
	def Run(x, epsilon, ratio):
		"""Run templated for partition engine.

		x - the input dataset
		epsilon - the total privacy budget
		ratio - the ratio of privacy budget used for partitioning.
		"""
		raise NotImplementedError('A Run method must be implemented'
								  ' for a partition engine.')


@register('l1partition')
class l1_partition(partition_engine):
	"""Use the L1 partition method."""

	@staticmethod
	def Run(x, epsilon, ratio, weights, seed):
		return l1partition.L1partition(x, epsilon, seed, weights, ratio, gethist=True)


@register('l1approx')
class l1_partition_approx(partition_engine):
	"""Use the approximate L1 partition method."""

	@staticmethod
	def Run(x, epsilon, ratio, weights, seed):
		return l1partition.L1partition_approx(x, epsilon, seed, weights, ratio, gethist=True)
