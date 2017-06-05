'''Estimate engines with their party code'''
import math
import os
import sys

# import the wrapped mean strcture first code
from dawa.thirdparty.xiaokui import structFirst

# import Acs12 lib
from dawa.thirdparty.Acs12.lib import EFPA
from dawa.thirdparty.Acs12.lib import Clustering

import estimate_engine


@estimate_engine.register('structurefirst')
class StructureFirst(estimate_engine.estimate_engine):
    '''Estimate engine with the structure first algorithm.'''

    @staticmethod
    def Run(Q, x, epsilon):
        return structFirst.structureFirst(x, len(x), epsilon)


@estimate_engine.register('EFPA')
class efpa(estimate_engine.estimate_engine):
    '''Estimate engine with the EFPA algorithm.'''

    @staticmethod
    def Run(Q, x, epsilon):
        return EFPA.EFPA(x, 1, epsilon)


@estimate_engine.register('P-HP')
class php(estimate_engine.estimate_engine):
    '''Estimate engine with P-HP algorithm.'''

    @staticmethod
    def Run(Q, x, epsilon):
        return Clustering.Clustering(epsilon*0.5, epsilon*0.5, 1,
                                     int(math.log(len(x),2)), 2000).run(x)


