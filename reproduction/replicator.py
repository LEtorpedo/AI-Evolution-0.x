import numpy as np
from core.algebra.hopf_operator import HopfCore
from core.cognition.concept_lattice import ConceptLattice
import copy

class Replicator:
    def __init__(self, lattice=None):
        self.hopf = HopfCore()
        self.lattice = lattice or ConceptLattice()  # 共享概念格
        self.parent_concept = self.lattice.add_concept(["origin"])

    def replicate(self, mutation_rate=0.3):
        """执行一次完整的自我复制过程"""
        # 代数结构使用较小的扰动
        self.hopf.evolve(mutation_rate * 0.5)
        
        # 概念继承使用更高的变异率
        child_concept = self.lattice.inherit_concepts(
            [self.parent_concept],
            mutation_rate=mutation_rate
        )
        
        # 创建子代时共享同一个概念格
        offspring = Replicator(lattice=self.lattice)
        offspring.hopf = copy.deepcopy(self.hopf)
        offspring.parent_concept = child_concept
        
        return offspring
