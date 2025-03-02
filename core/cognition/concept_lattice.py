from collections import defaultdict
import numpy as np
from utils.logger import logger

class ConceptLattice:
    def __init__(self, max_age=5):
        self.concepts = defaultdict(lambda: {'attrs': set(), 'age': 0})
        self.current_id = 0
        self.max_age = max_age

    def add_concept(self, attributes):
        """添加新概念到格结构"""
        concept_id = f"C{self.current_id}"
        self.concepts[concept_id]['attrs'] = set(attributes)
        self.current_id += 1
        return concept_id

    def age_attributes(self):
        """属性老化过程"""
        for cid in list(self.concepts.keys()):
            self.concepts[cid]['age'] += 1
            if self.concepts[cid]['age'] >= self.max_age:
                del self.concepts[cid]

    def inherit_concepts(self, parent_ids, mutation_rate=0.1):
        """平衡型变异机制"""
        inherited = set()
        for pid in parent_ids:
            inherited.update(self.concepts[pid]['attrs'])
        
        new_attrs = inherited.copy()
        has_mutation = False
        
        # 强制变异机制
        if np.random.rand() < mutation_rate:
            protected_attrs = {"origin", "base"}
            mutable_attrs = [attr for attr in inherited if attr not in protected_attrs]
            
            # 如果有可变异属性，进行替换
            if mutable_attrs:
                attr_to_replace = np.random.choice(mutable_attrs)
                new_attrs.remove(attr_to_replace)
                new_attr = f"mut_{np.random.choice(['α','β','γ','δ'])}_{np.random.randint(100)}"
                new_attrs.add(new_attr)
                has_mutation = True
                logger.info(f"属性替换: {attr_to_replace} → {new_attr}")
            # 如果没有可变异属性，直接添加新属性
            else:
                new_attr = f"mut_{np.random.choice(['α','β','γ','δ'])}_{np.random.randint(100)}"
                new_attrs.add(new_attr)
                has_mutation = True
                logger.info(f"新增变异属性: {new_attr}")
        
        # 提高扩展属性的概率
        if np.random.rand() < mutation_rate * 0.8:
            ext_attr = f"ext_{np.random.randint(1000):04x}"
            new_attrs.add(ext_attr)
            has_mutation = True
            logger.info(f"新增扩展属性: {ext_attr}")
        
        # 保障机制（仅在完全没有变化时触发）
        if not has_mutation:
            safe_attr = f"safe_{np.random.randint(10)}"
            new_attrs.add(safe_attr)
            logger.warning("变异未发生，触发保障机制")
        
        return self.add_concept(new_attrs)
