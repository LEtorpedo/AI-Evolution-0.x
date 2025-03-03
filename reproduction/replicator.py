import numpy as np
from core.algebra.hopf_operator import HopfCore
from core.cognition.concept_lattice import ConceptLattice
import copy
import uuid
import random

class Replicator:
    """复制器：能够自我复制的基本单位"""
    
    def __init__(self, attributes=None, concepts=None):
        """初始化复制器
        
        Args:
            attributes: 初始属性集
            concepts: 初始概念集
        """
        self.attributes = attributes or {}
        self.concepts = concepts or {}
        self.age = 0
        self.parent_id = None
        self.mutation_count = 0
        self.safe_trigger_count = 0
        self.id = str(uuid.uuid4())[:8]  # 生成唯一ID
    
    def replicate(self, mutation_rate=0.1):
        """复制自身，可能发生变异
        
        Args:
            mutation_rate: 变异率
            
        Returns:
            新的复制器实例
        """
        # 复制属性和概念
        new_attributes = copy.deepcopy(self.attributes)
        new_concepts = copy.deepcopy(self.concepts)
        
        # 记录变异次数
        mutations = 0
        safe_triggers = 0
        
        # 变异过程
        if random.random() < mutation_rate:
            # 属性变异
            if random.random() < 0.5:
                # 创建新属性
                new_attr_id = f"attr_{random.randint(1000, 9999)}"
                new_attributes[new_attr_id] = {
                    "name": f"属性_{random.randint(1, 100)}",
                    "value": random.random()
                }
                mutations += 1
            else:
                # 修改现有属性
                if new_attributes:
                    attr_to_modify = random.choice(list(new_attributes.keys()))
                    new_attributes[attr_to_modify]["value"] = random.random()
                    mutations += 1
                else:
                    # 如果没有属性，创建一个
                    new_attr_id = f"attr_{random.randint(1000, 9999)}"
                    new_attributes[new_attr_id] = {
                        "name": f"属性_{random.randint(1, 100)}",
                        "value": random.random()
                    }
                    mutations += 1
                    safe_triggers += 1
            
            # 概念变异
            if random.random() < 0.3:
                if new_concepts and random.random() < 0.2:
                    # 随机移除一个概念
                    concept_to_remove = random.choice(list(new_concepts.keys()))
                    del new_concepts[concept_to_remove]
                    mutations += 1
                else:
                    # 创建一个新概念
                    new_concept_id = f"concept_{random.randint(1000, 9999)}"
                    # 从现有属性中随机选择一些作为概念的组成部分
                    if new_attributes:
                        concept_attrs = random.sample(
                            list(new_attributes.keys()),
                            min(random.randint(1, 3), len(new_attributes))
                        )
                        new_concepts[new_concept_id] = {
                            "name": f"概念_{random.randint(1, 100)}",
                            "attributes": concept_attrs
                        }
                        mutations += 1
                    else:
                        safe_triggers += 1
        
        # 创建新的复制器
        offspring = Replicator(new_attributes, new_concepts)
        offspring.parent_id = self.id
        offspring.mutation_count = mutations
        offspring.safe_trigger_count = safe_triggers
        
        return offspring
    
    def get_stats(self):
        """获取复制器的统计信息
        
        Returns:
            包含统计信息的字典
        """
        return {
            "attribute_count": len(self.attributes),
            "concept_count": len(self.concepts),
            "age": self.age,
            "id": self.id,
            "parent_id": self.parent_id,
            "mutation_count": self.mutation_count,
            "safe_trigger_count": self.safe_trigger_count
        }
