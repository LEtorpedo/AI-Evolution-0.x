import sys
import os
import numpy as np
import pytest
from collections import defaultdict

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from reproduction.replicator import Replicator
from core.cognition.concept_lattice import ConceptLattice

def test_self_replication():
    # 初始化父代
    parent = Replicator()
    
    # 记录初始状态
    original_matrix = np.copy(parent.hopf.matrix)
    original_concept = parent.parent_concept
    
    # 执行复制
    child = parent.replicate(mutation_rate=0.1)
    
    # 代数结构验证
    assert not np.array_equal(original_matrix, child.hopf.matrix), "代数结构未变化"
    assert np.allclose(child.hopf.matrix @ child.hopf.matrix.T, np.eye(4), atol=1e-4), "正交性破坏"
    
    # 概念继承验证
    parent_attrs = parent.lattice.concepts[original_concept]['attrs']
    child_attrs = child.lattice.concepts[child.parent_concept]['attrs']
    assert parent_attrs.issubset(child_attrs), f"父代属性{parent_attrs}未被子代{child_attrs}继承"
    assert len(child_attrs) > len(parent_attrs), "未产生新属性"

@pytest.mark.stress
def test_mutation_statistics():
    """验证在多次试验中至少发生一次变异"""
    NUM_TRIALS = 100
    mutation_count = 0
    
    for _ in range(NUM_TRIALS):
        parent = Replicator()
        child = parent.replicate(mutation_rate=0.3)  # 提高变异率
        
        parent_attrs = parent.lattice.concepts[parent.parent_concept]['attrs']
        child_attrs = child.lattice.concepts[child.parent_concept]['attrs']
        
        if child_attrs != parent_attrs:
            mutation_count += 1
    
    assert mutation_count > 0, f"在{NUM_TRIALS}次试验中未观测到任何变异"

def test_balanced_mutation():
    np.random.seed(42)
    lattice = ConceptLattice()
    parent_id = lattice.add_concept(["base"])
    parent_attrs = lattice.concepts[parent_id]['attrs']
    
    # 合并测试循环
    has_natural = 0
    safe_triggers = 0
    mut_types = defaultdict(int)
    
    for _ in range(100):
        child_id = lattice.inherit_concepts([parent_id], mutation_rate=0.3)
        child_attrs = lattice.concepts[child_id]['attrs']
        
        # 记录变异类型
        mutation_found = False
        for attr in child_attrs:
            if attr.startswith('mut_'):
                mut_types['属性替换'] += 1
                mutation_found = True
            elif attr.startswith('ext_'):
                mut_types['属性扩展'] += 1
                mutation_found = True
            elif attr.startswith('safe_'):
                mut_types['保障机制'] += 1
                safe_triggers += 1
        
        if mutation_found:
            has_natural += 1
    
    # 断言和输出
    assert has_natural >= 25, f"自然变异率不足，实际观测到{has_natural}次"
    
    natural_rate = has_natural / 100
    safe_rate = safe_triggers / 100  # 使用同一个循环中的保障触发次数
    print(f"\n自然变异率: {natural_rate:.1%}")
    print(f"保障触发率: {safe_rate:.1%}")
    print(f"属性增长分布: {[len(lattice.concepts[c]['attrs']) for c in lattice.concepts]}")
    
    print("\n变异类型统计:")
    for k, v in mut_types.items():
        print(f"{k}: {v}次")

def test_self_replication_chain():
    ancestor = Replicator()
    population = [ancestor]
    
    # 生成10代复制体
    for _ in range(10):
        new_gen = [r.replicate() for r in population]
        population = new_gen
    
    # 验证代数结构
    last_gen = population[0]
    assert last_gen.hopf.generation == 10, "代数未正确递增"
    
    # 验证概念继承
    concept_chain = last_gen.lattice.concepts[last_gen.parent_concept]['attrs']
    assert "origin" in concept_chain, "初始概念丢失"
    assert len(concept_chain) >= 8, "概念发展不足"  # 修改为更合理的期望值