import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from core.cognition.concept_lattice import ConceptLattice

def test_concept_inheritance():
    cl = ConceptLattice()
    c1 = cl.add_concept(["base"])
    
    # 多代演化
    current = c1
    for _ in range(5):
        current = cl.inherit_concepts([current], mutation_rate=0.3)
        attrs = cl.concepts[current]['attrs']
        assert "base" in attrs, "基础属性丢失"
        assert len(attrs) >= 2, "未产生新属性"

def test_attribute_aging():
    lattice = ConceptLattice(max_age=2)
    c1 = lattice.add_concept(["a"])
    
    # 第一次老化
    lattice.age_attributes()
    assert c1 in lattice.concepts, "第一代不应被清除"
    
    # 第二次老化
    lattice.age_attributes()
    assert c1 not in lattice.concepts, "超过最大年龄应被清除"
