import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from core.algebra.hopf_operator import HopfCore
import numpy as np

def test_hopf_evolution():
    hc = HopfCore()
    original_det = np.linalg.det(hc.matrix)
    
    # 执行演化
    hc.evolve(mutation_rate=0.1)
    
    # 验证正交性
    ortho_test = hc.matrix @ hc.matrix.T
    assert np.allclose(ortho_test, np.eye(4), atol=1e-4), f"正交性破坏：\n{ortho_test}"
    
    # 验证行列式
    new_det = np.linalg.det(hc.matrix)
    assert abs(new_det - 1.0) < 1e-3, f"行列式偏离1：{new_det:.6f}"
