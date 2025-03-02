import numpy as np
from scipy.linalg import expm

class HopfCore:
    def __init__(self):
        self.matrix = np.eye(4)
        self.generation = 0

    def _initialize_matrix(self):
        """生成带随机扰动的初始矩阵"""
        base = np.eye(4)
        noise = np.random.normal(0, 0.1, (4,4))
        return base + noise

    def evolve(self, mutation_rate=0.1):
        """精确保持行列式的演化方法"""
        self.generation += 1
        # 生成斜对称扰动矩阵
        skew_sym = np.random.normal(0, mutation_rate, (4,4))
        skew_sym = (skew_sym - skew_sym.T)/2
        
        # 生成正交扰动矩阵
        delta_Q = expm(skew_sym)
        
        # 应用扰动并保持行列式为1
        new_matrix = self.matrix @ delta_Q
        det = np.linalg.det(new_matrix)
        new_matrix /= det ** (1/4)
        
        # 强制正交化（QR重新正交）
        q, r = np.linalg.qr(new_matrix)
        self.matrix = q @ np.diag(np.sign(np.diag(r)))
        
        return self.matrix

    def validate(self):
        """验证Hopf代数结构"""
        condition1 = np.allclose(self.matrix @ self.matrix.T, np.eye(4), atol=1e-3)
        condition2 = np.linalg.det(self.matrix) > 0
        return condition1 and condition2
