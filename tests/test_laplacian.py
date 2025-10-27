import trimesh
import scipy.sparse.linalg as spla

from AlveoLab.trimesh_utils import get_cotangent_weights_laplacian_matrix, build_Ab_from_L_and_constraints

def solve_harmonic(A, b):
    # 最小二乘：解正规方程 (A^T A)Φ = A^T b，矩阵对称正定，适合 Cholesky
    ATA = A.T @ A
    ATb = A.T @ b
    # 用 SciPy 直接解；实际工程建议用 CHOLMOD（论文同源）以更快更稳
    return spla.spsolve(ATA.tocsc(), ATb)

mesh = trimesh.creation.icosphere()

L = get_cotangent_weights_laplacian_matrix(mesh)

n = mesh.vertices.shape[0]
fs_idx, bs_idx, us_idx = [0], [10], [20]
A, b = build_Ab_from_L_and_constraints(L, n, fs_idx, bs_idx, us_idx)

print(A)