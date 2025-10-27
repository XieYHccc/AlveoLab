import time

import numpy as np
import trimesh
from trimesh.curvature import (
    discrete_gaussian_curvature_measure,
    discrete_mean_curvature_measure,
    sphere_ball_intersection,
)
import pyvista as pv

from AlveoLab.trimesh_utils import discrete_mean_curvature_measure, smooth_curvature
from AlveoLab.landmark_recognizer import LandmarkRecognizer

start_time = time.time()

m = trimesh.load_mesh('../data/models10y/0580_10yr_Maxillary_export.stl')
lr = LandmarkRecognizer(m, 'U')
m = lr._mesh

# c = trimesh.curvature.discrete_mean_curvature_measure(m, m.vertices, m.scale/10)
c = smooth_curvature(m, discrete_mean_curvature_measure(m), 30)

faces_pv = np.hstack([np.full((m.faces.shape[0], 1), 3), m.faces]).flatten()
pv_mesh = pv.PolyData(m.vertices, faces_pv)
curv = pv_mesh.curvature(curv_type="mean")

# lower, upper = np.percentile(curv, [5, 95])
lower, upper = np.percentile(c, [5, 95])

# 将离群值clamp到这个范围
curv_clamped = np.clip(c, lower, upper)

end_time = time.time()  # 计时结束
print(f"运行时间: {end_time - start_time:.2f} 秒")

# 绘制
pv_mesh.plot(scalars=curv_clamped)
