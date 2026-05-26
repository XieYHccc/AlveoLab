import numpy as np

from AlveoLab.utils import mask_or, LazyAttribute
from AlveoLab.trimesh_utils import get_oriented_bounding_box
import AlveoLab.math.geometry as geom
from AlveoLab.peak import Peak
from AlveoLab.mhb.cusp_detection import watershed_basins, merge_spurious_basins


class Tooth:
    _WATERSHED_MIN_DEPTH_RATIO = 0.04
    _WATERSHED_MIN_BASIN_SIZE = 3

    def __init__(self, area_groups, key, palmer=None):
        self.area_groups = area_groups
        self.key = key
        self.palmer = palmer
        self.mesh = area_groups[0].mesh

        self.mask = mask_or(*(s.mask for s in self.area_groups))
        self.orienter = None
        self.parent_orienter = self.area_groups[0].quadratic.orienter
        self.centre_of_mass = geom.center_of_mass(self.mesh.triangles_center[self.mask])
        self.quadratic = area_groups[0].quadratic

        seen = set()
        self.peaks = []
        for s in self.area_groups:
            for p in s.peaks:
                if id(p) not in seen:
                    seen.add(id(p))
                    self.peaks.append(p)

        self.filtered_peaks = None

        root = self.quadratic.get_root_at(self.centre_of_mass)
        self.tangent, self.distal, self.buccal = [
            geom.normalize_vector(method(root=root)) for method in (
                self.quadratic.tangent_at,
                self.quadratic.distal_at,
                self.quadratic.buccal_at,
            )
        ]

    @LazyAttribute
    def area(self):
        return np.sum([g.area for g in self.area_groups])

    @LazyAttribute
    def obb(self):
        submesh = self.mesh.submesh([self.mask], append=True)
        return get_oriented_bounding_box(submesh)

    def filter_peaks_watershed(self, occlusal_axis):
        """Watershed-based peak filtering stored in self.filtered_peaks.

        Runs watershed on the tooth's vertex elevation, merges basins that are
        too shallow (depth < _WATERSHED_MIN_DEPTH_RATIO * elevation range) or
        too small, then emits one Peak per surviving basin at the vertex with the
        highest occlusal elevation. At least one peak is always kept.
        """
        vertex_indices, adjacency = self._build_tooth_vertex_adjacency()

        elevation = self.mesh.vertices[vertex_indices] @ occlusal_axis
        # negate so cusp tips (elevation maxima) become watershed minima
        height_function = -elevation

        initial_watershed = watershed_basins(adjacency, height_function)

        height_range = float(np.ptp(height_function))
        min_depth = max(height_range * self._WATERSHED_MIN_DEPTH_RATIO, 1e-8)

        merged_watershed, _ = merge_spurious_basins(
            adjacency,
            height_function,
            initial_watershed,
            min_basin_depth=min_depth,
            min_basin_size=self._WATERSHED_MIN_BASIN_SIZE,
        )

        filtered_peaks = []
        for basin_min_local_idx in merged_watershed.basin_minima_local_indices:
            basin_local_indices = np.flatnonzero(
                merged_watershed.basin_labels == basin_min_local_idx
            )
            # vertex with the lowest height_function = highest occlusal elevation
            best_local_idx = int(
                basin_local_indices[np.argmin(height_function[basin_local_indices])]
            )
            global_idx = int(vertex_indices[best_local_idx])
            p = Peak(self.mesh.vertices[global_idx], global_idx)
            p.occlusal = occlusal_axis
            filtered_peaks.append(p)

        # defensive fallback: merge_spurious_basins always leaves ≥1 basin,
        # but guard against unexpected edge cases
        if not filtered_peaks:
            best_local = int(np.argmin(height_function))
            global_idx = int(vertex_indices[best_local])
            p = Peak(self.mesh.vertices[global_idx], global_idx)
            p.occlusal = occlusal_axis
            filtered_peaks = [p]

        self.filtered_peaks = filtered_peaks

    def _build_tooth_vertex_adjacency(self):
        """Return (vertex_indices, adjacency) for the tooth's submesh vertices.

        vertex_indices: global vertex indices belonging to this tooth's faces.
        adjacency: list of local-index arrays, one per vertex, restricted to
                   neighbors also inside the tooth.
        """
        vertex_indices = np.unique(self.mesh.faces[self.mask])
        local_lookup = {int(v): i for i, v in enumerate(vertex_indices)}

        adjacency = []
        for global_idx in vertex_indices:
            neighbor_local_indices = sorted(
                local_lookup[int(nb)]
                for nb in self.mesh.vertex_neighbors[int(global_idx)]
                if int(nb) in local_lookup
            )
            adjacency.append(np.asarray(neighbor_local_indices, dtype=np.int64))

        return vertex_indices, adjacency

