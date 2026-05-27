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
        too shallow or too small, then selects at most one original peak per
        surviving basin (the highest one). Peaks not in the tooth vertex set are
        silently ignored. Falls back to all original peaks if none map to any basin.
        """
        vertex_indices, adjacency = self._build_tooth_vertex_adjacency()
        local_lookup = {int(v): i for i, v in enumerate(vertex_indices)}

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

        # Map each original peak to its basin label (skip peaks outside this tooth)
        basin_to_peaks = {}
        for peak in self.peaks:
            local_idx = local_lookup.get(int(peak.index))
            if local_idx is None:
                continue
            basin_label = int(merged_watershed.basin_labels[local_idx])
            basin_to_peaks.setdefault(basin_label, []).append(peak)

        # Keep the highest original peak per basin
        filtered_peaks = [
            max(basin_peaks, key=lambda p: p.point @ occlusal_axis)
            for basin_peaks in basin_to_peaks.values()
        ]

        # Fallback: if no original peak maps to any basin, keep all of them
        if not filtered_peaks:
            filtered_peaks = list(self.peaks)

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

