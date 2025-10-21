import numpy as np
import trimesh as tm
import pyvista as pv
from pyvista import plotting

from AlveoLab.orienter.obb_dental_orienter import ObbOrienter
from AlveoLab.cleft_classifier import CleftClassifier

if __name__ == '__main__':
    mesh: tm.Trimesh = tm.load_mesh('../data/unilateral/0106_Birth_Maxillary_export.stl')
    orienter = ObbOrienter(mesh)
    mesh.apply_transform(orienter.to_origin_transform_matrix)
    classifier = CleftClassifier(mesh)

    # plotting
    # -------------------------
    faces_pv = np.hstack([np.full((mesh.faces.shape[0], 1), 3), mesh.faces]).flatten()
    pv_mesh = pv.PolyData(mesh.vertices, faces_pv)

    plot_theme = plotting.themes.DocumentTheme()
    plotter = pv.Plotter(theme=plot_theme, lighting='none')
    plotter.enable_anti_aliasing()
    plotter.set_background(color='#d8dcd6')  # light grey

    # plotter.add_bounding_box(color='blue')
    # plotter.add_axes_at_origin(labels_off=True)

    # set up lighting
    light = pv.Light(color='white', light_type='camera light', intensity=0.8)
    plotter.add_light(light)

    # add dental mesh
    # plotter.add_point_labels(bottom_plane_centroid, ['Bottom Plane Centroid'], italic=True, font_size=16,
    #                          point_color='red', point_size=25, render_points_as_spheres=True, always_visible=True)

    mesh_color = [1.0, 1.0, 1.0]
    colors = np.tile(mesh_color, (mesh.faces.shape[0], 1))
    # colors[classifier.mask_left_segment] = [1, 0.2706, 0]
    # colors[classifier.mask_right_segment] = [0.2745, 0.5098, 0.7059]
    # colors[classifier.mask_forward_segment] = [0.4196, 0.5569, 0.1373]
    # colors[classifier.mask_left_segment] = [0.9, 0, 0]
    # colors[classifier.mask_right_segment] = [0, 0, 1.0]
    # if classifier.cleft_position_mask == (1, 1):
    #     colors[classifier.mask_forward_segment] = [0, 0.9, 0]
    #     path1 = pv.MultipleLines(mesh.vertices[classifier.shortest_path_left_forward])
    #     path2 = pv.MultipleLines(mesh.vertices[classifier.shortest_path_right_forward])
    #     plotter.add_mesh(path1, color='purple', line_width=15)
    #     plotter.add_mesh(path2, color='purple', line_width=15)
    # else:
    #     path = pv.MultipleLines(mesh.vertices[classifier.shortest_path_left_right])
    #     plotter.add_mesh(path, color='purple', line_width=15)
    #
    pv_mesh.cell_data["colors"] = colors
    plotter.add_mesh(pv_mesh, scalars="colors", rgb=True, opacity=1.0, specular=0.4, specular_power=10, ambient=0.2)

    # plotter.add_legend([
    #     ['A_l', 'red'],
    #     ['A_r', 'blue'],
    #     ['A_f', 'green']
    # ], face='circle', bcolor='gray')


    # add planes
    # plane_left = pv.Plane([-10, 0, 0], [-1, 0, 0], 35, 35)
    # plotter.add_mesh(plane_left, color='steelblue', opacity=0.7)
    # plane_right = pv.Plane([10, 0, 0], [1, 0, 0], 35, 35)
    # plotter.add_mesh(plane_right, color='steelblue', opacity=0.7)
    # plane_forward = pv.Plane([0, 0, 5], [0, 0, 1], 40, 40)
    # plotter.add_mesh(plane_forward, color='steelblue', opacity=0.7)

    # add peaks
    point_ids = [classifier.peak_vertex_id_left, classifier.peak_vertex_id_right, classifier.peak_vertex_id_forward]
    labels = ['Left Peak', 'Right Peak', 'Forward Peak']
    peak_colors = ['red', 'blue', 'green']
    filtered_points = [mesh.vertices[p] for p in point_ids if p is not None]
    filtered_labels = [label for p, label, c in zip(point_ids, labels, peak_colors) if p is not None]
    filtered_colors = [color for p, color in zip(point_ids, peak_colors) if p is not None]
    filtered_arrows = [pv.Arrow(p + 3 * orienter.up, -orienter.up, scale=3) for p in filtered_points]
    legend_entries = [(label, color) for label, color in zip(filtered_labels, filtered_colors)]
    for a, c in zip(filtered_arrows, filtered_colors):
        plotter.add_mesh(a, color=c)
    plotter.add_legend(legend_entries, bcolor="grey", face=pv.Arrow(), loc="upper right", size=(0.2, 0.2))
    # plotter.add_point_labels(filtered_points, filtered_labels, italic=True, font_size=14, point_color='black',
    #                          point_size=1, render_points_as_spheres=True)

    # add near gap landmarks
    # gap_landmark_ids = [classifier.gap_landmark_vertex_id_left, classifier.gap_landmark_vertex_id_right,
    #                     classifier.gap_landmark_vertex_id_forward_left, classifier.gap_landmark_vertex_id_forward_right]
    # gap_landmark_labels = ['NG_L', 'NG_R', 'NG_FL', 'NG_FR']
    #
    # filtered_gap_landmarks = [mesh.vertices[p] for p in gap_landmark_ids if p is not None]
    # filtered_gap_landmark_labels = [label for p, label in zip(gap_landmark_ids, gap_landmark_labels) if p is not None]
    # plotter.add_point_labels(filtered_gap_landmarks, filtered_gap_landmark_labels, italic=True, font_size=25,
    #                          point_color='black', point_size=40, render_points_as_spheres=True)
    # show
    plotter.show()
