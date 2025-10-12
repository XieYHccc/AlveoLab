import pyvista as pv
from pyvista import plotting

def get_dental_plotter():
    plot_theme = plotting.themes.DocumentTheme()
    plotter = pv.Plotter(theme=plot_theme, lighting='none')
    plotter.enable_anti_aliasing()
    plotter.set_background(color='#d8dcd6')  # light grey

    light = pv.Light(color='white', light_type='camera light', intensity=0.8)
    plotter.add_light(light)

    return plotter