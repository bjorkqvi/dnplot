from .plot_functions.plot_functions import plot_gridded_magnitude, plot_object_points, plot_masked_points, plot_mask, plot_arrows, plot_coastline, plot_polar_spectra
import matplotlib.pyplot as plt
from .defaults import default_variable
from dnora import DnoraObjectType
from matplotlib.widgets import Slider
import numpy as np
from cartopy import crs as ccrs
def dnora_grid_plotter(fig_dict, model) -> dict:
        """Plot the depth information and set output points etc."""
        grid = model.grid()
        fig_dict = plot_gridded_magnitude(fig_dict, grid.x(native=True), grid.y(native=True), grid.topo(), cmap = default_variable['topo']['cmap'])
        fig_dict = plot_mask(fig_dict, grid.x(native=True), grid.y(native=True), grid.land_mask())

        fig_dict['ax'].set_xlabel(grid.x_str)
        fig_dict['ax'].set_ylabel(grid.y_str)
        fig_dict['cbar'].set_label('Depth [m]')
        fig_dict['ax'].set_title(f"{grid.name} {grid.ds().attrs.get('source', '')}")

        masks_to_plot = ['boundary_mask', 'output_mask']
        fig_dict = plot_masked_points(fig_dict, grid, masks_to_plot=masks_to_plot)

        objects_to_plot=[DnoraObjectType.Forcing, DnoraObjectType.Boundary, DnoraObjectType.Spectra, DnoraObjectType.WaterLevel, DnoraObjectType.OceanCurrent, DnoraObjectType.IceForcing, DnoraObjectType.WaveSeries] 
        fig_dict = plot_object_points(fig_dict, model, objects_to_plot=objects_to_plot)

        fig_dict.get('ax').legend()

        return fig_dict

def dnora_forcing_plotter(fig_dict, model) -> dict:
    def update_plot(val):
        nonlocal fig_dict
        nonlocal figure_initialized
        fig_dict = plot_gridded_magnitude(fig_dict, forcing.x(native=True), forcing.y(native=True), forcing.magnitude()[val, :, :], vmax = np.max(forcing.magnitude()),  vmin=0, cmap = default_variable['ff']['cmap'])
        fig_dict = plot_coastline(fig_dict)
        fig_dict = plot_arrows(fig_dict, forcing.x(native=True), forcing.y(native=True), forcing.u()[val, :, :], forcing.v()[val, :, :])
        if not figure_initialized:
            masks_to_plot = ['output_mask']
            fig_dict = plot_masked_points(fig_dict, grid, masks_to_plot=masks_to_plot)
            fig_dict.get('ax').legend()
        fig_dict['ax'].set_title(f"{forcing.time(datetime=False)[val]} {forcing.name} {forcing.ds().attrs.get('source', '')}")    
        figure_initialized = True
    forcing = model.forcing()
    grid = model.grid()
    figure_initialized = False
    if len(forcing.time()) > 1:
        ax_slider = plt.axes([0.17, 0.05, 0.65, 0.03])
        time_slider = Slider(ax_slider, 'time_index', 0, len(forcing.time())-1, valinit=0, valstep=1)
        time_slider.on_changed(update_plot)
    
    update_plot(0)
    fig_dict['ax'].set_xlabel(forcing.x_str)
    fig_dict['ax'].set_ylabel(forcing.y_str)
    fig_dict['cbar'].set_label('Wind speed [m/s]')

    plt.show(block=True)

    return fig_dict

def dnora_boundary_plotter(fig_dict, model) -> dict:
    def update_plot(val):
        nonlocal fig_dict
        nonlocal figure_initialized
        fig_dict = plot_polar_spectra(fig_dict, boundary.spec()[sliders['time'].val, sliders['inds'].val,:,:], boundary.freq(), boundary.dirs())
        
        
        fig_dict['ax'].set_title(f"{forcing.time(datetime=False)[sliders['time'].val]} {forcing.name} {forcing.ds().attrs.get('source', '')}")    
        figure_initialized = True
    forcing = model.forcing()
    boundary = model.boundary()
    grid = model.grid()
    figure_initialized = False
    sliders = {}
    if len(forcing.time()) > 1:
        ax_slider = plt.axes([0.17, 0.05, 0.65, 0.03])
        sliders['time'] = Slider(ax_slider, 'time_index', 0, len(forcing.time())-1, valinit=0, valstep=1)
        sliders['time'].on_changed(update_plot)
    if len(boundary.x()) > 1:
        ax_slider2 = plt.axes([0.17, 0.01, 0.65, 0.03])
        sliders['inds'] = Slider(ax_slider2, 'inds_index', 0, len(boundary.x())-1, valinit=0, valstep=1)
        sliders['inds'].on_changed(update_plot)
    update_plot(0)
    # fig_dict['ax'].set_xlabel(forcing.x_str)
    # fig_dict['ax'].set_ylabel(forcing.y_str)
    # fig_dict['cbar'].set_label('Wind speed [m/s]')

    plt.show(block=True)

    return fig_dict

class Plotter:
    def __init__(self ,model):
        self.model = model
    def plot_grid(self, grid_plotter=dnora_grid_plotter) -> None:
        fig, ax = plt.subplots(1)
        fig_dict = {'fig': fig, 'ax': ax}
        fig_dict = grid_plotter(fig_dict, self.model)
        fig_dict.get('fig').show()

    def plot_forcing(self, forcing_plotter=dnora_forcing_plotter):
        
        fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = None
        gl.right_labels = None
        fig_dict = {'fig': fig, 'ax': ax, 'gl': gl}
        fig_dict = forcing_plotter(fig_dict, self.model)
        fig_dict.get('fig').show()

    def plot_boundary(self, boundary_plotter=dnora_boundary_plotter):
        fig, ax = plt.subplots(subplot_kw={'polar': True})
        fig_dict = {'fig': fig, 'ax': ax}
        fig_dict = boundary_plotter(fig_dict, self.model)
        fig_dict.get('fig').show()