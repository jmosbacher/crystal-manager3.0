import os
from traits.api import *
from traitsui.api import *
from traits.api import HasTraits
from traitsui.api import View, Item
from mpl_figure_editor import MPLFigureEditor, MPLInitHandler

import matplotlib.pyplot as plt
import matplotlib.cm as cmx
from matplotlib import cm
import matplotlib
matplotlib.style.use('ggplot')
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

from data_plot_viewers import SingleDataPlot
try:
    import cPickle as pickle
except:
    import pickle


class PlottingToolBase(HasTraits):
    display = Instance(SingleDataPlot)
    new_window = Bool(False)

    plot = Button('Plot')
    clear_plot = Button('Clear Display')
    bin = Bool(False)
    nbins = Int(0)
    round_wl = Bool(False)

    def mpl_setup(self):
        self.figure.patch.set_facecolor('none')

    def _display_default(self):
        return SingleDataPlot()


class MeasurementPlottingTool(PlottingToolBase):
    measurement = Any()

    plot_data_names = List(['bg_corrected'])  # values = [('bg_corrected','BG Corrected'),(,'Signal'), (,'Background'), (,'Reference')]

    view = View(

        VGroup(
            HGroup(

                Item(name='plot', show_label=False, ),
                Item(name='clear_plot', show_label=False, ),
                Item(name='new_window', label='Plot in new window'),
                Item(name='plot_data_names', label='Plot', enabled_when='plot_select=="1D Lines"', style='custom',
                         editor=CheckListEditor(cols=4, values=[('bg_corrected', 'BG Corrected'),
                                                                ('signal', 'Signal'), ('bg', 'Background'),
                                                                ('ref', 'Reference')])),

            ),
            HGroup(
                Item(name='bin', label='Bin Data'),
                Item(name='nbins', label='Bins',enabled_when='bin' ),
                Item(name='round_wl', label='Round WLs'),
            ),
            VGroup(
            Item(name='display', show_label=False, style='custom', springy=True),
            springy=True),
            show_border=True, label='Visualization',springy=True),

    )

    def __init__(self, measurement):
        super(MeasurementPlottingTool, self).__init__()
        self.measurement = measurement

    def _plot_fired(self):
        if self.new_window:
            figure = None
            ax = None
        else:
            figure = self.display.figure
        kwargs = dict(
            #figure=figure,
            bin=self.bin,
            nbins=self.nbins,
            round_wl=self.round_wl
        )
        for n,data_name in enumerate(self.plot_data_names):
            if figure is not None:
                ax = figure.add_subplot(len(self.plot_data_names),1,n+1)
            kwargs['title'] = data_name
            ax = self.measurement.plot_data(data_name=data_name,figure=figure,ax=ax,**kwargs)




class ExperimentPlottingTool(PlottingToolBase):
    experiment = Any()
    plot_data_names = List(['bg_corrected']) #values = [('bg_corrected','BG Corrected'),(,'Signal'), (,'Background'), (,'Reference')]
    plot_select = Enum('Mixed', ['Mixed', '3D Surface', '3D Wires',
                                 '3D Polygons', '2D Contours', '2D Image', '1D Lines'])

    nlevel = Int(10)
    set_levels = Bool(False)
    level_range = Tuple((0,1e7),cols=2)
    selected_only = Bool(False)

    view = View(

        VGroup(
            HGroup(
                Item(name='plot_select', label='Plot Type'),
                Item(name='plot', show_label=False, ),
                Item(name='clear_plot', show_label=False, ),

                Item(name='new_window', label='Plot in new window'),
                Item(name='selected_only', label='Plot Selected Only',enabled_when='plot_select=="1D Lines"'),
            ),


            HGroup(Item(name='nlevel', label='Levels'),
                    Item(name='set_levels', label='Set Levels',enabled_when="plot_select=='2D Contours'"),
                    Item(name='level_range', label='Set Levels', enabled_when="set_levels"),

                   ),
            HGroup(
                Item(name='bin', label='Bin Data'),
                Item(name='nbins', label='Bins', enabled_when='bin' ),
                Item(name='round_wl', label='Round WLs'),
                Item(name='plot_data_names', label='Plot', enabled_when='plot_select=="1D Lines"', style='custom',
                     editor=CheckListEditor(cols=4, values=[('bg_corrected', 'BG Corrected'),
                                                            ('signal', 'Signal'), ('bg', 'Background'),
                                                            ('ref', 'Reference')])),
            ),
            Item(name='display', show_label=False, style='custom', springy=True),
            show_border=True, label='Visualization'),

            )

    def __init__(self, experiment):
        super(ExperimentPlottingTool, self).__init__()
        self.experiment = experiment

    def _plot_fired(self):
        if self.new_window:
            figure = None
        else:
            figure = self.display.figure

        kwargs = dict(
            figure=figure,
            data_names=self.plot_data_names,
            nlevel=self.nlevel,
            selected_only=self.selected_only,
            bin = self.bin,
            nbins = self.nbins,
            round_wl = self.round_wl
            )

        if self.set_levels:
            kwargs['level_range'] = self.level_range

        {'1D Lines':self.experiment.plot_1d,
        'Mixed': self.experiment.plot_3d_mixed,
        '3D Surface': self.experiment.plot_3d_surf,
        '3D Wires': self.experiment.plot_3d_wires,
        '2D Contours': self.experiment.plot_2d_contour,
        '2D Image': self.experiment.plot_2d_image,
        '3D Polygons': self.experiment.plot_3d_polygons,
        }[self.plot_select](**kwargs)


