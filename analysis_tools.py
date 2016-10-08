import os
from traits.api import *
from traitsui.api import *
from traitsui.ui_editors.data_frame_editor import DataFrameEditor
from mpl_figure_editor import MPLFigureEditor, MPLInitHandler
from pandas import DataFrame,Series
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
from matplotlib import cm
import matplotlib
matplotlib.style.use('ggplot')
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from auxilary_functions import calc_df_statistics
from traitsui.ui_editors.array_view_editor import ArrayViewEditor
from data_plot_viewers import SingleDataPlot
from export_tools import DataFrameExportTool
try:
    import cPickle as pickle
except:
    import pickle
from traits.has_dynamic_views \
    import DynamicView, HasDynamicViews, DynamicViewSubElement


class DynamicArrayViewHandler(Handler):
    #array_titles = List(['BG corrected', 'Signal', 'Background', 'Reference'])
    def object_refresh_view_changed(self,info):
        if not info.initialized:
            return
        #if info.object.refresh_view:
            #info.object.refresh_view = False
            #info.object.edit_traits(parent=info.ui.parent)


class DynamicArrayView(HasDynamicViews):
    array = Array()
    titles = List(['BG corrected', 'Signal', 'Background', 'Reference'])


    ui_array = Group(Item(
            name='array', show_label=False, style='custom',
            editor=ArrayViewEditor(titles=[' ',],
                                   format='%g',
                                   show_index=False, ),

        ),
        _array_view_order = 5,
        _array_view_priority = 1,

        )




    def __init__(self, *args, **traits ):
        super(DynamicArrayView, self).__init__(*args, **traits)

        declaration = DynamicView(
            name='array_view',
            id='dynamic_array_view',
            keywords={

                'dock': 'tab',
                'resizable': True,
                'scrollable': True,
            },
            use_as_default=True,
        )
        self.declare_dynamic_view(declaration)


        #self.ui_array = DynamicViewSubElement()

    #def _titles_changed(self):
        #new_editor=ArrayViewEditor(titles=self.titles,
                               #format='%g',
                               #show_index=False, )

        #ui_array = self.trait_view(name='ui_array')
        #ui_array.editor = new_editor


    def _array_default(self):
        return np.asarray([[0.0,0.0,0.0,0.0]])


class AnalysisToolBase(HasTraits):
    display = Instance(SingleDataPlot)
    calc_result = Instance(DataFrame)
    result_array = Array()
    has_result = Bool(False)

    calc = Button('Calculate')
    calc_on_result = Bool(False)

    ### Export Results Table ###
    export_result = Button('Export Results')


    plot = Button('Plot')
    plot_what = Enum('Statistic',['bg_corrected','signal','bg','ref','Result Table','Statistic','Scatter Matrix'])
    subplots = Bool(True)
    #plot_result = Button('Plot Result')
    #plot_data = Button('Plot Data')
    alpha = Float(1.0)
    clear_plot = Bool(True)
    clear_result = Button('Clear Result')

    ### Options ###
    calc_type = Enum(['Rolling Window', 'Global'])
    rolling_statistic = Enum(['max','median','skew','sum','kurt','mean','mean','min','quantile','var','std'])
    global_statistic = Enum(['hist','pct_change','kde','diff'])
    plot_type = Enum('hist',['hist','kde','autocorrelation','lag'])
    result_plot_type = Enum(['Lines','Area','Hist'])
    #use_window = Bool(False)
    window_size = Int(100)
    bin = Bool(False)
    nbins = Int(0)
    round_wl = Bool(False)


    def mpl_setup(self):
        self.figure.patch.set_facecolor('none')

    def _display_default(self):
        return SingleDataPlot()

    def _result_array_default(self):
        return np.asarray([[0.0,0.0,0.0,0.0]])#self.measurement.create_dataframe()

    def _export_result_fired(self):
        tool = DataFrameExportTool(df=self.calc_result)
        tool.edit_traits()

class MeasurementAnalysisTool(AnalysisToolBase):
    measurement = Any()

    calc_data_names = List(['bg_corrected','bg','ref','signal'])  # values = [('bg_corrected','BG Corrected'),(,'Signal'), (,'Background'), (,'Reference')]

    view = View(

        HSplit(
            VGroup(
            HGroup(
                Item(name='calc_type', label='Statistic scope', ),
                Item(name='rolling_statistic', label='Statistic', visible_when="calc_type=='Rolling Window'"),
                Item(name='global_statistic', label='Statistic', visible_when="calc_type=='Global'" ),


                ),
            HGroup(
                Item(name='calc_data_names', label='Data to use', style='custom',
                         editor=CheckListEditor(cols=4, values=[('bg_corrected', 'BG Corrected'),
                                                                ('signal', 'Signal'), ('bg', 'Background'),
                                                                ('ref', 'Reference')])),
            ),


            HGroup(
                Item(name='window_size', label='Window', enabled_when="calc_type=='Rolling Window'"),
                Item(name='bin', label='Bin Data'),
                Item(name='nbins', label='Bins',enabled_when='bin'),
                Item(name='round_wl', label='Round WLs'),

                ),
            HGroup(Item(name='calc', show_label=False, ),
                   Item(name='calc_on_result', label='Perform on result'),
                   spring,
                   Item(name='export_result', show_label=False, enabled_when='has_result'),
                   ),
            Item(name='result_array', show_label=False, style='custom',
                 editor=ArrayViewEditor(titles=['BG corrected', 'Signal', 'Background', 'Reference'],
                                            format='%g',
                                            show_index=False,)),
            ),
            VGroup(
                HGroup(

                    Item(name='plot_what', label='Data to plot'),
                    Item(name='plot_type', label='Statistic',enabled_when="plot_what=='Statistic'"),
                    Item(name='result_plot_type', show_label=False, visible_when="plot_what=='Result Table'"),
                ),
                HGroup(

                    Item(name='clear_plot', label='Clear Previous', ),
                    Item(name='subplots', label='Sub Plots',enabled_when="plot_what=='Result Table'" ),
                    spring,
                    Item(name='alpha', label='Opacity', ),
                    ),
            Item(name='plot', show_label=False, ),
            Item(name='display', show_label=False, style='custom', springy=False),
                ),
            #show_border=True, label='Analysis'
        ),

    )

    def __init__(self, measurement):
        super(MeasurementAnalysisTool, self).__init__()
        self.measurement = measurement

    def _plot_fired(self):
        if self.clear_plot:
            self.display.clear_plots()
        if self.display.axs is None:
            ax = self.display.add_subplots(1)[0]
        elif len(self.display.axs):
            ax=self.display.axs[0]
        else:
            ax = self.display.add_subplots(1)[0]


        if self.plot_what in ['bg_corrected','signal','bg','ref',]:
            self.measurement.plot_data(ax=ax, data_name=self.plot_what)

        elif self.plot_what=='Result Table':
            if self.result_plot_type=='Lines':
                self.calc_result.plot(ax=ax,subplots=self.subplots)
            elif self.result_plot_type=='Area':
                self.calc_result.plot.area(ax=ax, subplots=self.subplots,stacked=False,alpha=self.alpha)
            elif self.result_plot_type=='Hist':
                self.calc_result.plot.hist(ax=ax, subplots=self.subplots, stacked=False, alpha=self.alpha)
        elif self.plot_what=='Statistic':
            if self.plot_type in ['kde','hist']:
                self.measurement.plot_by_name(ax=ax, plot_name=self.plot_type)
            elif self.plot_type in ['lag','autocorrelation']:
                self.measurement.plot_special(ax=ax, plot_name=self.plot_type)
        elif self.plot_what=='Scatter Matrix':
            self.measurement.plot_scatter_matrix(ax=ax, plot_name=self.plot_type)

        ax.relim()
        ax.autoscale_view()
        self.display.figure.canvas.draw()

    def _calc_fired(self):
        final = pd.DataFrame(columns=['bg_corrected', 'signal', 'bg', 'ref'])
        if self.calc_on_result:
            df = self.calc_result
        else:
            df = self.measurement.create_dataframe(data_names=self.calc_data_names,
                                               bin=self.bin,
                                               nbins=self.nbins,
                                               round_wl=self.round_wl)
        if self.calc_type=='Rolling Window':
            r = df.rolling(window=self.window_size)

            results =  getattr(r,self.rolling_statistic)()

        elif self.calc_type=='Global':
            results = calc_df_statistics(df, statistic=self.global_statistic,)

        for col in results:
            final[col] = results[col]
        final.dropna(how='all', inplace=True)
        self.calc_result = final
        self.result_array = final.as_matrix()
        if len(self.result_array)>1:
            self.has_result = True





class ExperimentAnalysisTool(AnalysisToolBase):
    experiment = Any()
    #array_titles = List(['BG corrected', 'Signal', 'Background', 'Reference'])
    calc_data_name = Str('bg_corrected')  # values = [('bg_corrected','BG Corrected'),(,'Signal'), (,'Background'), (,'Reference')]
    result_array = Instance(HasTraits,())

    view = View(

        HSplit(
            VGroup(
                HGroup(
                    Item(name='calc_type', label='Statistic scope', ),
                    Item(name='rolling_statistic', label='Statistic', visible_when="calc_type=='Rolling Window'"),
                    Item(name='global_statistic', label='Statistic', visible_when="calc_type=='Global'"),

                ),
                HGroup(
                    Item(name='calc_data_name', label='Data to use', #style='custom',
                         editor=CheckListEditor(cols=4, values=[('bg_corrected', 'BG Corrected'),
                                                                ('signal', 'Signal'), ('bg', 'Background'),
                                                                ('ref', 'Reference')])),
                    Item(name='bin', label='Bin Data'),
                    Item(name='nbins', label='Bins', enabled_when='bin'),
                ),

                HGroup(
                    Item(name='window_size', label='Window', enabled_when="calc_type=='Rolling Window'"),

                    Item(name='round_wl', label='Round WLs'),

                ),
                HGroup(
                Item(name='calc', show_label=False, ),
                Item(name='calc_on_result', label='Perform on result'),
                spring,
                Item(name='export_result', show_label=False, enabled_when='has_result'),
                ),
                Item(name='result_array', show_label=False, style='custom',),
                     #editor=ArrayViewEditor(titles=['BG corrected', 'Signal', 'Background', 'Reference'],
                                            #format='%g',
                                            #show_index=False, )),
            ),
            VGroup(
                HGroup(

                        Item(name='plot_what', label='Data to plot'),
                        Item(name='plot_type', label='Statistic', enabled_when="plot_what=='Statistic'"),
                    Item(name='result_plot_type', show_label=False, visible_when="plot_what=='Result Table'"),
                    ),
                    HGroup(

                        Item(name='clear_plot', label='Clear Previous', ),
                        Item(name='subplots', label='Sub Plots', enabled_when="plot_what=='Result Table'"),
                        spring,
                        Item(name='alpha', label='Opacity', ),
                    ),
                    HGroup(Item(name='plot', show_label=False,springy=True ),),
                    Item(name='display', show_label=False, style='custom', springy=False),
                ),
                # show_border=True, label='Analysis'
            ),

        )


    def __init__(self, experiment):
        super(ExperimentAnalysisTool, self).__init__()
        self.experiment = experiment

    def _result_array_default(self):
        return DynamicArrayView()

    def _plot_fired(self):
        if self.clear_plot:
            self.display.remove_subplots()
        if self.display.axs is None:
            ax = self.display.add_subplots(1)[0]
        elif len(self.display.axs):
            ax = self.display.axs[0]
        else:
            ax = self.display.add_subplots(1)[0]

        if self.plot_what in ['bg_corrected', 'signal', 'bg', 'ref', ]:
            for measurement in self.experiment.measurements:
                measurement.plot_data(ax=ax, data_name=self.plot_what)

        elif self.plot_what == 'Result Table':
            if self.result_plot_type == 'Lines':
                self.calc_result.plot(ax=ax, subplots=self.subplots)
            elif self.result_plot_type == 'Area':
                self.calc_result.plot.area(ax=ax, subplots=self.subplots, stacked=False, alpha=self.alpha)
            elif self.result_plot_type == 'Hist':
                self.calc_result.plot.hist(ax=ax, subplots=self.subplots, stacked=False, alpha=self.alpha)

        elif self.plot_what == 'Statistic':
            if self.plot_type in ['kde', 'hist']:
                for measurement in self.experiment.measurements:
                    measurement.plot_by_name(ax=ax, plot_name=self.plot_type, alpha=self.alpha)

            elif self.plot_type in ['lag', 'autocorrelation']:
                for measurement in self.experiment.measurements:
                    measurement.plot_special(ax=ax, plot_name=self.plot_type)

        elif self.plot_what == 'Scatter Matrix':
            self.measurement.plot_scatter_matrix(ax=ax, plot_name=self.plot_type)

        ax.relim()
        ax.autoscale_view()
        self.display.figure.canvas.draw()


    def _calc_fired(self):
        def wl(spectrum):
            return spectrum.ex_wl, spectrum.em_wl[0]

        if self.calc_on_result:
            df = self.calc_result
        else:
            all_series = []
            measurements = sorted(self.experiment.measurements,key=wl)
            for measurement in measurements:

                ser = measurement.create_series(data_name=self.calc_data_name,
                                                       bin=self.bin,
                                                       nbins=self.nbins,
                                                       round_wl=self.round_wl)
                ser.reset_index(drop=True,inplace=True)
                all_series.append(ser)
            df = pd.concat(all_series,axis=1)
        if self.calc_type == 'Rolling Window':
            r = df.rolling(window=self.window_size)

            results = getattr(r, self.rolling_statistic)()

        elif self.calc_type == 'Global':
            results = calc_df_statistics(df, statistic=self.global_statistic, )
        else:
            return
        results.dropna(how='all',inplace=True)
        self.calc_result = results
        ArrawViewClass = type('ArrawViewClass',(HasTraits,),
                              {
                               'array': Array(),
                                'view': View(
                                    HGroup(Item(name='array',show_label=False, style='custom',springy=True,
                                        editor=ArrayViewEditor(titles=[str(x) for x in results.columns.values],
                                            format='%g',
                                            show_index=False,)),
                                           scrollable=True),
                                    scrollable=True,
                                resizable=True)
                              })
        new = ArrawViewClass()
        new.array = results.as_matrix()
        self.result_array = new
        if len(new.array)>1:
            self.has_result = True
        #array_view = DynamicArrayView()
        #column_df = pd.DataFrame(columns=results.columns)
        #column_df.loc[0] = results.columns.values
        #array_view.array = pd.concat([column_df,results]).as_matrix()
        #self.result_array = array_view

        #self.new_array_view = self.trait_view()

        #self.result_array.edit_traits(parent=view.parent)

