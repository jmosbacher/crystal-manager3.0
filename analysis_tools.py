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
from pandas.tools.plotting import scatter_matrix
import matplotlib
matplotlib.style.use('ggplot')
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from auxilary_functions import calc_df_statistics
from traitsui.ui_editors.array_view_editor import ArrayViewEditor
from data_plot_viewers import SingleDataPlot
from export_tools import DataFrameExportTool
from viewers import DFrameViewer
try:
    import cPickle as pickle
except:
    import pickle
from traits.has_dynamic_views \
    import DynamicView, HasDynamicViews, DynamicViewSubElement


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
    plot_data = Enum('bg_corrected',['bg_corrected','bg','ref','signal'])
    subplots = Bool(True)
    #plot_result = Button('Plot Result')
    #plot_data = Button('Plot Data')
    alpha = Float(1.0)
    clear_plot = Bool(True)
    clear_result = Button('Clear Result')

    ### Options ###
    calc_type = Enum(['Rolling Window', 'Global'])
    rolling_statistic = Enum(['max','median','skew','sum','kurt','mean','min','quantile','var','std'])
    global_statistic = Enum(['hist','pct_change','kde','diff'])
    hist_bins = Int(100)
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
                    Item(name='hist_bins', label='Bins',visible_when="(plot_what=='Result Table' and result_plot_type=='Hist')"
                                                                     " or (plot_what=='Statistic' and plot_type=='hist')"  ),
                    ),
                HGroup(
                    Item(name='plot', show_label=False, ),
                    Item(name='plot_data', label='Data', ),
                ),

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
                self.calc_result.plot.hist(ax=ax, subplots=self.subplots, stacked=False,
                                           alpha=self.alpha,bins=self.hist_bins)

        elif self.plot_what=='Statistic':
            if self.plot_type=='kde':
                self.measurement.plot_by_name(ax=ax, plot_name=self.plot_type,data_name=self.plot_data)


            elif self.plot_type=='hist':
                self.measurement.plot_by_name(ax=ax, plot_name=self.plot_type,
                                              data_name=self.plot_data,bins=self.hist_bins)

            elif self.plot_type in ['lag','autocorrelation']:
                self.measurement.plot_special(ax=ax, plot_name=self.plot_type,data_name=self.plot_data)

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
    result_array = Instance(DFrameViewer)

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
                        Item(name='plot_type', label='Statistic', visible_when="plot_what=='Statistic'"),
                    Item(name='result_plot_type', show_label=False, visible_when="plot_what=='Result Table'"),
                    ),
                    HGroup(


                        Item(name='subplots', label='Sub Plots', enabled_when="plot_what=='Result Table'"),
                        Item(name='hist_bins', label='Bins',
                             visible_when="(plot_what=='Result Table' and result_plot_type=='Hist')"
                                          " or (plot_what=='Statistic' and plot_type=='hist')"),
                        spring,
                        Item(name='alpha', label='Opacity', ),
                    ),
                    HGroup(Item(name='plot', show_label=False,springy=True ),
                           spring,
                           Item(name='clear_plot', label='Clear Previous', ),
                           Item(name='plot_data', label='Data',visible_when="plot_what=='Statistic'" ), ),


                    Item(name='display', show_label=False, style='custom', springy=False),
                ),
                # show_border=True, label='Analysis'
            ),

        )


    def __init__(self, experiment):
        super(ExperimentAnalysisTool, self).__init__()
        self.experiment = experiment

    def _result_array_default(self):
        return DFrameViewer()

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
                self.calc_result.plot.hist(ax=ax, subplots=self.subplots, stacked=False,
                                           alpha=self.alpha,bins=self.hist_bins)

        elif self.plot_what == 'Statistic':
            if self.plot_type == 'kde':
                for measurement in self.experiment.measurements:
                    measurement.plot_by_name(ax=ax, plot_name=self.plot_type, alpha=self.alpha,data_name=self.plot_data)
            elif self.plot_type == 'hist':
                for measurement in self.experiment.measurements:
                    measurement.plot_by_name(ax=ax, plot_name=self.plot_type, alpha=self.alpha,
                                             data_name=self.plot_data,bins=self.hist_bins)

            elif self.plot_type in ['lag', 'autocorrelation']:
                for measurement in self.experiment.measurements:
                    measurement.plot_special(ax=ax, plot_name=self.plot_type,data_name=self.plot_data, )

        elif self.plot_what == 'Scatter Matrix':
            self.experiment.plot_scatter_matrix(ax=ax,data_name=self.plot_data)

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
        self.result_array.df = results


class ProjectAnalysisTool(HasTraits):
    project = Any

    refresh = Button('Import All Data')
    has_df = Bool(False)
    groupby_list = List(['experiment','ex_wl','em_wl','signal','bg','ref'])
    groupby_choice = Str('ex_wl')
    groupby_obj = Any()
    groupby = Button('Group')

    apply_what = Enum(['filter', 'aggregate'], cols=1)
    #apply_by = Enum('selected', ['size', 'sum', 'std', 'mean', 'selected', 'describe'])
    #apply = Button('Apply')

    result_cols = List(['experiment', 'ex_wl', 'em_wl', 'signal', 'bg', 'ref'])

    filter_list = List([])
    filter_selected = List()
    filter_statistic = Enum('selected',['size','sum','std','mean','selected' ])
    filter_logic = Enum(['>','=>','<','=<','=='])
    filter_value = Float()
    filter_col = Enum(['ex_wl','em_wl','signal','bg','ref'])
    apply_filter = Button('Filter Groups')

    agg_statistic = Enum('describe', ['size', 'sum', 'std', 'mean', 'describe'])
    apply_agg = Button('Aggregate Data')

    plt_kind = Enum('line',['line','hist','box','kde','area','bar'])
    plt_matrix = Bool(False)
    plt_alpha = Float(0.5)
    plt_stacked = Bool(False)
    plt_subplots = Bool(False)
    plt_bins = Int(50)
    plot_result = Button('Plot')
    #agg_by = Enum(['size','sum','std','mean'])
    #aggregate = Button('Aggregate')

    export_table = Button('Export Table')

    data_df = Instance(DFrameViewer)

    view = View(HGroup(
                    VGroup(
                        VGroup(
                        HGroup(Item(name='refresh',show_label=False),
                            Item(name='groupby_choice',label='Group By',
                                    editor=EnumEditor(name='object.groupby_list')),
                               Item(name='groupby',show_label=False,enabled_when='not groupby_obj'),

                        show_border=True,label='Group Data'),


                        VGroup(
                            HGroup(
                            Item(name='apply_what',show_label=False,style='custom'),
                               Item(name='result_cols',label='Include',style='custom',
                                    editor=CheckListEditor(name='object.groupby_list',cols=2)),
                            ),
                            HGroup(
                                Item(name='agg_statistic', show_label=False, ),
                                show_border=True, label='Statistic', visible_when='apply_what == "aggregate"'),

                            HGroup(
                            Item(name='filter_statistic',show_label=False,),
                            Item(name='filter_col', label='of', enabled_when='filter_statistic != "selected"'),
                            Item(name='filter_logic', show_label=False,enabled_when='filter_statistic != "selected"'),
                            Item(name='filter_value', show_label=False,enabled_when='filter_statistic != "selected"'),
                                    show_border=True,label='Filter Logic',visible_when='apply_what == "filter"'),
                            HGroup(
                            Item(name='apply_filter', show_label=False,visible_when='apply_what == "filter"'),
                            Item(name='apply_agg', show_label=False,visible_when='apply_what == "aggregate"'),
                            ),
                            show_border=True, label='Apply',enabled_when='groupby_obj'),

                        ),

                        VGroup(
                            Item(name='filter_list', show_label=False,
                                 enabled_when='apply_what=="filter" and filter_statistic == "selected"',
                                 editor=ListStrEditor(selected='filter_selected',
                                                                          editable=False, multi_select=True),
                                 resizable=True),

                            show_border=True, label='Group List',enabled_when='groupby_obj'),
                    ),

                    VGroup(
                        HGroup(
                            Item(name='plt_kind', label='Plot'),
                            Item(name='plt_matrix', label='As matrix',),
                            Item(name='plt_bins', label='Bins', enabled_when='plt_kind == "hist"'),
                            show_left=False),

                            HGroup(
                            Item(name='plt_alpha', label='Opacity', enabled_when='plt_kind in ["area","hist"]'),
                            Item(name='plt_stacked', label='Stack', ),
                            Item(name='plt_subplots', label='Subplots', ),
                            show_left=False),


                        Item(name='plot_result', show_label=False, ),
                        Item(name='data_df', style='custom', show_label=False),
                        Item(name='export_table', show_label=False, ),
                    ),


                       ),
                resizable=True

    )

    def __init__(self, project):
        super(ProjectAnalysisTool, self).__init__()
        self.project = project
        #self.refresh_dataframe()

    def _data_df_default(self):
        df = pd.DataFrame(data=[[0.0]*6],columns=['experiment', 'ex_wl', 'em_wl', 'signal', 'bg', 'ref'])
        data_df = DFrameViewer()
        data_df.df = df
        return data_df

    def _refresh_fired(self):
        self.refresh_dataframe()

    def _export_table_fired(self):
        tool = DataFrameExportTool(self.data_df.df)
        tool.configure_traits()

    def _groupby_fired(self):
        self.groupby_obj = self.data_df.df.groupby(self.groupby_choice,as_index=False)
        self.refresh_filter_list()


    def _plot_result_fired(self):
        kwargs = {}
        if self.plt_kind=='hist':
            kwargs['bins'] = self.plt_bins
        if self.plt_matrix:
            kwargs['diagonal'] = self.plt_kind
            ax = scatter_matrix(self.data_df.df,**kwargs)
        else:
            kwargs['kind'] = self.plt_kind
            if self.plt_kind in ["area", "hist"]:
                kwargs['alpha'] = self.plt_alpha
                kwargs['stacked'] = self.plt_stacked
            kwargs['subplots'] = self.plt_subplots

            ax = self.data_df.df.plot(**kwargs)
        plt.show()

    def _apply_agg_fired(self):
        arg = self.agg_statistic
        cols = self.result_cols[:]
        df = getattr(self.groupby_obj, self.apply_what)(arg)
        if arg == 'size':
            df = df.rename('size')
            cols.append('size')
            #df = pd.DataFrame(df)
            df = df.reset_index()
        elif arg == 'describe':
            #df = df[[x for x in cols if x!=self.groupby_choice]]
            groupby_obj = self.data_df.df.groupby(self.groupby_choice)
            df = getattr(groupby_obj, self.apply_what)(arg)
            df.index.rename((self.groupby_choice, 'statistic'), inplace=True)
            df = df.reset_index()
            cols = [self.groupby_choice,'statistic']
            cols.extend([x for x in self.result_cols if x != self.groupby_choice])
            #df = df.rename(index=(self.groupby_choice, 'statistic'))

        #cols = [x for x in self.result_cols if x in df.columns.values]

        cols = [x for x in cols if x in list(df.columns.values.flatten())]

        #cols.extend( [x for x in self.result_cols if x in df.columns.values] )
        if len(cols):
            df = df[cols]
        self.data_df.df = df
        self.groupby_obj = None
        self.refresh_groupby_list()

    def _apply_filter_fired(self):
        arg = self.filter_statistic
        cols = self.result_cols[:]
        if arg == 'selected':
            dfs = {}
            for key in self.filter_selected:
                dfs[key] = self.groupby_obj.get_group(key)
            df = pd.concat(dfs)
            # df.index.rename(self.groupby_choice,level=0,inplace=True)
            # df = df.reset_index(level=0)
            # self.groupby_obj = None
            # cols = [x for x in self.result_cols if x in df.columns.values]
        else:
            # filter_group = self.groupby_obj[]
            statistic = self.filter_statistic
            if statistic != 'size':
                statistic += '()'
            arg = eval('lambda x: x["{}"].{} {} {}'.format(self.filter_col,
                                                             statistic, self.filter_logic,
                                                             self.filter_value))
            # arg = filter_group.filter(filter)
            df = getattr(self.groupby_obj, self.apply_what)(arg)
            # df.index.rename(self.groupby_choice, inplace=True)
            # cols = [x for x in self.result_cols if x in df.columns.values]

        cols = [x for x in cols if x in list(df.columns.values.flatten())]
        # cols.extend( [x for x in self.result_cols if x in df.columns.values] )
        if len(cols):
            df = df[cols]
        self.data_df.df = df
        self.groupby_obj = None
        self.refresh_groupby_list()


    def refresh_groupby_list(self):
        self.groupby_list = list(self.data_df.df.columns.values.flatten())

    def refresh_filter_list(self):
        self.filter_list = sorted(self.groupby_obj.groups.keys())

    def refresh_dataframe(self):
        df = self.project.make_db_dataframe().reset_index()
        df.columns = ['experiment', 'ex_wl', 'em_wl', 'signal', "bg", 'ref']
        self.data_df.df = df
        self.has_df = True
        self.refresh_groupby_list()


