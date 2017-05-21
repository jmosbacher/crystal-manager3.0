import os
from traits.api import *
from traitsui.api import *
from traitsui.extras.checkbox_column import CheckboxColumn
from traitsui.ui_editors.array_view_editor import ArrayViewEditor
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.colors import colorConverter
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from auxilary_functions import wl_to_rgb, bin_data_array, integrate_gaussian, gauss
from file_selector import string_list_editor
import numpy as np
import random
import pandas as pd
from pandas.tools.plotting import lag_plot, autocorrelation_plot
from data_plot_viewers import SingleDataPlot
from fitting_tools import FittingTool1D
from plotting_tools import MeasurementPlottingTool
from analysis_tools import MeasurementAnalysisTool
import scipy.stats as stats
from pandas.tools.plotting import scatter_matrix
try:
    import cPickle as pickle
except:
    import pickle


class ArrayViewer(HasTraits):

    data = Array

    view = View(
        Item('data',
              show_label = False,
              editor     = ArrayViewEditor(titles = [ 'Wavelength', 'Counts' ],
                                           format = '%.2f',
                                           show_index= False,
                                           # Font fails with wx in OSX;
                                           #   see traitsui issue #13:
                                           # font   = 'Arial 8'
                                          )
        ),
        title     = 'Array Viewer',
        width     = 0.3,
        height    = 0.8,
        resizable = True
    )


class FileDataViewer(HasTraits):
    data = Dict({'sig':[], 'bgd':[], 'ref':[]})
    sig_data = List()
    bgd_data = List()
    ref_data = List()

    view = View(
        VGroup(
            Group(Item(name='sig_data',editor=string_list_editor, show_label=False),show_border=True,label='Signal File'),

            Group(Item(name='bgd_data',editor=string_list_editor, show_label=False ),show_border=True,label='Background File'),
            Group(Item(name='ref_data',editor=string_list_editor, show_label=False  ),show_border=True,label='Reference File'),


        ),
        title = 'Supplemental file data',
        scrollable = True,
        resizable = True,

    )


class BaseMeasurement(HasTraits):
    __kind__ = 'Base'
    main = Any()
    name = Str('Name')
    date = Date()
    time = Time()
    summary = Property(Str)

    notes = Str('')
    notebook = Int(1)
    page = Int()

    is_selected = Bool(False)

    def __init__(self, **kargs):
        HasTraits.__init__(self)
        self.main = kargs.get('main', None)

    def _anytrait_changed(self,name):
        if self.main is None:
            return
        if name in ['date', 'name', 'time', 'summary',
                        'notes', 'notebook', 'page',
                    'duration','ex_pol','em_pol','ex_wl',
                    'em_wl','exposure','frames','e_per_count',
                    'signal','bg','ref','file_data']:

            self.main.dirty = True

    def _get_summary(self):
        raise NotImplemented


class SpectrumMeasurement(BaseMeasurement):
    __klass__ = 'Spectrum'

    #####       User Input      #####
    duration = Float(0)
    ex_pol = Int()  # Excitation Polarization
    em_pol = Int()  # Emission Polarization

    ex_wl = Float()  # Excitation Wavelength
    em_wl = Tuple((0.0, 0.0), labels=['Min', 'Max'])  # Emission Wavelength

    exposure = Float(1)
    frames = Int(1)
    e_per_count = Int(1)  # electrons per ADC count

    #####       Extracted Data      #####
    signal = Array()
    bg = Array()
    ref = Array()
    bg_corrected = Property(Array)
    file_data = Dict()

    simulation_data = Dict()
    data = Instance(pd.DataFrame)
    metadata = Dict()
    #####       Flags      #####
    is_simulated = Bool(False)
    has_signal = Property(Bool) #Bool(False)
    has_bg = Property(Bool) #Bool(False)
    has_bg_corrected = Property(Bool)
    has_ref = Property(Bool) #Bool(False)
    has_fits = Property(Bool)
    color = Property() #Tuple(0.0, 0.0, 0.0)  # Enum(['r', 'g', 'b', 'y', 'g', 'k','m','c','k'])

    #####       Calculated Data      #####
    fits = List([])
    fit_data = Property(Array)
    resolution = Property()
    all_data_array = Array(transient=True)
    show_data = Button('Show Data')

    #####       UI      #####



    #####       GUI layout      #####
    plotting_tool = Instance(MeasurementPlottingTool,transient=True)
    analysis_tool = Instance(MeasurementAnalysisTool,transient=True)
    fitting_tool = Instance(FittingTool1D,transient=True)

    view = View(
        Tabbed(
        HGroup(
            VGroup(

            VGroup(
                #Item(name='ex_pol', label='Excitation POL'),
                Item(name='ex_wl', label='Excitation WL'),

                Item(name='frames', label='Frames'),
                Item(name='exposure', label='Exposure'),
                Item(name='color', label='Plot Color'),

                show_border=True, label='Excitation'),
            VGroup(
                #Item(name='em_pol', label='Emission POL'),
                Item(name='em_wl', label='Emission WL'),
                Item(name='e_per_count', label='e/count'),


                show_border=True, label='Emission'),
            HGroup(Item(name='show_data', show_label=False, springy=True)),
            VGroup(
                    Item(name='file_data', editor=ValueEditor(), show_label=False),
                    label='File Metadata'),
            ),

            HGroup(
                Item(name='all_data_array',show_label=False,
                     editor=ArrayViewEditor(titles=['Wavelength','BG corrected', 'Signal', 'Background', 'Reference'],
                                            format='%g',
                                            show_index=False),
                    springy=True),


                #scrollable=True
                springy=True),
        label='Data'),


            VGroup(
                Item(name='plotting_tool', show_label=False, style='custom', springy=False),

            label='Visualization'),

            VGroup(
                Item(name='fitting_tool', show_label=False, style='custom', springy=False),
                show_border=True, label='Fitting'),

            VGroup(
                Item(name='analysis_tool', show_label=False, style='custom', springy=False),
            label='Statistical Analysis'),



        ),
    )

    #####       Initialzation Methods      #####

    def make_data_arrays(self):
        self.signal = self.data[['em_wl','sig']].as_matrix()
        self.bg = self.data[['em_wl','bgd']].as_matrix()
        self.ref = self.data[['em_wl','ref']].as_matrix()


    def _signal_default(self):
        return np.array([])

    def _bg_default(self):
        return np.array([[],[]])

    def _all_data_array_default(self):
        return np.array([[0.0,0.0,0.0,0.0,0.0]])

    def _plotting_tool_default(self):
        return MeasurementPlottingTool(measurement=self)

    def _analysis_tool_default(self):
        return MeasurementAnalysisTool(measurement=self)

    def _fitting_tool_default(self):
        return FittingTool1D(measurements=[self],
                             name=self.name)

    #####       getters      #####
    def _get_summary(self):
        report = 'Excitation: %d nm'%self.ex_wl + ' | Emission Range: %d:%d nm'%self.em_wl
        return report

    def _get_has_signal(self):
        if len(self.signal):
            return True
        else:
            return False

    def _get_resolution(self):
        if len(self.signal):
            return np.mean(np.diff(self.signal[:,0]))
        else:
            return 0.0075

    def _get_color(self):
        return wl_to_rgb(self.ex_wl)

    def _get_has_bg(self):
        if len(self.bg):
            return True
        else:
            return False

    def _get_has_bg_corrected(self):
        if len(self.bg) and len(self.signal):
            return True
        else:
            return False
    def _get_has_ref(self):
        if len(self.ref):
            return True
        else:
            return False

    def _get_has_fits(self):
        if len(self.fits):
            return True
        else:
            return False
    def _get_fit_data(self):
        data = np.zeros_like(self.signal)
        data[:, 0] = self.signal[:, 0]
        for a, m, s in self.fits:
            data[:, 1] += gauss(data[:, 0], a, m, s)

    def _show_data_fired(self):
        df = self.create_dataframe().reset_index()
        self.all_data_array = df.as_matrix(columns=['index','bg_corrected', 'signal', 'bg', 'ref'])


    def _signal_changed(self):
        if self.signal.size > 2:
            self.em_wl = (np.round(np.min(self.signal[:, 0])), np.round(np.max(self.signal[:, 0])))


    #####       Public Methods      #####
    def rescale(self, scale):
        if self.has_signal:
            self.signal[:, 1] *= scale
        if self.has_bg:
            self.bg[:, 1] *= scale
        if self.has_ref:
            self.ref[:, 1] *= scale

    def create_series(self,normalize=True,**kwargs):
        """

        :return:
        """
        bin = kwargs.get('bin',False)
        nbins = kwargs.get('nbins',None)
        round_wl = kwargs.get('round_wl',False)
        data_name = kwargs.get('data_name','bg_corrected')
        normed = np.zeros((1, 2))
        if not getattr(self,'has_'+data_name):
            return normed

        if normalize:
            normed = self.normalized(data_name)
        else:
            normed = getattr(self,data_name)

        if nbins is not None:
            pass
            if nbins:
                bins=nbins
            else:
                bins = round(normed[:, 0].max()) - round(normed[:, 0].min())
            normed = bin_data_array(normed,nbins=bins)

        if round_wl:
            indx = np.around(normed[:, 0],decimals=1)
        else:
            indx = normed[:, 0]

        return pd.Series(data=normed[:, 1], index=indx, name=self.ex_wl)

    def create_dataframe(self,**kwargs):
        data_names = kwargs.get('data_names', ['bg_corrected', 'signal', 'bg', 'ref'])
        data_dict = {}
        for data_name in data_names:
            if getattr(self,'has_'+data_name):
                data_dict[data_name] = self.create_series(data_name=data_name,**kwargs)
        return pd.DataFrame(data_dict)

    def make_db_dataframe(self,data_names=('signal', 'bg', 'ref')):

        final = None
        for data_name in data_names:
            if getattr(self, 'has_' + data_name):
                new = pd.DataFrame(data=self.normalized(data_name),columns=['em_wl',data_name])
                if final is not None:
                    final = pd.merge_asof(final,new,on='em_wl')
                else:
                    final = new
        return final.set_index('em_wl')

    def normalized(self,data_name):
        normed = np.copy(getattr(self,data_name))
        normed[:,1] = normed[:,1]/(self.exposure*self.frames)
        return normed



    def _get_bg_corrected(self):
        sig = np.copy(self.signal)
        sig[:,1] -= np.resize(self.bg[:,1],sig[:,1].size)
        return sig


    def bin_data(self,data_name='bg_corrected',**kwargs):
        """
        :return:
        """
        normalize = kwargs.get('normalize',True)
        nbins = kwargs.get('nbins',0)

        normed = np.zeros((1, 2))
        if not self.has_signal:
            return normed

        if normalize:
            normed = self.normalized(data_name)
        else:
            normed = getattr(self,data_name)

        if nbins:
            bins = nbins
        else:
            bins=round(normed[:,0].max())-round(normed[:,0].min())
        binned = bin_data_array(normed,nbins=bins)

        return binned

    def integrate_bg_corrected(self,l,r,fit=False):
        '''

        :param l: integration minimum (inclusive)
        :param r: integration maximum (inclusive)
        :return: background corrected integration result
        '''
        if not self.has_signal:
            return 0.0
        sig = 0.0
        signal = self.norm_signal()
        bgnd = self.norm_bg()
        if fit:
            sig = self.integrate_with_fit(signal,l,r)
        else:
            sig = np.sum(np.where(np.logical_and(signal[:,0]<=r,signal[:,0]>=l),signal[:,1],0.0))
        bg = np.sum(np.where(np.logical_and(bgnd[:, 0] <= r, bgnd[:, 0] >= l), bgnd[:, 1], 0.0))
        return sig-bg

    def integrate_data(self,l,r,data_name='bg_corrected'):
        '''
        :param data_name: data to integrate
        :param l: integration minimum (inclusive)
        :param r: integration maximum (inclusive)
        :return: background corrected integration result
        '''
        if not self.has_signal:
            return 0.0
        data = getattr(self,data_name)
        result = np.sum(np.where(np.logical_and(data[:, 0] <= r, data[:, 0] >= l), data[:, 1], 0.0))
        return result


    def plot_data(self,**kwargs ):
        ax = kwargs.get('ax',None)
        legend = kwargs.get('legend',True)
        data_name = kwargs.get('data_name','bg_corrected')
        title = kwargs.get('title',None)

        if self.has_signal:
            ser = self.create_series(**kwargs)
            axs = ser.plot(color=self.color, legend=legend, ax=ax)
            if ax is not None:
                if title is None:
                    ax.set_title(data_name,fontsize=12)
                ax.set_xlabel('Emission Wavelength')
                ax.set_ylabel('Counts')
                #plt.show()
            else:
                plt.show()
            return axs

    def plot_by_name(self,plot_name ='hist',title=None,data_name='bg_corrected', **kwargs):
        ax = kwargs.get('ax', None)
        #legend = kwargs.get('legend', True)
        #data = kwargs.get('data', 'bg_corrected')
        #title = kwargs.get('title', None)
        #alpha = kwargs.get('alpha', 1.0)
        if self.has_signal:
            ser = self.create_series(data_name=data_name,**kwargs)
            axs = getattr( ser.plot,plot_name)(color=self.color, **kwargs)
            if ax is not None:
                if title is None:
                    ax.set_title(data_name+' '+plot_name, fontsize=12)

            else:
                plt.show()
            return axs

    def plot_special(self,plot_name='autocorrelation',title=None,data_name ='bg_corrected', **kwargs):
        if not self.has_signal:
            return
        fig = kwargs.get('figure', plt.figure())
        ax = kwargs.get('ax', fig.add_subplot(111))
        nbins = kwargs.get('nbins',150)

        ser = self.create_series(**kwargs)
        #data_name = kwargs.get('data_name', 'BG corrected')
        axs = {
            'lag':lag_plot,
            'autocorrelation':autocorrelation_plot,
            }[plot_name](ser,**kwargs)
        if title is None:
            axs.set_title(' '.join([data_name,plot_name]) , fontsize=12)
        if ax is None:
            plt.show()

    def plot_scatter_matrix(self,**kwargs):
        diag = kwargs.get('diag','kde')
        if not self.has_signal:
            return
        fig = kwargs.get('figure', plt.figure())
        ax = kwargs.get('ax', fig.add_subplot(111))
        df = self.create_dataframe(**kwargs)
        scatter_matrix(df,ax=ax, diagonal=diag)

    def calc_statistics(self,statistic='hist',**kwargs):
        ser = self.create_series(**kwargs)
        if statistic == 'hist':
            bins = kwargs.get('bins', 150)
            cnts, divs = np.histogram(ser,bins=bins)
            return pd.Series(data=cnts, index=divs[:-1]+np.diff(divs)/2, name=self.ex_wl)

        elif statistic == 'kde':
            nsample = kwargs.get('nsample', 200)
            arr = ser.as_matrix()
            kern = stats.gaussian_kde(arr)
            rng = np.linspace(arr.min(),arr.max(),nsample)
            return pd.Series(data=kern(rng), index=rng, name=self.ex_wl)


class AnealingMeasurement(BaseMeasurement):
    __kind__ = 'Anealing'
    temperature = Int(0)
    heating_time = Int(0)
    view = View(
        VGroup(

            HGroup(
                Item(name='temperature', label='Temperature'),
                Item(name='heating_time', label='Heating time'),
                show_border=True, label='Anealing Details'),

        ),

)

class MeasurementTableEditor(TableEditor):

    columns = [
               CheckboxColumn(name='is_selected', label='', width=0.08, horizontal_alignment='center',editable=True ),
               ObjectColumn(name='name', label='Name', horizontal_alignment='left', width=0.3,editable=True ),
               ObjectColumn(name='summary', label='Details', width=0.55, horizontal_alignment='center',editable=False  ),
               ObjectColumn(name='date', label='Date', horizontal_alignment='left', width=0.18,editable=False),
               ObjectColumn(name='__kind__', label='Type', width=0.18, horizontal_alignment='center',editable=False),


               ]

    auto_size = False
    sortable = False
    editable = True
    #scrollable=False
