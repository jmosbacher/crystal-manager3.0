import os
from traits.api import *
from traitsui.api import *
from traitsui.extras.checkbox_column import CheckboxColumn
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.colors import colorConverter
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from matplotlib import cm
import matplotlib
matplotlib.style.use('ggplot')

from matplotlib.ticker import LinearLocator, FormatStrFormatter, NullFormatter
from auxilary_functions import wl_to_rgb
import numpy as np
import random
import pandas as pd
from measurement import BaseMeasurement, SpectrumMeasurement, MeasurementTableEditor, ArrayViewer
from integration_results import IntegrationResultBase
from data_importing import SpectrumImportToolTab
from data_plot_viewers import SingleDataPlot
from plotting_tools import ExperimentPlottingTool
from integration_tool import ExperimentIntegrationTool
from fitting_tools import FittingTool1D, FittingTool2D
from auxilary_functions import merge_spectrums, pad_with_zeros, gaussian_integral
from scipy.interpolate import griddata
from saving import BaseSaveHandler
from pyface.api import FileDialog, confirm, error, YES, CANCEL
from analysis_tools import ExperimentAnalysisTool
from pandas.tools.plotting import scatter_matrix
from scipy.ndimage.filters import gaussian_filter
try:
    import cPickle as pickle
except:
    import pickle

class ExperimentBaseHandler(BaseSaveHandler):
    #extension = Str('int')
    promptOnExit = False

    def object_export_data_changed(self, info):
        fileDialog = FileDialog(action='save as', title='Save As',
                                wildcard=self.wildcard,
                                parent=info.ui.control,
                                default_filename=info.object.name)
        if info.object.export_format=='Clipboard':
            info.object.save_pandas()
        else:
            fileDialog.open()
            if fileDialog.path == '' or fileDialog.return_code == CANCEL:
                return False
            else:
                info.object.save_pandas(fileDialog.path)

class BaseExperiment(HasTraits):
    __kind__ = 'Base'
    main = Any()
    name = Str('Name')
    date = Date()
    crystal_name = Str('')

    measurements = List(BaseMeasurement)
    selected = Instance(BaseMeasurement)



class SpectrumExperiment(BaseExperiment):
    __kind__ = 'Spectrum'
    #####       Data      #####

    ex_wl_range = Property(Tuple)
    em_wl_range = Property(Tuple)
    measurement_cnt = Property(Int)

    int_results = List(IntegrationResultBase)
    fit_results = Array()
    meshgrid = Tuple(transient=True)
    has_mesh = Bool(False,transient=True)

    #####       UI      #####
    add_type = Enum(['Spectrum', 'Raman', 'Anealing'])
    add_meas = Button('Add Measurements')
    edit = Button('Open')
    remove = Button('Remove Selected')
    select_all = Button('Select All')
    unselect_all = Button('Un-select All')
    plot_selected = Button('Plot Selected')
    merge = Button('Merge')
    # import_exp = Button('Import Experiment')
    #comp_sel = Button('Compare selected')
    show_file_data = Button('File data')
    sort_by_wl = Button('Sort by WL')
    auto_merge = Button('Merge by WL')

    export_data = Button('Export')
    export_format = Enum('Clipboard',['CSV', 'Text', 'Excel','Latex','Clipboard'])
    export_binned = Bool(True)

    scale = Float(1)
    scale_what = Enum('Selected',['All','Selected'])
    rescale = Button('Rescale')

    #show_signal = Button('View Signal')
    #show_bg = Button('View BG')
    #show_binned = Button('View Binned')
    plot_3d_select = Enum('Mixed',['Mixed','Surface','Wires','Image', 'Polygons'])
    plot_3d = Button('Plot 3D')
    #####       Flags      #####
    is_selected = Bool(False)
    has_measurements = Property()

    #####       GUI View     #####
    import_tool = Instance(SpectrumImportToolTab,transient=True)
    plotting_tool = Instance(ExperimentPlottingTool,transient=True)
    analysis_tool = Instance(ExperimentAnalysisTool,transient=True)
    integration_tool = Instance(ExperimentIntegrationTool,transient=True)
    fitting_tool_1d = Instance(FittingTool1D,transient=True)
    fitting_tool_2d = Instance(FittingTool2D,transient=True)

    view = View(

        Tabbed(
            VGroup(
                HGroup(
                    Item(name='export_data', show_label=False),
                    Item(name='export_format', label='Format'),
                    Item(name='export_binned', label='Bin Data'),
                    show_border=True, label='Export Data'),

                HGroup(
                    Item(name='add_meas', show_label=False),
                    Item(name='merge', show_label=False),
                    Item(name='sort_by_wl', show_label=False),
                    Item(name='auto_merge', show_label=False, ),
                    ),


                HGroup(
                    Item(name='select_all', show_label=False),
                    Item(name='unselect_all', show_label=False),
                    Item(name='remove', show_label=False, enabled_when='selected'),
                    Item(name='scale', label='Scale'),
                    Item(name='scale_what', show_label=False),
                    Item(name='rescale', show_label=False),
                    ),

                    Item(name='measurements', show_label=False, springy=True,
                         editor=MeasurementTableEditor(selected='selected')),

                show_border=True, label='Data', scrollable=True,),

            VGroup(
                Item(name='plotting_tool',show_label=False,style='custom',springy=False),
                show_border=True,label='Visualization'),

            VGroup(
                Item(name='integration_tool', show_label=False, style='custom', springy=False),
                show_border=True, label='Integration'),
            VGroup(
                Item(name='fitting_tool_1d', show_label=False, style='custom', springy=False),
                show_border=True, label='1D Fitting'),
            VGroup(
                Item(name='fitting_tool_2d', show_label=False, style='custom', springy=False),
                show_border=True, label='2D Fitting'),

            VGroup(
                Item(name='analysis_tool', show_label=False, style='custom', springy=False),
            label='Statistical Analysis'),

            Group(
                Item(name='import_tool',show_label=False,style='custom'),
            label='Import Measurements'),

            ),





        title='Experiment Editor',
        #buttons=['OK'],
        handler=ExperimentBaseHandler(),
        kind='panel',
        scrollable=True,
        resizable=False,
        #height=800,
        #width=1000,

    )

    #####       Initialization Methods      #####
    def __init__(self, **kargs):
        HasTraits.__init__(self)
        self.main = kargs.get('main', None)

    def _selected_default(self):
        return SpectrumMeasurement(main=self.main)

    def _import_tool_default(self):
        return SpectrumImportToolTab(experiment=self)

    def _plotting_tool_default(self):
        return ExperimentPlottingTool(experiment=self)

    def _analysis_tool_default(self):
        return ExperimentAnalysisTool(experiment=self)

    def _fitting_tool_1d_default(self):
        return FittingTool1D(measurements=self.measurements,
                             name=self.name)
    def _fitting_tool_2d_default(self):
        return FittingTool2D(experiment=self,
                             name=self.name)

    def _integration_tool_default(self):
        return ExperimentIntegrationTool(experiment=self)

    #####       Private Methods      #####
    def _anytrait_changed(self):
        if self.main is None:
            return
        self.main.dirty = True

    def _get_ex_wl_range(self):
        wls = [10000, 0]
        for exp in self.measurements:
            if exp.__kind__ == 'Spectrum':
                wls[0] = round(min(exp.ex_wl, wls[0]))
                wls[1] = round(max(exp.ex_wl, wls[1]))
        return tuple(wls)

    def _get_em_wl_range(self):
        wls = [10000, 0]
        for meas in self.measurements:
            if meas.__kind__ == 'Spectrum':
                wls[0] = round(min(meas.em_wl[0], wls[0]))
                wls[1] = round(max(meas.em_wl[1], wls[1]))
        return tuple(wls)

    def _get_measurement_cnt(self):
        return len(self.measurements)

    def _get_has_measurements(self):
        if self.measurements is None:
            return False
        if len(self.measurements):
            return True
        else:
            return False



    def _sort_by_wl_fired(self):
        def wl(spectrum):
            return spectrum.ex_wl,spectrum.em_wl[0]
        self.measurements.sort(key=wl)

    def _add_meas_fired(self):
        self.import_data()

    def _rescale_fired(self):

        for meas in self.measurements:
            if self.scale_what == 'All':
                meas.rescale(self.scale)
            elif self.scale_what=='Selected':
                if meas.is_selected:
                    meas.rescale(self.scale)

    def _auto_merge_fired(self):
        def wl_key(spectrum):
            return spectrum.ex_wl,spectrum.em_wl[0]

        organized = {}
        for meas in self.measurements:
            wl = meas.ex_wl
            if wl in organized.keys():
                organized[wl].append(meas)
            else:
                organized[wl] = [meas]
        for wl, measurments in organized.items():
            if len(measurments)>1:
                self.merge_group(sorted(measurments,key=wl_key))


    def _remove_fired(self):
        self.measurements.remove(self.selected)

    def _select_all_fired(self):
        for exp in self.measurements:
            exp.is_selected = True

    def _unselect_all_fired(self):
        for exp in self.measurements:
            exp.is_selected = False

    def _merge_fired(self):
        def wl_key(spectrum):
            return spectrum.ex_wl,spectrum.em_wl[0]
        for_merge = []
        for meas in self.measurements:
            if meas.is_selected and meas.__kind__=='Spectrum':
                for_merge.append(meas)
        if len(for_merge):
            self.merge_group(sorted(for_merge,key=wl_key))


    #####      Public Methods      #####
    def merge_group(self,for_merge):
        main = for_merge[0]
        rest = for_merge[1:]
        for meas in rest:
            main = merge_spectrums(main, meas)
            self.measurements.remove(meas)
        main.is_selected = False

    def add_measurement(self):
        new = SpectrumMeasurement(main=self.main)
        self.measurements.append(new)
        #self.selected = new
        return new

    def save_to_file(self,path):
        #localdir = os.path.dirname(os.path.abspath(__file__))
        #path = os.path.join(localdir,'saved.spctrm')
        path = self.save_load_path
        with open(path,'wb') as f:
            pickle.dump(self.measurements, f)

    def load_from_file(self):
        #localdir = os.path.dirname(os.path.abspath(__file__))
        #path = os.path.join(localdir,'saved.spctrm')
        path = self.save_load_path
        with open(path, 'rb') as f:
            self.measurements = pickle.load(f)

    def make_dataframe(self,data_name='bg_corrected',**kwargs):
        data = {}
        for meas in self.measurements:
            data[meas.ex_wl] = meas.create_series(data_name=data_name,**kwargs)
        return pd.DataFrame(data)

    def make_db_dataframe(self):
        data = {}
        for meas in self.measurements:
            data[meas.ex_wl] = meas.make_db_dataframe()
        final = pd.concat(data)
        final.index.rename('ex_wl',level=0,inplace=True)
        return final

    def save_pandas(self,path=None, format=None):
        df = self.make_dataframe(bin_data=self.export_binned,round_wl=True)

        functions = {
                    'CSV':df.to_csv, 'Text':df.to_string, 'Excel':df.to_excel, 'Latex':df.to_latex,
        }
        if format is None:
            fmt = self.export_format
        else:
            fmt = format
        if fmt=='Clipboard' or (path is None):
            df.to_clipboard()
            return
        else:
            df.functions[fmt](path)
            return

    def plot_1d(self,**kwargs):
        legend = kwargs.get('legend',True)
        selected_only = kwargs.get('selected_only',False)
        data_names = kwargs.get('data_names',['signal', 'bg', 'ref'])
        figure = kwargs.get('figure',None)
        title = kwargs.get('title',' ')
        kind = kwargs.get('kind','Spectrum')
        args = kwargs.copy()
        if figure is None:
            fig = plt.figure()
        else:
            fig = figure
            fig.clf()
        for meas in self.measurements:
            if meas.__kind__ == kind:
                if selected_only and not meas.is_selected:
                    continue
                for n,data in enumerate(data_names):
                    args['ax'] = fig.add_subplot(len(data_names),1,n+1, axisbg='#F4EAEA')
                    args['data'] = data
                    args['legend'] = legend
                    args['title'] = data
                    meas.plot_data(**args)

        if figure is None:
            plt.title(title)
            plt.show()
        else:
            figure.suptitle(title)
            figure.canvas.draw()


    def plot_3d_polygons(self,**kwargs):
        """

        :return:
        """
        figure = kwargs.get('figure',None)
        axs = kwargs.get('axs',None)
        title = kwargs.get('title',' ')
        kind = kwargs.get('kind','Spectrum')
        alpha = kwargs.get('alpha', 0.5)

        if figure is None:
            fig = plt.figure()
        else:
            fig = figure
            fig.clf()
        if axs is None:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = axs

        def cc(arg):
            return colorConverter.to_rgba(arg, alpha=0.6)
        #col_options = [cc('r'), cc('g'), cc('b'), cc('y')]

        verts = []
        colors = []
        zs = []
        wl_range = [3000,0]
        cnt_range = [0,10]
        for data in self.measurements:
            if data.__kind__ == kind:
                sig = data.bin_data()
                #print sig
                if len(sig):
                    zs.append(data.ex_wl)
                    if min(sig[:,1])!=0:
                        sig[:,1] = sig[:,1] - min(sig[:,1])
                    sig[-1, 1] = sig[0, 1] = 0
                    verts.append(sig)
                    colors.append(data.color)
                wl_range = [min(wl_range[0],min(sig[:,0])),max(wl_range[1],max(sig[:,0]))]
                cnt_range = [min(cnt_range[0], min(sig[:, 1])), max(cnt_range[1], max(sig[:, 1]))]
        poly = PolyCollection(verts,closed=False, facecolors=colors) #

        poly.set_alpha(alpha)
        ax.add_collection3d(poly, zs=zs, zdir='y')
        ax.set_xlabel('Emission [nm]')
        ax.set_xlim3d(wl_range)
        ax.set_ylabel('Excitation [nm]')
        ax.set_ylim3d(min(zs)-10, max(zs)+10)
        ax.set_zlabel('Counts')
        ax.set_zlim3d(cnt_range)
        plt.title(title)
        plt.show()

    def collect_3d_coordinates(self,bin_data=False,data_name='bg_corrected',**kwargs):
        exem, zs = [], []

        for meas in self.measurements:
            ex_wl=meas.ex_wl
            if bin_data:
                em_spectrum = meas.bin_data(data_name)
            else:
                em_spectrum = getattr(meas,data_name)
            #em_spectrum=meas.bin_data()

            wls = np.empty(em_spectrum.shape)
            wls[:,0] = np.full(len(em_spectrum), ex_wl)
            wls[:,1] = em_spectrum[:,0]
            zs.append(em_spectrum[:,1])
            exem.append(wls)
        cnts = np.concatenate(zs,axis=0)
        exem = np.concatenate(exem, axis=0)
        return exem,cnts

    def collect_XYZ_arrays(self, bin_data=False,**kwargs):
        xs, ys, zs = [], [], []

        for meas in self.measurements:
            ex_wl = meas.ex_wl
            if bin_data:
                em_spectrum = meas.bin_data()
            else:
                em_spectrum = meas.bg_corrected
            xs.append( np.full(len(em_spectrum), ex_wl) )
            ys.append( em_spectrum[:, 0] )
            zs.append( em_spectrum[:, 1] )

        Z = np.concatenate(zs, axis=0)
        X, Y = np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)
        return X,Y,Z

    def make_meshgrid(self,step=1,interp_method='linear',**kwargs):
        if not self.has_mesh:
            exem, cnts = self.collect_3d_coordinates(**kwargs)
            ex_min, ex_max = exem[:,0].min(), exem[:,0].max()
            em_min, em_max = exem[:,1].min(), exem[:,1].max()

            grid_x, grid_y = np.mgrid[ex_min:ex_max:step, em_min:em_max:step]

            grid_z = np.clip(griddata(exem, cnts/step, (grid_x, grid_y), method=interp_method,fill_value=0.0),0,np.inf)
            self.meshgrid = grid_x, grid_y, grid_z
            self.has_mesh = True

        return self.meshgrid

    def plot_3d_wires(self, **kwargs):
        """

        :return:
        """
        figure = kwargs.get('figure',None)
        axs = kwargs.get('axs',None)
        title = kwargs.get('title',' ')
        #alpha = kwargs.get('alpha',0.5)
        rstride = kwargs.get('rstride',2)
        cstride = kwargs.get('cstride',2)

        if figure is None:
            fig = plt.figure()
        else:
            fig = figure
            fig.clf()
        if axs is None:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = axs

        X,Y,Z = self.make_meshgrid()

        ax.plot_wireframe(X,Y,Z, rstride=rstride, cstride=cstride)
        ax.yaxis._axinfo['label']['space_factor'] = 2.8
        ax.set_xlabel('Excitation Wavelength [nm]')
        ax.set_xlim(X.min() - 30, X.max() + 30)
        ax.set_ylabel('Emission Wavelength [nm]')
        ax.set_ylim(Y.min() - 30, Y.max() + 30)
        ax.set_zlabel('Counts')
        ax.set_zlim(Z.min(), Z.max() + 100)
        fig.suptitle(title)
        if figure is None:
            plt.show()
        else:
            fig.canvas.draw()

    def plot_3d_surf(self,**kwargs ):
        figure = kwargs.get('figure',None)
        axs = kwargs.get('axs',None)
        title = kwargs.get('title',' ')
        #alpha = kwargs.get('alpha',0.5)
        rstride = kwargs.get('rstride',2)
        cstride = kwargs.get('cstride',2)

        if figure is None:
            fig = plt.figure()
        else:
            fig = figure
            fig.clf()
        if axs is None:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = axs
        X, Y, Z = self.make_meshgrid(**kwargs)
        surf = ax.plot_surface(X, Y, Z, rstride=rstride, cstride=cstride, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

        span = Z.max()-Z.min()
        ax.set_zlim(Z.min(), Z.max()+span/10)
        ax.yaxis._axinfo['label']['space_factor'] = 2.8
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.set_xlabel('Excitation Wavelength [nm]')
        ax.set_xlim(X.min()-30, X.max()+30)
        ax.set_ylabel('Emission Wavelength [nm]')
        ax.set_ylim(Y.min()-30, Y.max()+30)
        ax.set_zlabel('Counts')
        fig.suptitle(title)
        #fig.colorbar(surf, shrink=0.5, aspect=5)

        if figure is None:
            plt.show()

        else:
            fig.canvas.draw()

    def plot_3d_mixed(self,**kwargs ):
        figure = kwargs.get('figure',None)
        axs = kwargs.get('axs',None)
        title = kwargs.get('title',' ')
        alpha = kwargs.get('alpha',0.5)
        rstride = kwargs.get('rstride',5)
        cstride = kwargs.get('cstride',5)

        if figure is None:
            fig = plt.figure()
        else:
            fig = figure
            fig.clf()
        if axs is None:
            ax = fig.add_subplot(111, projection='3d',axisbg='none')
        else:
            ax = axs

        X, Y, Z = self.make_meshgrid(**kwargs)
        ax.plot_surface(X, Y, Z, rstride=rstride, cstride=cstride, alpha=alpha)
        cset1 = ax.contourf(X, Y, Z, zdir='z', offset=Z.min()-Z.max()/2, cmap=cmx.coolwarm)
        cset2 = ax.contourf(X, Y, Z, zdir='x', offset=X.min()-30, cmap=cmx.coolwarm)
        cset3 = ax.contourf(X, Y, Z, zdir='y', offset=Y.max()+30, cmap=cmx.coolwarm)
        ax.yaxis._axinfo['label']['space_factor'] = 2.8
        ax.set_xlabel('Excitation Wavelength [nm]')
        ax.set_xlim(X.min()-30, X.max()+30)
        ax.set_ylabel('Emission Wavelength [nm]')
        ax.set_ylim(Y.min()-30, Y.max()+30)
        ax.set_zlabel('Counts')
        fig.suptitle(title)
        ax.set_zlim(Z.min()-Z.max()/2, Z.max()+1000)

        if figure is None:
            plt.show()
        else:
            fig.canvas.draw()

    def plot_2d_image(self, **kwargs):
        figure = kwargs.get('figure',None)
        axs = kwargs.get('axs',None)
        title = kwargs.get('title',' ')

        if figure is None:
            fig = plt.figure()
        else:
            fig = figure
            fig.clf()
        if axs is None:
            ax = fig.add_subplot(111)
        else:
            ax = axs
        X, Y, Z = self.make_meshgrid(**kwargs)
        im = ax.imshow(Z.T, extent=(X.min(), X.max(), Y.min(), Y.max()), origin='lower', cmap=cm.jet)
        fig.suptitle(title)
        #plt.imshow(Z, extent=(X.min(), X.max(), Y.min(), Y.max()), origin='lower')
        if figure is None:
            plt.show()
        else:
            fig.canvas.draw()

    def plot_2d_contour(self,**kwargs ):
        figure = kwargs.get('figure',None)
        axs = kwargs.get('axs',None)
        title = kwargs.get('title',' ')
        nlevel = kwargs.get('nlevel',10)
        level_range = kwargs.get('level_range',None)
        contf = kwargs.get('contf', True)
        colbar = kwargs.get('colbar', False)
        setlabels = kwargs.get('setlabels', True)
        setlimits = kwargs.get('setlimits', True)
        gsigma = kwargs.get('gsigma', 0.7)
        set_levels = kwargs.get('set_levels', 'linear')

        if figure is None:
            fig = plt.figure()
        else:
            fig = figure
        if axs is None:
            fig.clf()
            ax = fig.add_subplot(111)
        else:
            ax = axs
        X, Y, Z = self.make_meshgrid(**kwargs)
        if level_range is None:
            minlev,maxlev = Z.min(),Z.max()
        else:
            minlev, maxlev =level_range
        if set_levels is 'linear':
            levels = np.linspace(minlev,maxlev,nlevel)
            norm = None
        else:
            if minlev<0.001:
                minlev=0.001
            lev_exp = np.linspace(np.floor(np.log10(minlev)), np.ceil(np.log10(maxlev)),nlevel)
            levels = np.power(10, lev_exp)
            norm=colors.LogNorm()
        if contf:
            contfplot = ax.contourf(X, Y, Z, cmap=cm.jet, levels=levels,norm=norm)
            if colbar:
                fig.colorbar(contfplot, ax=ax, format="%.2e")
        Z = gaussian_filter(Z,sigma=gsigma)
        contplot = ax.contour(X, Y, Z,  levels=levels, colors='k', )
        if setlabels:
            ax.set_xlabel('Excitation Wavelength [nm]')
            ax.set_ylabel('Emission Wavelength [nm]')
        if setlimits:
            ax.set_xlim(X.min(), X.max())
            ax.set_ylim(Y.min(), Y.max())
        if title is not ' ':
            fig.suptitle(title)

        # plt.imshow(Z, extent=(X.min(), X.max(), Y.min(), Y.max()), origin='lower')
        if figure is None:
            plt.show()
        else:
            fig.canvas.draw()
        return ax

    def plot_2d_mixed(self,figure=None,colbar=False,rstride=20,cstride=20,**kwargs):
        #figure = kwargs.get('figure',None)
        if figure is None:
            fig = plt.figure(1, figsize=(9, 9))
        else:
            fig = figure
        nullfmt = NullFormatter()
        fig.clf()
        # definitions for the axes
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        bottom_h = left_h = left + width + 0.02

        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom_h, width, 0.2]
        rect_histy = [left_h, bottom, 0.2, height]

        # start with a rectangular Figure


        ax_main = fig.add_axes(rect_scatter)
        ax_x = fig.add_axes(rect_histx)
        ax_y = fig.add_axes(rect_histy)

        ax_x.xaxis.set_major_formatter(nullfmt)
        ax_x.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))

        ax_y.set_xticklabels(ax_y.xaxis.get_majorticklabels(), rotation=45)
        ax_y.yaxis.set_major_formatter(nullfmt)
        ax_y.xaxis.set_major_formatter(FormatStrFormatter('%.1e'))

        self.plot_2d_contour(figure=fig,axs=ax_main,colbar=False,**kwargs)
        X,Y,Z = self.make_meshgrid(**kwargs)
        for n in range(0,len(X[0]),rstride):
            ax_x.plot(X[:,n],Z[:,n], color=wl_to_rgb(min(Y[0,n],780))) #

        for n in range(0,len(Y),cstride):
            ax_y.plot(Z[n,:],Y[n,:], color=wl_to_rgb(X[n,0]) ) #

        ax_x.set_xlim(ax_main.get_xlim())
        ax_y.set_ylim(ax_main.get_ylim())
        if figure is None:
            plt.show()
        else:
            fig.canvas.draw()
        return

    def plot_scatter_matrix(self,**kwargs):
        figure = kwargs.get('figure',None)
        axs = kwargs.get('axs',None)
        title = kwargs.get('title',' ')
        if figure is None:
            fig = plt.figure()
        else:
            fig = figure

        if axs is None:
            fig.clf()
            ax = fig.add_subplot(111)
        else:
            ax = axs

        df = self.make_dataframe(**kwargs)
        scatter_matrix(df,ax=ax, diagonal='kde')

        if figure is None:
            plt.show()
        else:
            fig.canvas.draw()
        return ax

class ExperimentTableEditor(TableEditor):

    columns = [
                CheckboxColumn(name='is_selected', label='', width=0.08, horizontal_alignment='center', ),
                ObjectColumn(name = 'name',label = 'Name',width = 0.25,horizontal_alignment = 'left',editable=True),

                ObjectColumn(name='crystal_name', label='Crystal', width=0.25, horizontal_alignment='left', editable=True),

                ObjectColumn(name = 'ex_wl_range',label = 'Excitation WLs',horizontal_alignment = 'center',
                             width = 0.13,editable=False),

                ObjectColumn(name = 'em_wl_range',label = 'Emission WLs',width = 0.13,
                             horizontal_alignment = 'center',editable=False),
                #ObjectColumn(name = 'em_pol',label = 'Emission POL',width = 0.08,horizontal_alignment = 'center'),

                ObjectColumn(name='measurement_cnt', label='Datasets', width=0.08,
                             horizontal_alignment='center',editable=False),

                ObjectColumn(name='desc', label='Description', width=0.08,
                             horizontal_alignment='center', editable=False),
              ]

    auto_size = True
    sortable = False
    editable = True

class ExperimentNameTableEditor(TableEditor):

    columns = [
                CheckboxColumn(name='is_selected', label='', width=0.08, horizontal_alignment='center', ),
                ObjectColumn(name = 'name',label = 'Name',width = 0.25,horizontal_alignment = 'left',editable=True),
                ObjectColumn(name='crystal_name', label='Crystal', width=0.25, horizontal_alignment='left', editable=True),

              ]

    auto_size = True
    sortable = False
    editable = False