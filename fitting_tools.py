from traits.api import *
from traitsui.api import *
from traitsui.ui_editors.array_view_editor import ArrayViewEditor
from data_plot_viewers import DataPlotEditorBase
import matplotlib.pyplot as plt
from data_plot_viewers import FittingDataPlot1D, FittingDataPlot2D
import numpy as np
from scipy.optimize import curve_fit
from fitters import SpectrumFitterBase, SpectrumFitter1D, SpectrumFitter2D
from auxilary_functions import gaussian_integral, pad_with_zeros


class FittingToolBase(HasTraits):
    name = Str(' ')
    fits_list = Array()  # DelegatesTo('selected')
    fitter = Instance(SpectrumFitterBase,transient=True)
    fit_result = Any()
    # fitter = Instance(SpectrumFitter)

    # use_fit = Bool(False)

    #####       Plots     #####
    display = Instance(DataPlotEditorBase,transient=True)

    #####       GUI     #####
    perform_fit = Button('Perfom Fit')
    integrate = Button('Integrate Selections')
    clear = Button('Clear Selections')
    clear_fits = Button('Clear Fits')
    refresh = Button('Refresh View')
    save = Button('Save Results')
    editing = DelegatesTo('display')
    message = Str(' ',transient=True)
    has_peaks = Property(Bool)
    has_frange = Property(Bool)


    def _get_has_peaks(self):
        if self.display is None:
            return False

        else:
            return self.display.has_peaks


    def _get_has_frange(self):
        if self.display is None:
            return False
        else:
            return self.display.has_frange


class FittingTool1D(FittingToolBase):
    measurements = List([])
    view = View(
        VGroup(
            VGroup(

                HGroup(
                    Item(name='editing', style='custom', label='Edit', ),
                    Item(name='clear', show_label=False, ),
                    Item(name='clear_fits', show_label=False, ),
                    Item(name='refresh', show_label=False),
                    Item(name='message', style='readonly', show_label=False, springy=True),
                    spring,
                    Item(name='perform_fit', show_label=False, ),

                    show_border=True, label='Region Selection'
                ),

                Group(Item(name='display', style='custom', show_label=False),
                      show_border=True, label='Plots'),

            ),

            VGroup(
                Group(Item(name='fits_list', show_label=False, editor=ArrayViewEditor(
                    titles=['Wavelength', 'Amplitude', 'Mean',
                            'Sigma', 'Integral', ],
                    format='%g',

                    show_index=False, )),
                      show_border=True, label='Fits', scrollable=True),

            ),

        ),
        buttons=['OK'],
        title='Fitting Tool',
        kind='live',

        scrollable=True,
        resizable=True,
        height=800,
        width=1200,
    )

    def __init__(self,measurements,**kwargs):
        super(FittingTool1D, self).__init__()
        self.name = kwargs.get('name','Fitting Plots')
        self.measurements = measurements
        #self.refresh_display()

    def _display_default(self):
        return FittingDataPlot1D()

    def _fits_list_default(self):
        return np.asarray([[0.0,0.0,0.0,0.0, 0.0]])

    def _fitter_default(self):
        return SpectrumFitter1D()

    def _perform_fit_fired(self):
        if self.has_peaks and self.has_frange:
            self.message = ' '
        else:
            if self.has_peaks:
                self.message='Please select a Fit Range'
            elif self.has_frange:
                self.message = 'Please select Peaks'
            else:
                self.message = 'Please select Peaks and a Fit Range '
            return
        peaks = []
        for xmin,xmax in self.display.peaks:
            if xmax-xmin>1:
                peaks.append((xmin,xmax))
        if len(peaks)==0:
            self.message = 'Please select Peaks'
            return
        fitter = SpectrumFitter1D(peaks=peaks)
        frange = self.display.frange
        for meas in self.measurements:
            data = meas.bg_corrected
            data = data[np.where(np.logical_and(data[:,0]>=frange[0],data[:,0]<=frange[1]))]
            if len(data)==0:
                continue
            data = pad_with_zeros(data,frange[0],frange[1])
            fitter.xdata, fitter.ydata = data[:,0], data[:,1]
            fitter.perform_fit()
            if fitter.fit_success:
                for fit in fitter.p.reshape(fitter.nexp,3):
                    meas.fits.append(fit)
        self.refresh_display()

    def _clear_fits_fired(self):
        for meas in self.measurements:
            meas.fits = []
        self.refresh_display()


    def get_fits_list(self):
        def keyf(data):
            return data[0],data[2]
        fits = []
        for meas in self.measurements:
            if len(meas.fits):
                fits.extend([[meas.ex_wl, a,m,s, gaussian_integral(meas.ex_wl, 1000, a, m, s, meas.resolution)] for a,m,s in meas.fits])
        if len(fits):
            return np.asarray(sorted(fits,key=keyf))
        else:
            return np.asarray([[0.0,0.0,0.0,0.0, 0.0]])

    def _refresh_fired(self):
        self.refresh_display()


    def _clear_fired(self):
        self.display.clear_selections()
        self.refresh_display()
        self.display.configure_selector(peaks=True)

    def refresh_display(self):
        self.fits_list = self.get_fits_list()
        self.display.remove_subplots()
        self.display.add_subplots(self.display.nplots)
        if len(self.display.axs):
            for meas in self.measurements:
                meas.plot_data(ax=self.display.axs[0], legend=False)
                #meas.plot_fits(ax=self.display.axs[1], legend=False, frange=self.display.frange)
        self.set_titles()
        self.display.draw()
        self.display.configure_selector()

    def set_titles(self):
        self.display.set_title(self.name,size=12,y=1.0)
        if len(self.display.axs):
            self.display.axs[0].set_title('BG Corrected Data', fontsize=11)
            self.display.axs[1].set_title('Fits', fontsize=11)

            self.display.axs[0].set_xlabel('')
            self.display.axs[1].set_xlabel('Emission Wavelength')
            self.display.axs[0].set_ylabel('Counts')
            self.display.axs[1].set_ylabel('Counts')

class FittingTool2D(FittingToolBase):

    experiment = Any()

    interp_method = Enum('linear', ['cubic', 'linear', 'nearest'])
    nlevel = Int(70)
    set_levels = Enum('log',['linear', 'log'])
    level_range = Tuple((1e0, 1e5), cols=2, labels=['min', 'max'])

    view = View(
        VGroup(
            VGroup(

                HGroup(
                    Item(name='refresh', show_label=False),
                    Item(name='clear', show_label=False, ),
                    Item(name='clear_fits', show_label=False, ),

                    spring,
                    ),
                HGroup(Item(name='interp_method', label='Interpolation Method', ),
                       Item(name='nlevel', label='Contour Levels', ),
                       Item(name='set_levels', label='Scale', ),
                       #Item(name='level_range', style='custom', label='Range', ),
                       spring,

                       ),
                VGroup(
                    HGroup(Item(name='editing', style='custom', label='Edit', ),
                           Item(name='perform_fit', show_label=False, ),
                           Item(name='message', style='readonly', show_label=False, springy=True),
                           spring,),
                    Item(name='display', style='custom', show_label=False),
                      show_border=True, label='Plots'),

            ),

            VGroup(
                Group(Item(name='fits_list', show_label=False, editor=ArrayViewEditor(
                    titles=['Amplitude','X0', 'Y0', 'SigmaX',
                            'SigmaY', 'Theta', ],
                    format='%g',

                    show_index=False, )),
                      show_border=True, label='Fit Result', scrollable=True),

            ),

        ),
        buttons=['OK'],
        title='Fitting Tool',
        kind='live',

        scrollable=True,
        resizable=True,
        height=800,
        width=1200,
    )
    def __init__(self,experiment,**kwargs):
        super(FittingTool2D, self).__init__()
        self.name = kwargs.get('name','Fitting Plots')
        self.experiment = experiment
        #self.refresh_display()

    def _interp_method_changed(self):
        self.experiment.has_mesh = False

    def _display_default(self):
        return FittingDataPlot2D()

    def _fits_list_default(self):
        return self.get_fits_list()

    def _fitter_default(self):
        return SpectrumFitter2D()

    def _perform_fit_fired(self):
        if self.has_peaks and self.has_frange:
            self.message = ' '
        else:
            if self.has_peaks:
                self.message='Please select a Fit Range'
            elif self.has_frange:
                self.message = 'Please select Peaks'
            else:
                self.message = 'Please select Peaks and a Fit Range '
            return
        peaks = []
        for xmid,ymid,width,height in self.display.peaks:
            if width>10 and height>10:
                peaks.append([xmid,ymid,width,height])
        if len(peaks)==0:
            self.message = 'Selected Peaks are too small'
            return
        fitter = SpectrumFitter2D()
        fitter.peaks=peaks
        frangex = self.display.frangex
        frangey = self.display.frangey
        X,Y,Z = self.experiment.collect_XYZ_arrays()
        idx = np.where(np.logical_and(
                        np.logical_and(X>=frangex[0],
                                       X<=frangex[1]),
                        np.logical_and(Y >= frangey[0],
                                       Y <= frangey[1],)))

        if not all([len(X[idx]),len(Y[idx]),len(Z[idx])]):
            return
        fitter.xdata, fitter.ydata, fitter.zdata = X[idx], Y[idx], Z[idx]

        fitter.perform_fit()
        if fitter.fit_success:
            if fitter.nexp>1:
                p = fitter.p.reshape(fitter.nexp,6)
            else:
                p = np.asarray([fitter.p.ravel()])
            self.experiment.fit_results = p
            self.fits_list = p
            self.fit_result = fitter.result_object()

        self.fitter = fitter
        self.refresh_display()

    def _clear_fits_fired(self):
        self.fits_list = np.asarray([[0.0]*6])
        self.refresh_display()

    def get_fits_list(self):
        if len(self.experiment.fit_results):
            return list(self.experiment.fit_results)
        else:
            return np.asarray([[0.0]*6])

    def _refresh_fired(self):
        self.refresh_display()

    def _clear_fired(self):
        self.display.clear_selections()
        self.display.draw_patches()
        self.refresh_display()
        self.display.configure_selector(peaks=True)

    def refresh_display(self):
        self.fits_list = self.get_fits_list()
        self.display.remove_subplots()
        self.display.add_subplots(2)
        if len(self.display.axs):
            self.experiment.plot_2d_contour(figure=self.display.figure,
                                            axs=self.display.axs[0],
                                            setlabels=False,
                                            colbar=False,
                                            nlevel = self.nlevel,
                                            #bin = self.bin,
                                            #nbins = self.nbins,
                                            interp_method = self.interp_method,
                                            set_levels = self.set_levels,
                                            #level_range = self.level_range
                                            )


        if np.any(self.fits_list):
            fitter = SpectrumFitter2D() #self.fitter
            fitter.peaks = [[]]*(self.fits_list.size/6)
            fitter.xdata, fitter.ydata, fitter.zdata = self.experiment.collect_XYZ_arrays()
            fitter.p = self.fits_list.ravel()
            fitter.plot_data(figure=self.display.figure,
                                axs=self.display.axs[1],
                             nlevel=self.nlevel,
                             interp_method=self.interp_method,)
            fitter.plot_fit(figure=self.display.figure,
                                axs=self.display.axs[1],
                            nlevel=self.nlevel,
                           )


        self.set_titles()
        self.display.draw_patches()
        self.display.configure_selector()

    def set_titles(self):
        self.display.set_title(self.name,size=12,y=1.0)
        if len(self.display.axs):
            self.display.axs[0].set_title('BG Corrected Data', fontsize=11)
            self.display.axs[1].set_title('Fit', fontsize=11)
            self.display.add_common_labels(xlabel='Excitation Wavelength',ylabel='Emission Wavelength')
            #self.display.axs[0].set_xlabel('')
            #self.display.axs[1].set_xlabel('Excitation Wavelength')
            #self.display.axs[0].set_ylabel('Emission Wavelength')
            #self.display.axs[1].set_ylabel('')

