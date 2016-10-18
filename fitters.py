from traits.api import *
from traitsui.api import *
from traitsui.ui_editors.array_view_editor import ArrayViewEditor
from data_plot_viewers import DataPlotEditorBase
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from data_plot_viewers import FittingDataPlot1D
from auxilary_functions import twoD_Gaussian
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import griddata
from fit_results import FitResult1D, FitResult2D, FitResultBase



class SpectrumFitterBase(HasTraits):
    result_factory = FitResultBase

    xdata = Array()
    ydata = Array()
    peaks = List()

    normalize = Bool(False)
    posdef = Bool(True)
    nbins = Int(0)

    nexp = Property(Int)
    fit_fcn = Property(Function)

    p = Array()
    pcov = Array()

    chi2 = Property()
    fit_f = Property(Array)
    fit_success = Bool(True)

    #def __init__(self,):
        #super(SpectrumFitter, self).__init__()

    ####       GUI     ####
    fit_data = Button('Fit Data')
    view = View(


    )

    def _get_fit_fcn(self):
        raise NotImplementedError

    def _get_chi2(self):
        raise NotImplementedError

    def _get_fit_f(self):
        raise NotImplementedError

    def _get_nexp(self):
        raise NotImplementedError

    def perform_fit(self):
        raise NotImplementedError

    def plot_data(self, title=' ', figure=None, axs = None, titlesize=12):
        raise NotImplementedError


    def result_object(self):
        result = self.result_factory()
        values = self.get(result.editable_traits())
        result.set(trait_change_notify=False, **values)

class SpectrumFitter1D(SpectrumFitterBase):
    result_factory = FitResult1D
    nparams = 3
    def _get_fit_fcn(self):
        N = self.nexp
        def gaussians(x, *p):
            if p is None:
                return np.zeros_like(x)
            params = np.asarray(p).reshape(N,3)
            return np.sum([a * np.exp(-(x - m) ** 2 / (2 * s**2)) for a,m,s in params],axis=0)
        return gaussians

    def _get_nexp(self):
        return len(self.peaks)

    def _get_fit_f(self):
        if not self.fit_success:
            self.perform_fit()
        return self.fit_fcn(self.xdata, self.p)

    def _get_chi2(self):
        return np.sum((self.fit_data()-self.ydata)**2)/(self.ydata.size-self.p.size)

    def perform_fit(self):

        if self.nbins:
            xdata = np.mean(np.array_split(self.xdata, self.nbins,axis=0), axis=1)
            ydata = np.mean(np.array_split(self.ydata, self.nbins, axis=0), axis=1)
        else:
            xdata, ydata = self.xdata, self.ydata

        if self.normalize:
            ydata = ydata/np.mean(np.diff(xdata))

        p0 = []
        for xmin, xmax in self.peaks:
            p0.append(ydata[np.where(np.logical_and(xdata<=xmax, xdata>=xmin))].max())
            p0.append((xmax+xmin)/2.0)
            p0.append((xmax-xmin)/2.0)
        try:
            if self.posdef:
                bnds = (0, np.inf)
            else:
                bnds = (-np.inf, np.inf)
            self.p, self.pcov = curve_fit(self.fit_fcn, xdata, ydata, p0=p0, bounds=bnds)
            self.fit_success = True
        except:
            self.fit_success = False



    def plot_data(self, title=' ', figure=None, axs = None, titlesize=12):
        if figure is None:
            fig = plt.figure()
        else:
            fig = figure
        if axs is None:
            ax = fig.add_subplot(111)
        else:
            ax = axs
        ax.plot(self.xdata, self.ydata, '.', label='Data')
        ax.plot(self.xdata, self.fit_f, '--',label='Fit')


        ax.set_title(title, fontsize=titlesize)
        ax.set_xlabel('Emission Wavelength')
        ax.set_ylabel('Counts')
        legend = ax.legend(shadow=True)

        if figure is None:
            plt.show()
        else:
            fig.canvas.draw()


class SpectrumFitter2D(SpectrumFitterBase):
    result_factory = FitResult2D
    nparams = 6
    shape = Tuple()
    zdata = Array()


    def _get_nexp(self):
        return len(self.peaks)

    def _get_fit_fcn(self):
        N = self.nexp
        def gaussians((x,y), *p):
            if p is None:
                return np.zeros_like(x)
            params = np.asarray(p).reshape(N,6)
            return np.sum([twoD_Gaussian((x,y), a, x0, y0, sx, sy,theta) for a,x0,y0,sx,sy,theta in params],axis=0)

        return gaussians

    def _get_fit_f(self):
        if not len(self.p):
            self.perform_fit()
        return self.fit_fcn((self.xdata, self.ydata), self.p)

    def _get_chi2(self):
        return np.sum((self.fit_data() - self.zdata) ** 2) / (self.zdata.size - self.p.size)

    def perform_fit(self):
        if self.nbins:
            xdata = np.mean(np.array_split(self.xdata, self.nbins, axis=0), axis=1)
            ydata = np.mean(np.array_split(self.ydata, self.nbins, axis=0), axis=1)
            zdata = np.mean(np.array_split(self.zdata, self.nbins, axis=0), axis=1)
        else:
            xdata, ydata, zdata = self.xdata, self.ydata, self.zdata

        if self.normalize:
            zdata = zdata / np.mean(np.diff(ydata))

        p0 = []
        for x0, y0, a, b in self.peaks:
            mask = ((xdata - x0) / a) ** 2 + ((ydata - y0) / b) ** 2 <= 1
            p0.extend([zdata[mask].max(), x0, y0, a, b, 0.0])

        try:
            if self.posdef:
                bnds = (0, np.inf)
            else:
                bnds = (-np.inf, np.inf)
            self.p, self.pcov = curve_fit(self.fit_fcn, (xdata, ydata), zdata, p0=p0, bounds=bnds)
            self.fit_success = True
        except:
            self.fit_success = False

    def plot_data(self, title=' ', figure=None, axs=None, titlesize=12, step=0.5,
                  image=False, nlevel=50, scale='log', interp_method='linear'):
        grid_x, grid_y = np.mgrid[self.xdata.min():self.xdata.max():step,
                         self.ydata.min():self.ydata.max():step]
        xy = np.empty((len(self.xdata), 2))
        xy[:,0], xy[:,1] = self.xdata, self.ydata
        grid_z = griddata(xy, self.zdata / step, (grid_x, grid_y), method=interp_method, fill_value=0.0)
        if figure is None:
            fig = plt.figure()
        else:
            fig = figure
        if axs is None:
            ax = fig.add_subplot(111)
        else:
            ax = axs

        minlev, maxlev = grid_z.min(), grid_z.max()
        if scale is 'linear':
            levels = np.linspace(minlev,maxlev,nlevel)
            norm = None
        else:
            if minlev<0.001:
                minlev=0.001
            lev_exp = np.linspace(np.floor(np.log10(minlev)), np.ceil(np.log10(maxlev)),nlevel)
            levels = np.power(10, lev_exp)
            norm=colors.LogNorm()
        #levels = np.linspace(grid_z.min(),  grid_z.max(), nlevel,)
        contfplot = ax.contourf(grid_x, grid_y, grid_z, cmap=cm.jet, levels=levels,norm=norm )
        #im = ax.imshow(grid_z.T, extent=(grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()), origin='lower',
                        #cmap=cm.jet)

        if title is not ' ':
            fig.suptitle(title)
        # plt.imshow(Z, extent=(X.min(), X.max(), Y.min(), Y.max()), origin='lower')
        if figure is None:
            plt.show()
        else:
            fig.canvas.draw()
        return fig,ax

    def plot_fit(self, title=' ', figure=None, axs=None, titlesize=12, step=0.5, image=False, nlevel=50,scale='log'):

        grid_x, grid_y = np.mgrid[self.xdata.min():self.xdata.max():step,
                         self.ydata.min():self.ydata.max():step]
        #xy = np.empty((len(self.xdata), 2))
        #xy[:, 0], xy[:, 1] = self.xdata, self.ydata
        grid_z =  self.fit_fcn((grid_x, grid_y),self.p).reshape(grid_x.shape) #griddata(xy, self.fit_f / step, (grid_x, grid_y), method='cubic', fill_value=0.0)
        if figure is None:
            fig = plt.figure()
        else:
            fig = figure
        if axs is None:
            ax = fig.add_subplot(111)
        else:
            ax = axs
        minlev, maxlev = grid_z.min(), grid_z.max()
        if scale is 'linear':
            levels = np.linspace(minlev, maxlev, nlevel)
            norm = None
        else:
            if minlev < 0.001:
                minlev = 0.001
            lev_exp = np.linspace(np.floor(np.log10(minlev)), np.ceil(np.log10(maxlev)), nlevel)
            levels = np.power(10, lev_exp)
            norm = colors.LogNorm()
        #levels = np.linspace(grid_z.min(), grid_z.max(), nlevel)
        #contfplot = ax.contourf(grid_x, grid_y, grid_z, cmap=cm.jet, levels=levels, norm=norm)
        contour = ax.contour(grid_x, grid_y, grid_z, levels=levels,colors='k')  # cmap=cm.jet, norm=norm
        if title is not ' ':
            fig.suptitle(title)
        # plt.imshow(Z, extent=(X.min(), X.max(), Y.min(), Y.max()), origin='lower')
        if figure is None:
            plt.show()
        else:
            fig.canvas.draw()
        return fig, ax

#### Unit Tests ####

if __name__=='__main__':
    N = 2
    def gaussians(x, *p):
        if p is None:
            return np.zeros_like(x)
        params = np.asarray(p).reshape(N, 3)
        # if N == 1:
        # return p[0]* np.exp(-(x - p[1]) ** 2 / (2 * p[2]))
        return np.sum([a*np.exp(-(x - m)/(2*s** 2)) for a, m, s in params], axis=0)

    xdata = np.linspace(0, 10, 10000)
    y = gaussians(xdata,[5.5, 1.3, 0.8,4.0, 6.0, 1.7] )
    ydata = y + 0.3 * np.random.normal(size=len(xdata))

    fitter1 = SpectrumFitter1D(xdata=xdata, ydata=ydata, peaks=[(0.5,2.5), (5.0, 6.5)],normalize=False, nbins=0)
    ax = fitter1.plot_data()


    grid_x, grid_y = np.mgrid[0:10:0.2, 0:10:0.2]


    grid_z = twoD_Gaussian((grid_x,grid_y),5.0,4.0,3.0,2.1,1.2,0.0).reshape(grid_x.shape)
    grid_z += twoD_Gaussian((grid_x,grid_y),2.0,1.3,6.0,1.1,3.2,0.0).reshape(grid_x.shape)
    grid_z += 0.2 * np.random.normal(size=grid_x.shape)

    fitter2 = SpectrumFitter2D(xdata=grid_x.ravel(), ydata=grid_y.ravel(),zdata=grid_z.ravel(),
                               shape=grid_z.shape, peaks=[[4.2,3.8,1.8,1.0],[1.1,5.7,2.1,3.5]], normalize=False, nbins=0)
    figure = plt.figure()

    fig,ax = fitter2.plot_data(figure=figure,step=0.1)
    fitter2.plot_fit(step=0.1, axs=ax,figure=fig)
    plt.show()
    print fitter2.p.reshape(2,6)


