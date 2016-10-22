import os
from traits.api import *
from traitsui.api import *
from traits.api import HasTraits
from traitsui.api import View, Item
from numpy import sin, cos, linspace, pi
from matplotlib.widgets import  RectangleSelector
from matplotlib.widgets import SpanSelector
from mpl_figure_editor import MPLFigureEditor, MPLInitHandler
from matplotlib.figure import Figure
from matplotlib.widgets import  EllipseSelector, RectangleSelector
from matplotlib.patches import Ellipse, Rectangle
import matplotlib.pyplot as plt
import numpy as np

class DataPlotEditorBase(HasTraits):

    figure = Instance(Figure, (),transient=True)
    axs = List([],transient=True)
    nplots = Int(1)
    layout = Enum('vertical',['horizontal','vertical'])

    view = View(Item('figure', editor=MPLFigureEditor(),
                     show_label=False,
                     height=400,
                     width=400,

                     springy=True),
                handler=MPLInitHandler,
                resizable=True,
                kind='live',
                     )

    def __init__(self):
        super(DataPlotEditorBase, self).__init__()
        self.figure.patch.set_facecolor('none')
        #self.add_subplots(self.nplots)


    def mpl_setup(self):
        self.figure.patch.set_facecolor('none')
        self.add_subplots(self.nplots)

    def remove_subplots(self):
        self.figure.clf(keep_observers=True)
        self.axs = []
        #self.display.figure.canvas.draw()

    def remove_figure(self):
        plt.close(self.figure)

    def add_subplots(self, num):
        self.figure.patch.set_facecolor('none')
        self.axs = []
        for n in range(1, num + 1):
            if self.layout == 'vertical':
                self.axs.append(self.figure.add_subplot(num, 1, n, axisbg='#F4EAEA',zorder=2))
            elif self.layout == 'horizontal':
                self.axs.append(self.figure.add_subplot(1, num,n , axisbg='#F4EAEA',zorder=2)) #FFFFCC
        return self.axs

    def add_common_labels(self,xlabel=None,ylabel=None):
        ax = self.figure.add_subplot(111, axisbg='none', frameon=False,zorder=1)
        ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
        ax.grid(b=False)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        self.figure.sca(self.axs[0])

    def clear_plots(self):
        for ax in self.axs:
            ax.cla(keep_observers=True)
        self.figure.canvas.draw()

    def set_title(self, title=' ', size=13,y=0.98):
        self.figure.suptitle(title,fontsize=size,y=y)

class IntegrationDataPlot1D(DataPlotEditorBase):
    nplots = 3
    axvspn = List([],transient=True)
    selections = List([],transient=True)
    span = Any(transient=True)


    def configure_selector(self):
        if len(self.axs):
            self.span = SpanSelector(self.axs[0], self.onselect, 'horizontal', useblit=True,
                             rectprops=dict(alpha=0.5, facecolor='red'))
        if self.figure is not None:
            self.figure.canvas.draw()

    def plot_data(self,data,plot_num,title=' '):
        self.axs[plot_num].plot(data[:,0],data[:,1],)
        #plt.pause(1)

    def clear_selections(self):
        self.selections = []
        for span in self.axvspn:
            try:
                span.remove()
            except:
                pass
        self.axvspn = []



    def onselect(self,xmin, xmax):
        if xmax-xmin == 0:
            return
        self.selections.append((xmin, xmax))
        for ax in self.axs:
            self.axvspn.append(ax.axvspan(xmin, xmax, color='red', alpha=0.4))
        if self.figure is not None:
            self.figure.canvas.draw()

    def redraw_selections(self):
        self.axvspn=[]
        for xmin,xmax in self.selections:
            for ax in self.axs:
                self.axvspn.append(ax.axvspan(xmin, xmax, color='red', alpha=0.4))
        if self.figure is not None:
            self.figure.canvas.draw()

    def mpl_setup(self):
        if len(self.axs) != self.nplots:
            self.add_subplots(self.nplots)
        self.configure_selector()


class FittingDataPlot1D(DataPlotEditorBase):
    nplots = 2
    range_axvspn = Any(transient=True)
    peaks_axvspn = List([],transient=True)
    frange = Tuple((0.0,0.0),transient=True)
    peaks = List([],transient=True)
    range_selector = Any(transient=True) #Instance(SpanSelector)
    peaks_selector = Any(transient=True) #(SpanSelector)
    editing = Enum('Peaks',['Range','Peaks'],transient=True)
    has_frange = Property(Bool)
    has_peaks = Property(Bool)

    def _get_has_frange(self):
        if (self.frange[1]-self.frange[0])>10:
            return True
        else:
            return False

    def _get_has_peaks(self):
        if len(self.peaks):
            return True
        else:
            return False

    def clear_spans(self,frange=True,peaks=True):
        if frange:
            try:
                self.range_axvspn.remove()
            except:
                pass

        if peaks:
            for axvspn in self.peaks_axvspn:
                try:
                    axvspn.remove()
                except:
                    pass

    def clear_selections(self):

        self.clear_spans(frange=True,peaks=False)
        self.peaks = []
            #self.frange = (0.0,0.0)



    def draw(self,frange=True,peaks=True):
        if all([frange, len(self.axs), len(self.frange)]):
            self.range_axvspn = self.axs[0].axvspan(self.frange[0], self.frange[1], color='g', alpha=0.1)

        if all([peaks, len(self.axs)]):
            for xmin,xmax in self.peaks:
                self.peaks_axvspn.append(self.axs[0].axvspan(xmin, xmax, color='red', alpha=0.4))
        if self.figure is not None:
            if self.figure.canvas is not None:
                self.figure.canvas.draw()

    def mpl_setup(self):
        self.add_subplots(self.nplots)
        self.configure_selector(peaks=True)
        #self.figure.canvas.draw()
        #self.activate_selector()

    def onselect(self,xmin,xmax):
        if self.editing=='Range':
            self.frange = (xmin,xmax)
            self.clear_spans(peaks=False)
            self.draw(peaks=False)
            #self.figure.canvas.draw()

        elif self.editing=='Peaks':
            self.peaks.append((xmin,xmax))
            self.clear_spans(frange=False)
            self.draw(frange=False)
            #self.figure.canvas.draw()

    def configure_selector(self,frange=False,peaks=False):
        self.peaks_selector = SpanSelector(self.axs[0], self.onselect, 'horizontal', useblit=True,
                                               rectprops=dict(alpha=0.5, facecolor='red'))
        self.peaks_selector.set_active(peaks)

        self.range_selector = SpanSelector(self.axs[0], self.onselect, 'horizontal', useblit=True,
                                     rectprops=dict(alpha=0.5, facecolor='g'))
        self.range_selector.set_active(frange)

    def activate_selector(self,frange=False,peaks=False):
        if self.peaks_selector is not None:
            self.peaks_selector.set_active(peaks)
        if self.range_selector is not None:
            self.range_selector.set_active(frange)


    def _editing_changed(self,new):
        if new=='Range':
            self.configure_selector(frange=True,peaks=False)
        elif new=='Peaks':
            self.configure_selector(frange=False, peaks=True)


class SingleDataPlot(DataPlotEditorBase):
    pass


class IntegrationDataPlot2D(DataPlotEditorBase):
    pass

class FittingDataPlot2D(DataPlotEditorBase):
    nplots = 2
    layout = 'horizontal'
    range_rect = Any(transient=True)
    peaks_ellipses = List([],transient=True)

    frangex = Tuple((0.0, 0.0),transient=True)
    frangey = Tuple((0.0, 0.0),transient=True)
    peaks = List([],transient=True)
    range_selector = Any(transient=True)  # Instance(SpanSelector)
    peaks_selector = Any(transient=True)  # (SpanSelector)
    editing = Enum('Peaks', ['Range', 'Peaks'])
    has_frange = Property(Bool)
    has_peaks = Property(Bool)

    def _get_has_frange(self):
        if (self.frangex[1]-self.frangex[0])>10 and (self.frangey[1]-self.frangey[0])>10:
            return True
        else:
            return False

    def _get_has_peaks(self):
        if len(self.peaks):
            return True
        else:
            return False

    def clear_patches(self, frange=True, peaks=True):
        if frange:
            try:
                self.range_rect.remove()
                self.range_rect = None
            except:
                pass

        if peaks:
            for ellipse in self.peaks_ellipses:
                try:
                    ellipse.remove()
                    self.peaks_ellipses.remove(ellipse)
                except:
                    pass

    def clear_selections(self):
        self.clear_patches()
        self.peaks = []
        self.frangex = (0.0,0.0)
        self.frangey = (0.0, 0.0)

    def draw_patches(self, frange=True, peaks=True):
        if all([frange, len(self.axs), len(self.frangex), len(self.frangey)]):
            self.range_rect = self.draw_rectangle(self.frangex, self.frangey, color='g', alpha=0.15)

        if all([peaks, len(self.axs), len(self.peaks)]):
            for p in self.peaks:
                self.peaks_ellipses.append(self.draw_ellipse(*p, color='r', alpha=0.4))

        if self.figure is not None:
            if self.figure.canvas is not None:
                self.figure.canvas.draw()

    def mpl_setup(self):
        self.add_subplots(self.nplots)
        self.configure_selector(peaks=True)
        # self.figure.canvas.draw()
        # self.activate_selector()



    def draw_rectangle(self,xs,ys,alpha=0.2,color='g'):
        xy = (min(xs),min(ys))
        width = np.abs(np.diff(xs))
        height = np.abs(np.diff(ys))
        re = Rectangle(xy, width, height, angle=0.0)
        ax = self.axs[0]
        ax.add_artist(re)
        re.set_alpha(alpha=alpha)
        re.set_facecolor(color)
        return re

    def draw_ellipse(self,xmid,ymid,width,height,alpha=0.4,color='r'):
        el = Ellipse(xy=(xmid, ymid), width=width,
                     height=height, angle=0)
        ax = self.axs[0]
        ax.add_artist(el)
        #el.set_clip_box(ax.bbox)
        el.set_alpha(alpha=alpha)
        el.set_facecolor(color)
        return el

    def rect_onselect(self, eclick, erelease):
        xs = [eclick.xdata, erelease.xdata]
        ys = [eclick.ydata, erelease.ydata]
        #print xs, ys
        self.frangex = (min(xs), max(xs))
        self.frangey = (min(ys), max(ys))
        self.clear_patches(peaks=False)
        self.draw_patches(peaks=False)

    def ellipse_onselect(self, eclick, erelease):
        xs = [eclick.xdata, erelease.xdata]
        ys = [eclick.ydata, erelease.ydata]
        xmid, ymid = np.mean(xs), np.mean(ys)
        width, height = np.abs(np.diff(xs)), np.abs(np.diff(ys))
        #print [xmid, ymid, width, height]
        self.peaks.append([xmid, ymid, width, height])
        self.clear_patches(frange=False)
        self.draw_patches(frange=False)


    def configure_selector(self, frange=False, peaks=False):
        self.range_selector = RectangleSelector(self.axs[0], self.rect_onselect,
                          drawtype='box', useblit=True,
                          button=[1, 3],  # don't use middle button
                          minspanx=15, minspany=15,
                          spancoords='pixels',
                          #interactive=True,
                          rectprops=dict(alpha=0.5, facecolor='g'))


        self.peaks_selector = EllipseSelector(self.axs[0], self.ellipse_onselect,
                                              drawtype='line',
                                              button=[1, 3],  # don't use middle button
                                              spancoords='pixels',
                                              useblit=True,
                                              minspanx=10,
                                              minspany=10,
                                              rectprops=dict(alpha=0.5, facecolor='red'))

        self.peaks_selector.set_active(peaks)
        self.range_selector.set_active(frange)

    def activate_selector(self, frange=False, peaks=False):
        if self.peaks_selector is not None:
            self.peaks_selector.set_active(peaks)
        if self.range_selector is not None:
            self.range_selector.set_active(frange)

    def _editing_changed(self, new):
        if new == 'Range':
            self.configure_selector(frange=True, peaks=False)
        elif new == 'Peaks':
            self.configure_selector(frange=False, peaks=True)


class Test(HasTraits):
    display = Instance(FittingDataPlot2D,())
    toggle = Button('Toggle')
    first = Bool(True)
    view = View('toggle',Item(name='display',style='custom',show_label=False)
    )

    def __init__(self):
        super(Test,self).__init__()

    def _toggle_fired(self):

        if self.display.editing == 'Range':
            self.display.editing = 'Peaks'
        elif self.display.editing == 'Peaks':
            self.display.editing = 'Range'
        if self.first:
            self.display.axs[0].plot(np.random.random(100))
            self.display.figure.canvas.draw()
            self.first=False

if __name__=='__main__':
    test = Test()
    test.configure_traits()