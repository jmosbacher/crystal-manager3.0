from traits.api import *
from traitsui.api import *
from traitsui.ui_editors.array_view_editor import ArrayViewEditor
from data_plot_viewers import DataPlotEditorBase
import matplotlib.pyplot as plt
from experiment import SpectrumExperiment,BaseExperiment, ExperimentTableEditor
from measurement import SpectrumMeasurement
from compare_experiments import ExperimentComparison
from data_plot_viewers import FittingDataPlot1D
from auxilary_functions import twoD_Gaussian
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import griddata


class FitResultBase(HasTraits):
    normalize = Bool()
    posdef = Bool()
    nbins = Int()
    nexp = Int()
    fit_fcn = Function()
    p = Array()
    pcov = Array()


    def fit_data(self, coord):
        return self.fit_fcn(coord,self.p)

    def calc_chi2(self,coord,f):
        return np.sum((self.fit_data(coord)-f)**2)

class FitResult1D(FitResultBase):
    pass


class FitResult2D(FitResultBase):
    pass




