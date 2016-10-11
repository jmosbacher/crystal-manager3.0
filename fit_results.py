from traits.api import *
from traitsui.api import *
from traitsui.tabular_adapter import TabularAdapter
import numpy as np

class FitResult1DAdapter(TabularAdapter):

    columns = [ 'Amplitude','Mean','Stdev', ]
    can_edit = False

class FitResult2DAdapter(TabularAdapter):

    columns = [ 'Amplitude','X0','Y0','Sigma X', 'Sigma Y','Theta', ]
    can_edit = False


class FitResultBase(HasTraits):
    nparams = Int()
    normalize = Bool()
    posdef = Bool()
    nbins = Int()
    nexp = Int()
    fit_fcn = Function()
    p = Array()
    pcov = Array()
    chi2 = Float()
    p_text = Property(Str)
    p_list = Property()



    def _get_p_list(self):
        return self.p.reshape(self.nexp,self.nparams)

    def _get_p_text(self):
        pass

    def fit_data(self, coord):
        return self.fit_fcn(coord,self.p)

    def calc_chi2(self,coord,f):
        return np.sum((self.fit_data(coord)-f)**2)

    def single_gaussian(self):
        pass

    def plot_fit_fcn(self):
        pass

    def plot_fit_gaussians(self):
        pass

    def integrate_range(self,x_range=None,y_range=None):
        pass


class FitResult1D(FitResultBase):

    view = View(HGroup(
        Item(name='normalize', label='', style='readonly'),
        Item(name='posdef', label='Positive Def', style='readonly'),
        Item(name='nbins', label='Bins', style='readonly'),
        Item(name='nexp', label='Number of Gaussians', style='readonly'),
    ),
        Item(name='p_list',
             editor=TabularEditor(auto_update=True, adapter=FitResult1DAdapter()), )

    )


class FitResult2D(FitResultBase):

    view = View(HGroup(
        Item(name='normalize', label='', style='readonly'),
        Item(name='posdef', label='Positive Def', style='readonly'),
        Item(name='nbins', label='Bins', style='readonly'),
        Item(name='nexp', label='Number of Gaussians', style='readonly'),
    ),
        Item(name='p_list',
             editor=TabularEditor(auto_update=True, adapter=FitResult2DAdapter()), )

    )




