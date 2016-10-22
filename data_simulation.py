from traits.api import *
from traitsui.api import *
import numpy as np
from measurement import SpectrumMeasurement
from experiment import SpectrumExperiment

class MeasurementSimulator(HasTraits):
    experiment = Any()
    ex_wl = Float(350)
    max_wl = Float(1050)
    resolution = Float(0.075)

    ngauss = Int(2)
    amp_range = Tuple((100,1e7),cols=2)
    sigma_range = Tuple((10,400),cols=2)
    noise_std = Float(30)
    bg_level = Float(200.0)

    simulate = Button('Simulate')

    view = View()

    def __init__(self,experiment=None):
        super(MeasurementSimulator,self).__init__()
        self.experiment = experiment

    def random_gaussian(self,xdata):
        amp = np.random.uniform(*self.amp_range)
        mean = np.random.uniform(self.ex_wl, self.max_wl)
        sigma = np.random.uniform(*self.sigma_range)
        return amp*np.exp(-((xdata-mean)/(2*sigma))**2)+ np.random.normal(size=xdata.size,scale=self.noise_std)

    def simulate_signal(self,ngauss):
        xdata = np.arange(self.ex_wl, self.max_wl, self.resolution)
        ydata = np.sum([self.random_gaussian(xdata) for j in range(ngauss) ],axis=0)
        data = np.empty((xdata.size,2))
        data[:,0] = xdata
        data[:,1] = ydata
        return data

    def simulate_bg(self):
        xdata = np.arange(self.ex_wl, self.max_wl, self.resolution)
        data = np.empty((xdata.size,2))
        data[:,0] = xdata
        data[:,1] = np.ones(xdata.size)*self.bg_level + np.random.normal(size=xdata.size,scale=self.noise_std)
        return data

    def simulate_ref(self):
        xdata = np.arange(self.ex_wl, self.max_wl, self.resolution)
        data = np.empty((xdata.size,2))
        data[:, 0] = xdata
        data[:, 1] = self.simulate_bg() + np.random.normal(size=xdata.size,scale=self.noise_std)
        return


    def simulate_measurement(self):
        new = SpectrumMeasurement()
        new.name = '{}in_{}-{}out_Simulated'.format(self.ex_wl,self.ex_wl,self.max_wl)
        new.ex_wl = self.ex_wl
        new.em_wl = (self.ex_wl,self.max_wl)
        new.signal = self.simulate_signal(self.ngauss)
        new.bg = self.simulate_bg()
        new.ref = self.simulate_ref()
        new.is_simulated = True
        new.simulation_data = self.get([x for x in self.editable_traits() if x is not 'experiment'])

        return new



class ExperimentSimulator(HasTraits):
    project = Any()
    ex_range = Tuple((250,750), cols=2)
    ex_res = Float(10)
    em_max = Float(1050)
    em_res = Float(0.075)

    ngauss = Int(2)
    #nmeasure = Int(10)
    amp_range = Tuple((100, 1e7), cols=2)
    sigma_range = Tuple((70, 400), cols=2)
    noise_std = Float(30)
    bg_level = Float(200.0)

    simulate = Button('Simulate')

    view = View()

    def __init__(self, project=None):
        super(ExperimentSimulator, self).__init__()
        self.project = project

    def create_random_gaussian(self):
        amp = np.random.uniform(*self.amp_range)
        x0 = np.random.uniform(*self.ex_range)
        y0 = np.random.uniform(*self.em_range)
        sigma_x = np.random.uniform(*self.sigma_range)
        sigma_y = np.random.uniform(*self.sigma_range)
        theta = np.random.uniform(0, 0.005)

        def gaussian_2d(x,y):

            a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (2 * sigma_y ** 2)
            b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (4 * sigma_y ** 2)
            c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (2 * sigma_y ** 2)
            z = amp * np.exp(- (a * ((x - x0) ** 2) + 2 * b * (x - x0) * (y - y0)
                                      + c * ((y - y0) ** 2)))
            return z

        return gaussian_2d


    def simulate_signal(self,ngauss):
        xdata = np.arange(self.ex_wl, self.max_wl, self.resolution)
        ydata = np.sum([self.random_gaussian(xdata) for j in range(ngauss) ],axis=0)
        data = np.empty((xdata.size,2))
        data[:,0] = xdata
        data[:,1] = ydata
        return data


    def create_signal_fcn(self):
        def signal_fcn(x, y):
            gaussians = [self.create_random_gaussian() for x in range(self.ngauss)]
            return np.sum([g(x, y) for g in gaussians])
        return signal_fcn

    def simulate_experiment(self):
        new_exp = SpectrumExperiment()
        signal_fcn = self.create_signal_fcn()
        ex_wls = np.arange(*self.ex_range,step=self.ex_res)

        for n, ex_wl in enumerate(ex_wls):
            em_wls = np.arange(ex_wl,self.em_max,step=self.ex_res)

            meas = SpectrumMeasurement()
            meas.ex_wl = ex_wl
            meas.em_wl = (ex_wl,self.em_max)
            meas.signal = np.empty(em_wls.size,2)
            meas.signal[:, 0] = em_wls
            meas.signal[:,1] = signal_fcn(ex_wl,em_wls)

