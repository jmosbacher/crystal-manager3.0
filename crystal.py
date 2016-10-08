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
import numpy as np
import random
import pandas as pd
import matplotlib
matplotlib.style.use('ggplot')
import pandas as pd
try:
    import cPickle as pickle
except:
    import pickle

from experiment import BaseExperiment,SpectrumExperiment, ExperimentTableEditor
from measurement import SpectrumMeasurement
from auxilary_functions import merge_spectrums
from data_importing import AutoSpectrumImportTool


class Crystal(HasTraits):
    """

    """
    main = Any()
    name = Str('New Crystal')
    sn = Str('SN')
    desc = Str('Description')
    notes =Str('Notes...')

    #####       Data      #####


    #####       Flags      #####
    is_selected = Bool(False)


    #####       Info      #####

    #####       UI      #####
    add_type = Enum(['Spectrum', 'Raman', 'Anealing'])
    add_exp = Button('Add Experiment')
    edit = Button('Open')
    remove = Button('Remove Selected')
    select_all = Button('Select All')
    unselect_all = Button('Un-select All')
    plot_selected = Button('Plot')
    merge = Button('Merge')
    #import_exp = Button('Import Experiment')

    #####       Visualization     #####
    kind = Enum('Spectrum',['Spectrum', 'Raman'])
    alpha = Float(0.6)


class CrystalTableEditor(TableEditor):

    columns = [
                CheckboxColumn(name='is_selected', label='', width=0.08, horizontal_alignment='center', ),
                ObjectColumn(name = 'name',label = 'Name',width = 0.25,horizontal_alignment = 'left',editable=True),

                ObjectColumn(name='sn', label='SN', width=0.25, horizontal_alignment='left', editable=True),

                #ObjectColumn(name = 'ex_wl_range',label = 'Excitation WLs',horizontal_alignment = 'center',
                             #width = 0.13,editable=False),

                #ObjectColumn(name = 'em_wl_range',label = 'Emission WLs',width = 0.13,
                             #horizontal_alignment = 'center',editable=False),
                #ObjectColumn(name = 'em_pol',label = 'Emission POL',width = 0.08,horizontal_alignment = 'center'),

                ObjectColumn(name='experiment_cnt', label='Experiments', width=0.08,
                             horizontal_alignment='center',editable=False),

                ObjectColumn(name='desc', label='Description', width=0.08,
                             horizontal_alignment='center', editable=False),
              ]

    auto_size = True
    sortable = False
    editable = True

if __name__ == '__main__':
    app = Crystal()
    app.configure_traits()