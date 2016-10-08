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
from project import Project, ProjectTableEditor
from experiment import SpectrumExperiment
from measurement import SpectrumMeasurement
from workspaces import WorkSpace
from traitsui.qt4.tree_editor \
    import NewAction, CopyAction, CutAction, \
           PasteAction, DeleteAction, RenameAction
from saving import CanSaveMixin
from handlers import MainSaveHandler
from time import sleep
try:
    import cPickle as pickle
except:
    import pickle



class MainApplication(CanSaveMixin):
    """

    """
    #####       File Menu      #####
    save_a = Action(name = 'Save Workspace', action = 'save')
    save_as_a = Action(name = 'Save As', action = 'saveAs')
    load_all = Action(name = 'Reload', action = 'Reload',enabled_when='filepath')
    load_from_a = Action(name = 'Load from File', action = 'loadProject')
    #comp_int_tool = Action(name='Comparison Integration Tool', action='comp_integration_tool')
    #exp_int_tool = Action(name='Experiment Integration Tool', action='exp_integration_tool')
    #comp_tool = Action(name='Comparison Tool', action='comp_tool')
    #plot_tool = Action(name='Plotting Tool', action='plot_tool')
    #fit_tool = Action(name='Fitting Tool', action='fit_tool')

    #####       Tree Menus      #####
    #add_workspace = Action(name='New Worspace', action='new_project')
    add_project = Action(name='New Project', action='new_project')
    add_experiment = Action(name='New Experiment', action='new_experiment')
    add_measurement = Action(name='New Measurement', action='new_measurement')

    #####       Autosave Menu      #####
    cfg_autosave = Action(name = 'Configure Autosave', action = 'cfg_autosave')
    autosave = Bool(False)
    autosaveInterval = Int(300)



    #####       Data      #####
    ws = Instance(WorkSpace)
    selected = Instance(HasTraits)


    #####       UI      #####
    status = Str('Autosave Disabled')
    import_selected = Action(name = 'Add data', action = 'import_selected')
    merge_selected = Action(name = 'Merge Selected', action = 'merge_selected')
    rem_selected = Action(name = 'Remove', action = 'rem_selected')
    add_new = Action(name = 'New Project', action = 'add_new')
    edit = Action(name = 'Open', action = 'edit')



    #edit = Button('Edit')

    #####       GUI View     #####
    traits_view = View(
            HSplit(
                Item(name='ws', show_label=False,
        editor=TreeEditor(
        nodes=[
            TreeNode(node_for=[WorkSpace],
                     auto_open=True,
                     children='projects',
                     label='name',
                     #add=[Project],
                     menu=Menu(RenameAction,
                               add_project,
                               Separator(),
                               ),

                     ),

            TreeNode(node_for=[Project],
                     auto_open=True,
                     children='experiments',
                     label='name',
                     #add=[SpectrumExperiment],
                     menu=Menu(add_experiment,
                               RenameAction,
                               #Separator(),
                               #CopyAction,
                               #CutAction,
                               #PasteAction,
                               #Separator(),
                               #DeleteAction,
                               #Separator(),
                               ),
                     ),

            TreeNode(node_for=[SpectrumExperiment],
                     #auto_open=True,
                     children='measurements',
                     label='name',
                     #add=[SpectrumMeasurement],
                     menu=Menu(add_measurement,
                               #Separator(),
                               #CopyAction,
                               #CutAction,
                               #PasteAction,
                               #Separator(),
                               #DeleteAction,
                               #Separator(),
                               RenameAction),
                     ),

            TreeNode(node_for=[SpectrumMeasurement],
                     #auto_open=True,
                     label='name',
                     menu=Menu(RenameAction,
                               #CopyAction,
                               #CutAction,
                               #PasteAction,
                               #Separator(),
                               #DeleteAction,
                               #Separator(),
                               ),

                     )
        ],
        editable=False,
        selected='selected',
        #selection_mode = 'extended',

        ),
                     width=0.25,resizable=False),

            VGroup(Item(name='selected',show_label=False,style='custom',resizable=False),
                   scrollable=True),

            ),

        title='Spectrum Project Manager',
        scrollable=True,
        resizable=True,
        height=800,
        width=1280,
        handler=MainSaveHandler(),
        menubar=MenuBar(
            Menu(save_a, save_as_a,cfg_autosave, load_all, load_from_a,
                 name='File'),
            #Menu(exp_int_tool, comp_tool, comp_int_tool, plot_tool, fit_tool,
                #name='Tools'),

                        ),
        toolbar = ToolBar(add_new, rem_selected,edit  ),
        statusbar=[StatusItem(name='status', width=0.5), ],
    )


    #####       Initialization Methods      #####
    def _ws_default(self):
        return WorkSpace(main=self)

    #####       Methods      #####
    def _ws_changed(self):
        self.dirty = True

    def _rem_selected(self):
        self.ws.projects.remove(self.selected)

    def _add_new(self):
        new = Project(main=self)
        self.ws.projects.append(new)
        self.selected = new

    def _edit(self):
        self.selected.edit_traits()

    def save(self):
        #localdir = os.path.dirname(os.path.abspath(__file__))
        #path = os.path.join(localdir,'saved.spctrm')
        stat = self.status
        self.status = 'Saving...'
        sleep(0.3)
        with open(self.filepath,'wb') as f:
            pickle.dump(self.ws.projects, f)
        self.status = stat

    def load(self):
        #localdir = os.path.dirname(os.path.abspath(__file__))
        #path = os.path.join(localdir,'saved.spctrm')
        with open(self.filepath, 'rb') as f:
            self.ws.projects = pickle.load(f)




if __name__ == '__main__':
    app = MainApplication()
    app.configure_traits()