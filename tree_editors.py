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
from main import WorkSpace
from saving import CanSaveMixin
from handlers import MainSaveHandler
from traitsui.qt4.tree_editor \
    import NewAction, CopyAction, CutAction, \
           PasteAction, DeleteAction, RenameAction




workspace_tree_editor = TreeEditor(

    nodes=[
        TreeNode(node_for=[WorkSpace],
                 auto_open=True,
                 children='projects',
                 label='name',
                 ),

        TreeNode(node_for=[Project],
                 auto_open=True,
                 children='experiments',
                 label='name',

                 add=[SpectrumExperiment]),

        TreeNode(node_for=[SpectrumExperiment],
                 auto_open=True,
                 children='measurements',
                 label='name',


                 add=[SpectrumMeasurement]),
        TreeNode(node_for=[SpectrumMeasurement],
                 auto_open=True,
                 label='name',

                                )
    ],
    editable=False,
    selected='selected'

)

'''
                 menu=Menu(NewAction,
                           Separator(),
                           CopyAction,
                           CutAction,
                           PasteAction,
                           Separator(),
                           DeleteAction,
                           Separator(),
                           RenameAction),
'''


class Test(HasTraits):
    workspace = Instance(WorkSpace)
    selected = Instance(HasTraits)
    view = View(
        HSplit(
        Item('workspace',editor=workspace_tree_editor,show_label=False,width=0.2),


    Group(Item(name='selected',style='custom',show_label=False))
        ),

    title = 'Tree Test',
    resizable = True,
    height = 800,
    width = 1200,
    #scrollable = True,
    )

    def _workspace_default(self):
        work = WorkSpace()
        projects = [Project(name='Project 1'), Project(name='Project 2'), Project(name='Project 3')]
        for n, project in enumerate(projects):
            project.name = 'Project '+str(n)
            for name in ['Exp 1', 'Exp 2', 'Exp 3']:
                exp = project.add_new_experiment()
                exp.name = name
                for namee in ['Meas 1', 'Meas 2', 'Meas 3']:
                    meas = exp.add_measurement()
                    meas.name = namee
        work.projects = projects
        return work

if __name__=='__main__':
    test = Test()
    test.configure_traits()
