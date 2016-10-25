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

try:
    import cPickle as pickle
except:
    import pickle
from crystal import Crystal, CrystalTableEditor
from experiment import BaseExperiment
from auxilary_functions import merge_experiments
from saving import CanSaveMixin
from compare_experiments import AllExperimentList, ExperimentComparison
from experiment import ExperimentTableEditor, SpectrumExperiment, BaseExperiment
from data_importing import ExpImportToolTab, AutoSpectrumImportTool
from import_tools import ImportToolBase
from analysis_tools import ProjectAnalysisTool
from integration_results import IntegrationResultBase
from saving import BaseSaveHandler
from traitsui.ui_editors.array_view_editor import ArrayViewEditor
from pyface.api import FileDialog, confirm, error, YES, CANCEL

class ProjectHandler(BaseSaveHandler):
    extension = Str('prj')


    def object_export_dataframe_changed(self, info):
        fileDialog = FileDialog(action='save as', title='Save As',
                                wildcard=self.wildcard,
                                parent=info.ui.control,
                                default_filename=info.object.name.replace(' ','_') + '_dataframe')

        fileDialog.open()
        if fileDialog.path == '' or fileDialog.return_code == CANCEL:
            return False
        df = info.object.make_db_dataframe()
        df.to_csv(fileDialog.path)


class Project(CanSaveMixin):
    __klass__ = 'Project'
    main = Any()
    name = Str('New Project')
    notes = Str()
    comments = Str('No Comments')

    #####       Data     #####
    experiments = List(BaseExperiment)
    comparisons = List(ExperimentComparison)
    comp_int_results = List(IntegrationResultBase)
    exp_int_results = List(IntegrationResultBase)

    experiment_cnt = Property()
    selected = Instance(BaseExperiment)

    #####       Flags      #####
    is_selected = Bool(False)

    #####       UI     #####
    add_new = Button('New Experiment')
    #import_data = Button('Add Measurements')
    remove = Button('Remove')
    edit = Button('Open')
    merge = Button('Merge')
    select_all = Button('Select All')
    unselect_all = Button('Un-select All')
    #compare = Button('Compare Tool')
    import_folders = Button('Import Folders')
    import_files = Button('Import Files')

    export_dataframe = Button('Export DataFrame')

    import_tool = Instance(ImportToolBase,transient=True)
    analysis_tool = Instance(ProjectAnalysisTool,transient=True)

    view = View(

        Tabbed(
        VGroup(

            HGroup(
                Item(name='add_new', show_label=False),
                Item(name='edit', show_label=False, enabled_when='selected'),
                Item(name='export_dataframe',show_label=False),
                #Item(name='compare', show_label=False),
                #Item(name='import_folders', show_label=False),
                #Item(name='import_files', show_label=False),
                spring,
                Item(name='select_all', show_label=False),
                Item(name='unselect_all', show_label=False),
                Item(name='remove', show_label=False, enabled_when='selected'),
                Item(name='merge', show_label=False),
                #Item(name='import_data', show_label=False, enabled_when='selected'),
            ),

            Group(

                Item(name='experiments', show_label=False, editor=ExperimentTableEditor(selected='selected')),
                show_border=True, label='Experiments'),


            Group(
                Item(name='notes', show_label=False, springy=True, editor=TextEditor(multi_line=True), style='custom'),
                show_border=True, label='Notes'),
            label='Data'),

            Group(Item(name='analysis_tool', show_label=False, style='custom'),
                  label='Analysis'),


        Group(Item(name='import_tool',show_label=False,style='custom'),
              label='Import Experiments'),

        ),

        buttons=['OK'],
        title='Project Editor',
        kind='nonmodal',
        handler = ProjectHandler(),
        scrollable=True,
        resizable=True,
        height=800,
        width=1000,

    )
    def __init__(self, **kargs):
        HasTraits.__init__(self)
        self.main = kargs.get('main',None)

    def _import_tool_default(self):
        return ImportToolBase(self)

    def _analysis_tool_default(self):
        return ProjectAnalysisTool(project=self)

    def _anytrait_changed(self,name):

        if self.main is None:
            return
        if name in ['comments','name','notes','experiments',
                    'comparisons','comp_int_results','exp_int_results']:
            self.main.dirty = True

    def _get_experiment_cnt(self):
        return len(self.experiments)

    def _edit_fired(self):
        self.selected.edit_traits()


    def _select_all_fired(self):
        for exp in self.experiments:
            exp.is_selected = True

    def _unselect_all_fired(self):
        for exp in self.experiments:
            exp.is_selected = False

    def _merge_fired(self):
        for_merge = []

        for exp in self.experiments:
            if exp.is_selected:
                for_merge.append(exp)
        main = for_merge[0]
        rest = for_merge[1:]
        for exp in rest:
            main = merge_experiments(main, exp)
            self.experiments.remove(exp)
        main.is_selected = False

    def _remove_fired(self):
        if self.selected is not None:
            self.experiments.remove(self.selected)

    def add_experiment(self):
        new = SpectrumExperiment(main=self.main)
        self.experiments.append(new)
        return new

    def _add_new_fired(self):
        self.add_experiment()

    def make_db_dataframe(self,**kwargs):
        data = {}
        for exp in self.experiments:
            data[exp.name] = exp.make_db_dataframe()
        final = pd.concat(data)
        final.index.rename('experiment',level=0,inplace=True)
        return final

    #def _import_data_fired(self):
        #self.selected.import_data()



class ProjectTableEditor(TableEditor):

    columns = [
                CheckboxColumn(name='is_selected', label='', width=0.1, horizontal_alignment='center', ),
                ObjectColumn(name = 'name',label = 'Name',width = 0.35,horizontal_alignment = 'left',editable=True),
                ObjectColumn(name = 'experiment_cnt',label = 'Experiments',horizontal_alignment = 'center',
                             width = 0.1,editable=False),

                ObjectColumn(name = 'comments',label = 'Comments',width = 0.45,
                             horizontal_alignment = 'center',editable=False),
                #ObjectColumn(name = 'em_pol',label = 'Emission POL',width = 0.08,horizontal_alignment = 'center'),



              ]
    orientation = 'vertical'

    auto_size = True
    sortable = False
    editable = False

