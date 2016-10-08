from traits.api import *
from traitsui.api import *
import os
from traitsui.tabular_adapter import TabularAdapter


class StringListAdapter(TabularAdapter):

    columns = [ ('', 'myvalue') ]

    myvalue_text = Property

    def _get_myvalue_text(self):
        return self.item


string_list_editor = TabularEditor(
    show_titles=True,

    editable=False,
    multi_select=True,
    adapter=StringListAdapter())


class SelectionToolBase(HasTraits):
    file_names = List
    filtered_names = List
    selected = List([])
    dir = Directory
    filter = Str

    view = View(HSplit(
        VGroup(
        Item(name='dir',show_label=False,style='simple'),
        Item(name='dir',show_label=False,style='custom'),
        ),
                VGroup( HGroup(Item(name='filter',label='Filter',springy=True),),
                        UItem('filtered_names',
                        editor=string_list_editor(selected='selected') ),
                        ),
                ),
    resizable=True,
    width=800,
    height=600,
    )

    def filter_names(self):
        result = []
        for name in self.file_names:
            if self.filter in name:
                result.append(name)
        return sorted(result)

    def _filter_changed(self):
        self.filtered_names = self.filter_names()

    def _dir_changed(self):
        all_names = os.listdir(self.dir)
        file_names = []
        for name in all_names:
            if os.path.isfile(os.path.join(self.dir,name)):
                file_names.append(name)
        self.file_names = file_names
        self.filtered_names = self.filter_names()

    def _file_names_default(self):
        return []


class FileSelectionTool(SelectionToolBase):

    def _dir_changed(self):
        all_names = os.listdir(self.dir)
        file_names = []
        for name in all_names:
            if os.path.isfile(os.path.join(self.dir,name)):
                file_names.append(name)
        self.file_names = file_names
        self.filtered_names = self.filter_names()


class FolderSelectionTool(SelectionToolBase):



    def _dir_changed(self):
        all_names = os.listdir(self.dir)
        file_names = []
        for name in all_names:
            if os.path.isdir(os.path.join(self.dir,name)):
                file_names.append(name)
        self.file_names = file_names
        self.filtered_names = self.filter_names()

