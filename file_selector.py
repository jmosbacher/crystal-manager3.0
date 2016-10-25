from traits.api import *
from traitsui.api import *
import os
from traitsui.tabular_adapter import TabularAdapter


class StringListAdapter(TabularAdapter):

    columns = [ ('', 'myvalue') ]

    myvalue_text = Property

    def _get_myvalue_text(self):
        return self.item

class FolderListAdapter(TabularAdapter):
    columns = [ ('Folders', 'myvalue') ]

    myvalue_text = Property

    def _get_myvalue_text(self):
        return self.item



class FileListAdapter(TabularAdapter):
    columns = [('Files', 'myvalue')]

    myvalue_text = Property
    selected_bg_color = Property

    def _get_selected_bg_color(self):
        return self.default_bg_color

    def _get_myvalue_text(self):
        return self.item

file_editor = TabularEditor(
    show_titles=True,
    operations = [],
    editable=False,
    multi_select=True,
    adapter=FileListAdapter())


folder_editor = TabularEditor(
    show_titles=True,

    editable=False,
    multi_select=True,
    adapter=FolderListAdapter())

string_list_editor = TabularEditor(
    show_titles=True,

    editable=False,
    multi_select=True,
    adapter=StringListAdapter())


class SelectionToolBase(HasTraits):
    file_names = List([])
    filtered_file_names = List([])
    selected_files = List([])

    folder_names = List([])
    filtered_folder_names = List([])
    selected_folders = List([])
    show = Enum(['Files','Folders'],cols=2)

    filtered_file_cnt = Property(Int,depends_on='filtered_file_names')
    filtered_folder_cnt = Property(Int,depends_on='filtered_folder_names')
    #show_folders = Bool(True)
    #show_files = Bool(True)
    dir = Directory
    filter = Str

    view = View(HSplit(
        VGroup(
        Item(name='dir',show_label=False,style='simple'),
        Item(name='dir',show_label=False,style='custom'),
        ),
                VGroup( Item(name='show',show_label=False,style='custom'),
                    HGroup(Item(name='filter',label='Filter',springy=True),
                           Item(name='filtered_file_cnt', label='Count',
                                visible_when='show=="Files"',style='readonly' ),
                           Item(name='filtered_folder_cnt', label='Count',
                                visible_when='show=="Folders"',style='readonly'),
                           ),
                        UItem('filtered_folder_names', visible_when='show=="Folders"',
                        editor=folder_editor(selected='selected_folders') ),
                        UItem('filtered_file_names', visible_when='show=="Files"',
                              editor=file_editor(selected='selected_files')),
                        ),
                ),
    resizable=True,
    width=800,
    height=600,
    )

    def _filter_changed(self):
        self.filter_all()

    def _get_filtered_file_cnt(self):
        return len(self.filtered_file_names)

    def _get_filtered_folder_cnt(self):
        return len(self.filtered_folder_names)

    def _dir_changed(self):
        all_names = os.listdir(self.dir)
        self.file_names = []
        self.folder_names = []
        for name in all_names:
            path = os.path.join(self.dir,name)
            if os.path.isfile(path):
                self.file_names.append(name)
            elif os.path.isdir(path):
                self.folder_names.append(name)

        self.filter_all()

    def filter_names(self,names):
        filtered = []
        for name in names:
            if self.filter in name:
                filtered.append(name)
        return sorted(filtered)

    def filter_all(self):
        self.filtered_file_names = self.filter_names(self.file_names)
        self.filtered_folder_names = self.filter_names(self.folder_names)

class FileSelectionTool(SelectionToolBase):
    show_folders = False

class FolderSelectionTool(SelectionToolBase):
    show_files = False


if __name__ == '__main__':
    tool = SelectionToolBase()
    tool.configure_traits()