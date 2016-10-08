from __future__ import print_function
import os
from traits.api import *
from traitsui.api import *
from traitsui.ui_editors.array_view_editor import ArrayViewEditor
from pyface.api import FileDialog, confirm, error, YES, CANCEL
import numpy as np
from saving import BaseSaveHandler
from auxilary_functions import data_array_to_text_file
import pandas as pd
from tabulate import tabulate


class DataFrameExportTool(HasTraits):
    df = Instance(pd.DataFrame)
    header = Str()
    save_dest = Enum(['File','Clipboard'])
    export_type = Enum(['Table', 'Excel', 'CSV'])
    table_fmt = Enum(['plain', 'simple', 'grid', 'fancy_grid', 'pipe', 'orgtbl', 'rst', 'mediawiki', 'html', 'latex',
                      'latex_booktabs', ])
    fmt = Str('e')
    ndec = Int(2)
    float_fmt = Property(Str)
    include_idx = Bool(True)
    save_path = File()
    export = Button('Export')

    clip_sep = Str('    ')
    clip_excel = Bool(False)
    view = View(
        VGroup(
            Item(name='save_dest'),
            VGroup(
                Item(name='save_path',show_label=False),
                HGroup(Item(name='export_type',show_label=False),
                       Item(name='table_fmt', label='Table style',visible_when='export_type=="Table"'),
                       Item(name='header', label='Header', enabled_when='export_type!="CSV"'),
                show_border=True, label='File type'),



            show_border=True,label='Export to File',enabled_when='save_dest=="File"'),
            HGroup(
                Item(name='clip_sep', label='Seperator'),
                Item(name='clip_excel', label='Excel Friendly'),
            show_border=True, label='Export to Clipboard',enabled_when='save_dest=="Clipboard"'),

        HGroup(Item(name='fmt', label='Notation', editor=EnumEditor(values={'f': 'Regular', 'e': 'Scientific'}), ),
                Item(name='ndec', label='Decimals'),
               #Item(name='include_index', label='Index Column'),
                   show_border=True, label='Formatting'),
        HGroup(Item(name='export', show_label=False, springy=True),),
        ),
        title='Save Data to File',
        scrollable=True,
        resizable=True,
        height=500,
        width=600,
        buttons=['OK']

    )

    def __init__(self, df,**kwargs):
        super(DataFrameExportTool, self).__init__()
        self.df = df
        self.header = kwargs.get('header', ' ')

    def _get_float_fmt(self):
        if self.export_type=='Table' and self.save_dest=='File':
            return '.{}{}'.format(self.ndec, self.fmt)
        else:
            return '%.{}{}'.format(self.ndec, self.fmt)


    def _export_fired(self):
        if self.save_dest=='Clipboard':
            self.df.to_clipboard(excel=self.clip_excel,
                                 sep=self.clip_sep,
                                 float_format=self.float_fmt)
        else:
            if not self.save_path:
                return
            if self.export_type=='Table':
                with open(self.save_path, 'w') as f:
                    if self.header is not ' ':
                        print(self.header, file=f)
                    print(tabulate(self.df,headers='keys',
                                   tablefmt=self.table_fmt,
                                   floatfmt=self.float_fmt),file=f)

            elif self.export_type=='Excel':
                self.df.to_excel(self.save_path+'.xlsx' ,
                                float_format=self.float_fmt,
                                sheet_name=self.header,
                                index_label='Index',)

            elif self.export_type == 'CSV':
                self.df.to_csv(self.save_path + '.txt',
                                 float_format=self.float_fmt,
                                 index_label='Index', )
if __name__=='__main__':
    df = pd.DataFrame(np.random.random((100,2)),columns=['Test col A', 'Test col B'])
    tool = DataFrameExportTool(df)
    tool.configure_traits()