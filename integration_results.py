import os
from traits.api import *
from traitsui.api import *
from traitsui.ui_editors.array_view_editor import ArrayViewEditor
from pyface.api import FileDialog, confirm, error, YES, CANCEL
import numpy as np
from saving import BaseSaveHandler
from auxilary_functions import data_array_to_text_file
from export_tools import DataFrameExportTool
import pandas as pd


class IntegrationResultBaseHandler(BaseSaveHandler):
    #extension = Str('int')
    promptOnExit = False

    def object_save_data_changed(self, info):
        fileDialog = FileDialog(action='save as', title='Save As',
                                wildcard=self.wildcard,
                                parent=info.ui.control,
                                default_filename=info.object.name+'_integration')

        fileDialog.open()
        if fileDialog.path == '' or fileDialog.return_code == CANCEL:
            return False
        elif info.object.save_type=='Text':
            range_data='Integration Range: %d to %d nm' %(info.object.int_range[0],info.object.int_range[1])
            data_array_to_text_file(path=fileDialog.path,array=info.object.results,
                                    headers=info.object.headers,table_fmt=info.object.table_fmt,
                                    float_fmt=info.object.float_fmt,first_line=range_data)

        elif info.object.save_type=='Excel':
            df = info.object.create_dataframe()
            df.to_excel(fileDialog.path+'.xlsx',
                #float_format=info.object.float_fmt,
                sheet_name='%dto%dnm' %info.object.int_range,
                index_label='Excitation WL',
            )

        elif info.object.save_type == 'CSV':
            df = info.object.create_dataframe()
            df.to_csv(fileDialog.path,
                    #float_format=info.object.float_fmt,
                    index_label='Excitation WL',
                    )

class IntegrationResultBase(HasTraits):
    headers = ['Excitation WL', '1st', '2nd', '3rd']
    save_type = Enum(['Text','Excel','CSV'])
    name = Str('Results')
    int_range = Tuple((0.0,0.0),labels=['Min','Max'],cols=2,format='%g')
    results = Array()
    export = Button('Export Results')

    def create_dataframe(self):
        df = pd.DataFrame(
            data=self.results[:,1:],
            index=self.results[:,0],
            columns=self.headers[1:],
        )
        return df

    def _export_fired(self):
        df = self.create_dataframe()
        tool = DataFrameExportTool(df=df)
        tool.edit_traits()

class ComparisonIntegrationResult(IntegrationResultBase):
    headers = ['Excitation WL', 'Counts 1st', 'Counts 2nd', 'Counts Subtraction']
    #save_data = Button('Export Data') #Action(name = 'Export', action = 'save_data')

    #table_fmt = Enum(['plain', 'simple', 'grid', 'fancy_grid', 'pipe','orgtbl','rst','mediawiki','html', 'latex', 'latex_booktabs',])
    #fmt = Str('e')
    #ndec = Int(2)
    #float_fmt = Property(Str)


    view = View(
        VGroup(
            HGroup(Item(name='int_range', label='Integration Range', style='readonly',),
                   ),
            HGroup(
                #Item(name='save_type', show_label=False),
                Item(name='export',show_label=False),
                #Item(name='table_fmt', label='Table Format',enabled_when='save_type=="Text"'),
                #Item(name='fmt',editor=EnumEditor(values={'f':'Regular', 'e':'Exponential'}), label='Number Format'),
                #Item(name='ndec', label='Decimals'),
            ),

            Group(Item('results',
                 show_label=False,
                 editor=ArrayViewEditor(titles=['Excitation WL', '1st Exp. (BG corrected)', '2nd Exp. (BG corrected)', 'Subtraction (BG corrected)'],
                                        format='%.2e',
                                        show_index=False,
                                        # Font fails with wx in OSX;
                                        #   see traitsui issue #13:
                                        # font   = 'Arial 8'
                                        )),
                  show_border=True,label='Results'),

        ),


    #handler=IntegrationResultBaseHandler(),
    resizable=True,
    scrollable=True,
    )

   # def _get_float_fmt(self):
        #return '.{}{}'.format(self.ndec,self.fmt)

    def _results_default(self):
        return np.asarray([[0.0,0.0,0.0,0.0]]*4)

class ExperimentIntegrationResult(IntegrationResultBase):
    headers = ['Excitation WL', 'Signal (Counts)', 'BG (Counts)','Signal-BG', 'REF (Counts)']
    #save_data = Button('Export Data') #Action(name = 'Export Data', action = 'save_data')
    #table_fmt = Enum(['plain', 'simple', 'grid', 'fancy_grid', 'pipe','orgtbl','rst','mediawiki','html', 'latex', 'latex_booktabs'])
    #fmt = Str('e')
    #ndec = Int(2)
    #float_fmt = Property(Str)


    view = View(
        VGroup(
            HGroup(Item(name='int_range', label='Integration Range', style='readonly'),

                   ),
            HGroup(
                #Item(name='save_type', label='Save as'),
                #Item(name='table_fmt', label='Table Format', enabled_when='save_type=="Text"'),
                Item(name='export',show_label=False),
                    ),
                HGroup(

                #Item(name='fmt', editor=EnumEditor(values={'f':'Regular', 'e':'Exponential'}), label='Number Format'),
                #Item(name='ndec', label='Decimals'),

                    ),
            Group(Item('results',
                 editor=ArrayViewEditor(titles=['Excitation WL', 'Signal (Counts)', 'BG (Counts)','Signal-BG', 'REF (Counts)'],
                                        format='%.2e',
                                        show_index=False,
                                        # Font fails with wx in OSX;
                                        #   see traitsui issue #13:
                                        # font   = 'Arial 8'
                                        ),show_label=False),

                  show_border=True, label='Results'),


                ),
    #handler = IntegrationResultBaseHandler(),
    )

    #def _get_float_fmt(self):
        #return '.{}{}'.format(self.ndec,self.fmt)

    def _results_default(self):
        return np.asarray([[0.0,0.0,0.0,0.0]]*4)