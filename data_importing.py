from __future__ import print_function
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
from file_selector import FileSelectionTool, FolderSelectionTool, SelectionToolBase
from viewers import OutputStream
from log_viewer import LogStream
from measurement import SpectrumMeasurement, BaseMeasurement
from auxilary_functions import color_map, wl_to_rgb
#from crystal import Crystal
from pyface.api import confirm, error, YES, CANCEL
from datetime import datetime
from threading import Thread
from time import sleep
import logging
try:
    import cPickle as pickle
except:
    import pickle


def read_ascii_file(path, file_del):
    logger = logging.getLogger(__name__)
    data = []
    sup = []
    with open(path, 'r') as f:
        sig = True
        for line in f:
            if line in ['\n', '\r\n']:
                sig = False
                continue
            if sig:
                data.append(np.fromstring(line,count=2, sep=file_del))
                #s = line.split(file_del)
                #data.append([eval(s[0]), eval(s[1])])

            else:
                sup.append(line)
    if len(data):
        return np.array(data), sup
    return None



def organize_data(in_data,tags=('sig','bgd','ref'), ext='.asc'):
    '''
    :param data: Dictionary {file name: data array}
    :param tags: text tags to organize by
    :param ext: text file name extension to ignore
    :return: Dictionarey {Name: {signal:array, bg:array, ref:array}
    '''
    logger = logging.getLogger(__name__)
    out = {}
    for name, data in in_data.items():
        for tag in tags:
            if tag in name:
                if name.replace('_' + tag,'').replace(ext, '') in out.keys():
                    out[name.replace('_' + tag,'').replace(ext, '')][tag] = data
                else:
                    out[name.replace('_' + tag, '').replace(ext, '')] = {tag:data}
    return out

def import_group(path, names,**kwargs):
    logger = logging.getLogger(__name__)
    delimiter = kwargs.get('delimiter', ' ')
    tool = kwargs.get('tool', None)

    dir_path = path
    data = {}
    for name in names:
        path = os.path.join(dir_path, name)
        if os.path.isfile(path):
            logger.info('Reading data from file at:\n %s' % path)
            result = read_ascii_file(path, delimiter)
            if result is not None:
                data[name] = result
            logger.info('File Read.')
            logger.info('Skipping (not a file): \n%s' % path)

    logger.info('Organizing Data.')
    organized = organize_data(data)
    logger.info('%d Files successfully imported.' %len(data.keys()))

    if tool is not None:
        tool.result = organized
        tool.done = True
    else:
        return organized

def import_folder(path, **kwargs):
    logger = logging.getLogger(__name__)
    delimiter = kwargs.get('delimiter',' ')
    log = kwargs.get('log',None)
    tool = kwargs.get('tool',None)
    result_name = kwargs.get('result_name',None)
    if not os.path.isdir(path):
        logger.info('Cannot find folder:\n%s.' % path)
        return False

    names = os.listdir(path)
    files = []

    logger.info('Searching for files in:\n%s.' %path)
    for name in names:
        if os.path.isfile(os.path.join(path,name)):
            files.append(name)

    logger.info('Files found: %d' %len(files))
    #org_data = import_group(path,files,delimiter=delimiter,log=log)
    dir_path = path
    data = {}
    for name in files:
        path = os.path.join(dir_path, name)
        if os.path.isfile(path):

            logger.info('Reading data from file at:\n %s' % path)
            result = read_ascii_file(path, delimiter)
            if result is not None:
                data[name] = result
                logger.info('File read succesfully.')
            else:
                logger.info('File import unsuccesfull.')

            logger.info('Skipping (not a file): \n%s' % path)

    logger.info('Organizing Data.')
    organized = organize_data(data)

    logger.info('Done. Files successfully imported: %d' % len(data.keys()))
    logger.info('Storing Data...')

    if tool is not None:
        #print('Tool is not None' , file=log)
        if result_name is None:
            #print('Result name is None', file=log)
            tool.result = organized
            tool.done = True
        else:
            #print('Result name is not None', file=log)
            tool.result[result_name] = organized
            tool.done = True
    else:
        return organized
    if log is not None:
        logger.info('Finished importing files.')
        sleep(0.5)
    return True


class AutoImportToolHandler(Handler):

    def import_data(self,info,group):
        org_data = info.object.import_data(group)
        if org_data is not None:
            if len(org_data.keys()):
                code = confirm(info.ui.control, 'Data imported successfully, continue importing? Press Cancel to discard data',
                               title="Import more data?", cancel=True)

                if code == CANCEL:
                    return
                else:
                    info.object.store_data(org_data)
                    if code == YES:
                        return
                    else:
                        info.ui.dispose()
        else:
            error(None, 'Data extraction unsuccessful', title='No Data')

    def object_import_all_changed(self,info):
        self.import_data(info,info.object.selector.filtered_names)

    def object_import_selected_changed(self,info):
        self.import_data(info,info.object.selector.selected)


class AutoImportToolBase(HasTraits):
    log = Instance(LogStream)
    clear_log = Button('Clear Log')
    experiment = Any() #Instance(Crystal)
    selector = Instance(SelectionToolBase)
    delimiter = Str(' ')
    data_folder = Str('ascii')
    #mode = Enum(['New Measurement', 'Append to Selected'])
    import_selected = Button(name='Import Selected',action='import_selected_fired')
    import_all = Button(name='Import All')
    folders = Bool(False)
    result = Dict()
    done = Bool(False)
    stored = Bool(False)
    expected_results = Int(0)
    threads = List([])

    view = View(
        HGroup(
        Group(
            Item(name='selector',show_label=False,style='custom'),
        show_border=True,label='Files to import'),
        HGroup(
            Item(name='delimiter', label='Delimiter', ),
            Item(name='data_folder', label='Data Folder',visible_when='folders==True' ),
            Item(name='import_selected', show_label=False, ),
            Item(name='import_all', show_label=False, ),
            spring,
            Item(name='clear_log', show_label=False, ),
        ),
        Group(Item(name='log', show_label=False, style='custom')),

        ),
        buttons=['OK'],
        kind='modal',
        handler=AutoImportToolHandler(),
    )


    def __init__(self):
        raise NotImplementedError

    def _log_default(self):
        log = LogStream()
        log.config_logger(name=__name__, level='DEBUG')
        return log

    def _selector_default(self):
        raise NotImplementedError

    def store_data(self,org_data):
        raise NotImplementedError

    def import_data(self, group):
        raise NotImplementedError

    def _clear_log_fired(self):
        self.log.text = ''

    def _done_changed(self):
        if self.done:
            if self.stored:
                return
            if self.expected_results and len(self.result.keys())!=self.expected_results:
                self.done = False
                return
            self.store_data(self.result)
            self.stored = True
            self.result = {}
            self.done = False
            #for thread in self.threads:
                #if thread.is_alive():
                    #thread.join(100)
            self.threads = []

    def store_meas(self,experiment,data,name):
        new = experiment.add_measurement()
        new.name = name
        try:
            ex_wl = eval(name.split('in')[0])
            if isinstance(ex_wl, (float, int)):
                new.ex_wl = ex_wl
        except:
            pass
        new.signal = data.get('sig', ([], []))[0]
        new.bg = data.get('bgd', ([], []))[0]
        new.ref = data.get('ref', ([], []))[0]
        new.file_data = {'sig':{}, 'bgd':{}, 'ref':{}}
        for file_type, storage in new.file_data.items():
            for line in data.get(file_type, ([], []))[1]:
                key, value = line.split(':',1)
                storage[key.strip()] = value.strip()

        for keyy, valuee in new.file_data['sig'].items():
            if 'Exposure Time (secs)' in keyy:
                t = eval(valuee)
                if isinstance(t, (float, int)):
                    new.exposure = t

            elif 'Number of Accumulations' in keyy:
                t = eval(valuee)
                if isinstance(t, int):
                    new.frames = t

            elif 'Date and Time' in keyy:
                try:
                    date_time = datetime.strptime(valuee, '%a %b %d %X %Y')
                    new.date = date_time.date()
                    new.time = date_time.time()
                except:
                    pass


class AutoSpectrumImportTool(AutoImportToolBase):
    def __init__(self, experiment):
        HasTraits.__init__(self)
        self.experiment = experiment
        #self.kind = 'Spectrum'

    def _selector_default(self):
        return FileSelectionTool()

    def import_data(self, group):
        #self.expected_results = len(group)
        self.stored = False
        thread = Thread(target=import_group,args=(self.selector.dir, group),
                        kwargs=dict(delimiter=self.delimiter, log=self.log, tool=self ))
        thread.daemon = True
        thread.start()
        self.threads.append(thread)
        #org_data = import_group(self.selector.dir,group,delimiter=self.delimiter,log=self.log)
        #return org_data

    def store_data(self, org_data):
        for name, data in org_data.items():
            self.store_meas(self.experiment,data,name)
            sleep(0.6)

class SpectrumImportToolTab(AutoSpectrumImportTool):

    view = View(
        HSplit(
            Item(name='selector',show_label=False,style='custom',width=0.65),
        VGroup(
            HGroup(
                Item(name='delimiter', label='Delimiter', ),
                Item(name='data_folder', label='Data Folder',visible_when='folders==True' ),
                ),
            HGroup(
                Item(name='import_selected', show_label=False, ),
                Item(name='import_all', show_label=False, ),
                spring,
                Item(name='clear_log', show_label=False, ),
                ),

            Group(Item(name='log',show_label=False,style='custom')),
            ),
        ),
        #buttons=['OK'],
        #kind='modal',

    )

    def _import_selected_fired(self):
        self.import_data(self.selector.selected)


    def _import_all_fired(self):
        self.import_data(self.selector.filtered_names)


class AutoExperimentImportTool(AutoImportToolBase):
    project = Any() #Instance(Crystal)

    def __init__(self, project):
        HasTraits.__init__(self)
        self.project = project
        self.folders = True

    def _selector_default(self):
        return FolderSelectionTool()

    def import_data(self, group):
        names = set(group)
        self.expected_results = len(names)
        self.result = {}
        self.stored = False
        threads = []
        for name in names:
            path = os.path.join(self.selector.dir,name,self.data_folder)
            thread = Thread(target=import_folder, args=(path,),
                            kwargs=dict(delimiter=self.delimiter, log=self.log, tool=self, result_name=name))
            thread.start()
            self.threads.append(thread)
            sleep(0.2)
            #all_data[name] = import_folder(path,delimiter=self.delimiter,log=self.log)

        #return all_data

    def store_data(self,all_data):
        for f_name, org_data in all_data.items():
            experiment = self.project.add_experiment()
            sleep(0.2)
            experiment.name = f_name
            experiment.crystal_name = '_'.join(f_name.split('_')[1:3])
            for name, data in org_data.items():
                self.store_meas(experiment, data, name)
            sleep(0.1)

            def wl(spectrum):
                return spectrum.ex_wl, spectrum.em_wl[0]
            experiment.measurements.sort(key=wl)
            sleep(0.1)

class ExpImportToolTab(AutoExperimentImportTool):

    view = View(
        HSplit(

            Item(name='selector',show_label=False,style='custom',width=0.65),

        VGroup(
            HGroup(
                Item(name='delimiter', label='Delimiter', ),
                Item(name='data_folder', label='Data Folder',visible_when='folders==True' ),
                ),
            HGroup(
                Item(name='import_selected', show_label=False, ),
                Item(name='import_all', show_label=False, ),
                spring,
                Item(name='clear_log', show_label=False, ),
                ),
            Group(Item(name='log',show_label=False,style='custom')),
            ),
        ),
        #buttons=['OK'],
        #kind='modal',

    )



    def _import_selected_fired(self):

        self.import_data(self.selector.selected)
        #self.store_data(org_data)

    def _import_all_fired(self):
        self.import_data(self.selector.filtered_names)
        #self.store_data(org_data)



