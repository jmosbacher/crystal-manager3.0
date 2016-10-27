from __future__ import print_function
import os
from traits.api import *
from traitsui.api import *
import numpy as np
import logging
import threading
import Queue
from time import sleep
from file_selector import SelectionToolBase
from log_viewer import LogStream
import pandas as pd
from collections import Counter
from datetime import datetime
import sys
#from measurement import SpectrumMeasurement
#from experiment import SpectrumExperiment




class ImportSettings(HasTraits):
    delimiter = Str(' ')
    tags = Tuple(('sig','bgd','ref'),labels=['Signal','Background','Reference'])
    extension = Str('asc')
    use_parent_as_name = Bool(True)
    nworker = Int(4)
    debug = Bool(False)

    view = View(
        VGroup(

            HGroup(
                Group(Item(name='tags',show_label=False, width=40),
                      show_border=True,label='Data tags'),
                VGroup(
                   Item(name='extension', label='Extension',width=40),
                   Item(name='delimiter', label='Delimiter',width=40),
                    VGroup(
                        Item(name='nworker', label='Worker threads', width=-20),
                        Item(name='use_parent_as_name', label='In subdirectory'),
                        Item(name='debug', label='Debug'),
                        show_left=False),
                ),
            ),

    ))


class DataImporter(HasTraits):
    path_list = List([])
    nfiles = Int()
    settings = Instance(ImportSettings)
    destination = Any
    successful = List([])
    counter = Instance(Counter)
    #path_queue = Instance(Queue.Queue)
    #data_queue = Instance(Queue.Queue)
    #stop_event = Event()
    #pause_event = Event()

    def __init__(self):
        super(DataImporter,self).__init__()

    def _get_nfiles(self):
        return len(self.path_list)

    def start(self,import_ui):
        import_ui.user_wants_stop = False
        import_ui.user_wants_pause = False
        import_ui.user_wants_cancel = False
        import_ui.ndone = 0
        logger = logging.getLogger(__name__)
        self.settings = import_ui.settings
        self.path_list = import_ui.path_list[:]
        self.successful = []
        self.destination = import_ui.destination
        self.nfiles = len(self.path_list)
        self.path_queue = Queue.Queue()
        self.data_queue = Queue.Queue()
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.counter = Counter()
        self.working = True
        self.user_wants_pause = False
        self.user_wants_stop = False

        manager = threading.Thread(target=self.import_manager,
                                   name='Manager_Thread',
                                   args=(import_ui,),
                                   )
        manager.setDaemon(True)
        manager.start()

    def import_manager(self,import_ui):
        logger = logging.getLogger(__name__)
        logger.debug('Manager Started')


        for n in range(self.settings.nworker):
            t = threading.Thread(target=self.import_worker,
                                 name='Worker_Thread_%d' % n,)
                                 #args=(path_queue, data_queue, events),
                                 #kwargs=dict(delim=self.settings.delimiter)
            t.setDaemon(True)
            t.start()

        while len(self.path_list) and not self.stop_event.is_set():
            self.path_queue.put(self.path_list.pop())

        for n in range(self.settings.nworker):
            self.path_queue.put(None)

        results = []
        while import_ui.ndone < self.nfiles:
            sleep(0.01)
            if import_ui.user_wants_pause and not self.pause_event.is_set():
                self.pause_event.set()
                logger.info('User paused.')

            if not import_ui.user_wants_pause and self.pause_event.is_set():
                self.pause_event.clear()
                logger.info('User resumed.')

            if import_ui.user_wants_stop and not self.stop_event.is_set():
                self.stop_event.set()
                logger.info('User stopped. Saving results and closing threads.')
                break

            if import_ui.user_wants_cancel:
                self.stop_event.set()
                import_ui.ndone = 0
                import_ui.working = False
                logger.info('User canceld. Closing all threads.')
                return
            try:
                result = self.data_queue.get(timeout=0.1)
                if result is None:
                    logger.debug('Nonetype returned from Workers.')
                else:
                    results.append(result)
                    logger.debug('Result returned from Workers.')
                import_ui.ndone += 1

            except:
                pass
        logger.info('Finished importing %d files successfully out of %d paths.'
                    % (import_ui.ndone, self.nfiles))
        self.store_results_list(results)


        for path in self.successful:
            success = import_ui.remove_from_import_queue(path)
            if success:
                self.counter['removed'] +=1
        logger.info('Removed %d paths from import queue.'%self.counter['removed'])

        import_ui.working = False
        return



    def import_worker(self):
        logger = logging.getLogger(__name__)
        logger.debug('Worker started.')
        while True:
            sleep(0.05)
            try:
                path = self.path_queue.get(timeout=20)
            except:
                return None

            if path is None:
                logger.debug('Death pill swallowed. closing.')
                return

            if self.stop_event.is_set():
                logger.debug('User pressed stop. closing.')
                return None

            while self.pause_event.is_set():
                sleep(0.05)

            data, metadata = self.read_ascii_file(path, delim=self.settings.delimiter)
            metadata['path'] = path
            if data is not None:
                result = dict(data=data,
                              metadata=metadata)
                self.data_queue.put(result)
                logger.debug('Task done successfully.')
            else:
                self.data_queue.put(None)
                logger.debug('Task unsuccessful.')
            self.path_queue.task_done()


    def read_ascii_file(self,path, delim):
        logger = logging.getLogger(__name__)
        data = []
        metadata = {}
        with open(path, 'r') as f:
            sig = True
            for line in f:
                if line in ['\n', '\r\n']:
                    sig = False
                    continue
                if sig:
                    data.append(np.fromstring(line, count=2, sep=delim))
                else:
                    key, value = line.split(':', 1)
                    metadata[key.strip()] = value.strip()
        if data:
            return np.array(data), metadata
        else:
            return None, None

    def store_results_list(self, results_list):
        logger = logging.getLogger(__name__)
        results_dict = self.organize_results_by_folder(results_list)
        self.direct_measurements_to_experiments(results_dict)
        logger.info('Finished creating %d measurements and %d experiments for %d files.'
                    %(self.counter['nmeas'],
                      self.counter['nexp'],
                      len(results_list)))


    def organize_results_by_folder(self, results_list):
        logger = logging.getLogger(__name__)
        results_dict = {}
        tags = self.settings.tags
        ext = self.settings.extension

        for result in results_list:
            path = result['metadata']['path']
            folder, name = os.path.split(os.path.abspath(path))
            if name.split('.', 1)[-1] != ext:
                continue
            if folder not in results_dict.keys():
                results_dict[folder] = {}
            for tag in tags:
                if tag in name:
                    striped_name = name.replace('_' + tag, '').replace('.' + ext, '')
                    #logger.debug(striped_name)
                    if striped_name not in results_dict[folder].keys():
                        results_dict[folder][striped_name] = {}
                    results_dict[folder][striped_name][tag] = result
            self.successful.append(path)
        # logger.debug(str(results_dict))
        return results_dict

    def direct_measurements_to_experiments(self, folder_dict):
        logger = logging.getLogger(__name__)
        for folder_path, result_dict in folder_dict.items():
            if self.destination.__klass__ == 'Project':
                self.create_experiment_from_data(project=self.destination,
                                                 folder_path=folder_path,
                                                 result_dict=result_dict)
            else:
                self.create_measurements_from_data(self.destination, result_dict)

    def create_experiment_from_data(self,project,folder_path,result_dict):
        logger = logging.getLogger(__name__)
        parent_path, folder_name = os.path.split(folder_path)
        if self.settings.use_parent_as_name:
            exp_name = os.path.split(parent_path)[-1]
        else:
            exp_name = folder_name
        try:
            new_exp = self.destination.add_experiment()
            new_exp.name = exp_name
            self.counter['nexp'] +=1
            self.create_measurements_from_data(new_exp, result_dict)
            logger.debug('%s experiment created.' % exp_name)
        except:
            logger.debug('Couldnt create experiment %s.'%exp_name)

    def create_measurements_from_data(self,experiment,result_dict):
        logger = logging.getLogger(__name__)
        logger.debug('Creating %d measurement objects for experiment %s'
                     %(len(result_dict),experiment.name))

        for meas_name, extracted_data in result_dict.items():
            success = False
            try:
                success = self.create_measurement_from_data(experiment,meas_name,extracted_data)
            except:
                logger.debug('Couldnt create measurment %s.' % meas_name)
            if success:
                self.counter['nmeas'] +=1

    def create_measurement_from_data(self,experiment,meas_name,extracted_data):
        logger = logging.getLogger(__name__)
        logger.debug('Creating %s measurement for %s experiment'
                     %(meas_name,experiment.name))
        new_meas = experiment.add_measurement()
        new_meas.name = meas_name
        try:
            new_meas.ex_wl = float(meas_name.split('in')[0])
        except:
            pass
        datasets = None
        new_meas.metadata = {}
        for tag, file_data in extracted_data.items():
            # logger.debug(tag)
            if file_data is None:
                continue
            # logger.debug(str(file_data['metadata']))
            new_meas.metadata[tag] = file_data['metadata']
            new = pd.DataFrame(data=file_data['data'], columns=['em_wl', tag])
            if datasets is None:
                datasets = new
            else:
                datasets = pd.merge_asof(datasets, new, on='em_wl')

        new_meas.data = datasets
        new_meas.file_data = new_meas.metadata
        new_meas.make_data_arrays()

        return True
        #logger.debug(datasets.head())



class ImportTab(HasTraits):
    destination = Any()
    importer = Instance(DataImporter)
    settings = Instance(ImportSettings)

    path_list = List([])
    ntodo = Property(Int)#Int(10)
    min = Int(0)
    ndone = Int(0)#Range('min','ntodo')
    done_percent = Range(0,100)
    debug = DelegatesTo('settings')

    start = Button('Start')
    stop = Button('Stop')
    pause = Button('Pause')
    resume = Button('Resume')
    cancel = Button('Cancel')
    #restart = Button('Restart')

    working = Bool(False)
    user_wants_stop = Bool(False)
    user_wants_cancel = Bool(False)
    user_wants_pause = Bool(False)
    log_view = Instance(LogStream)

    view = View(
        VGroup(

        Item(name='settings',show_label=False,style='custom'),

        HGroup(
        Item(name='start',show_label=False,springy=True,
             enabled_when='len(import_queue)',visible_when='not working'),
        Item(name='stop', show_label=False, visible_when='working',springy=True),
        Item(name='pause', show_label=False, visible_when='working and not user_wants_pause',springy=True),
        Item(name='resume', show_label=False, visible_when='user_wants_pause',springy=True),
        Item(name='cancel', show_label=False, visible_when='working',springy=True),
             ),
        VGroup(
            Group(Item(name='done_percent',style='simple',enabled_when='False',
              show_label=False ),
                  show_border=True,label='Percent Finished'),

        Item(name='log_view', show_label=False, style='custom'),
            ),
            ),
        #scrollable=False,
        resizable=True,
    )

    def __init__(self,destination):
        super(ImportTab,self).__init__()
        self.destination = destination


    def _get_ntodo(self):
        return len(self.path_list)

    def _log_view_default(self):
        log_view = LogStream()
        log_view.max_len = 50
        log_view.config_logger(name=__name__,level='INFO')
        return log_view

    def _debug_changed(self):
        if self.debug:
            self.log_view.config_logger(name=__name__,level='DEBUG')
        else:
            self.log_view.config_logger(name=__name__, level='INFO')

    def _importer_default(self):
        return DataImporter()

    def _settings_default(self):
        return ImportSettings()

    def _ndone_changed(self):
        if self.ntodo:
            self.done_percent = int(100*self.ndone/self.ntodo)
        else:
            return 0

    def _start_fired(self):
        logger = logging.getLogger(__name__)
        logger.debug('Start pressed.')
        self.importer.start(self)
        self.working = True

    def _stop_fired(self):
        logger = logging.getLogger(__name__)
        logger.debug('Stop pressed.')
        self.user_wants_stop = True
        self.user_wants_pause = False

    def _cancel_fired(self):
        logger = logging.getLogger(__name__)
        logger.debug('Stop pressed.')
        self.user_wants_cancel = True
        self.user_wants_pause = False

    def _pause_fired(self):
        logger = logging.getLogger(__name__)
        logger.debug('Pause pressed.')
        self.user_wants_pause = True

    def _resume_fired(self):
        logger = logging.getLogger(__name__)
        logger.debug('Resume pressed.')
        self.user_wants_pause = False


    def extend_import_queue(self, paths):
        logger = logging.getLogger(__name__)
        for path in paths:
            self.path_list.append(path)
            logger.debug('%s added to import queue.' % path)
        logger.info('Added %d paths to import queue.' % len(paths))

    def remove_from_import_queue(self, path):
        logger = logging.getLogger(__name__)
        if path in self.path_list:
            self.path_list.remove(path)
            logger.debug('%s removed from import queue.' % path)
            return True
        else:
            logger.debug('Cannot remove %s from import queue. Path not in queue.' % path)
            return False

    def clear_import_queue(self):
        logger = logging.getLogger(__name__)
        self.path_list = []
        logger.info('Import queue cleared.')


    def list_import_queue(self):

        for n, path in enumerate(self.path_list):
            print('[%d]  %s' % (n, path), file=self.log_view)
        print('Total files in queue: %s' % len(self.path_list), file=self.log_view)


class ImportToolBase(HasTraits):
    parent = Any
    selector = Instance(SelectionToolBase)
    import_tab = Instance(ImportTab)

    add_selected = Button('Add Selected')
    add_all = Button('Add All')

    subfolder = Str('ascii')
    use_subfolder = Bool(True)

    list_paths = Button('Show')
    clear_paths = Button('Clear')
    rem_by_ind = Button('Remove')
    index = Int()

    show = DelegatesTo('selector')
    view = View(
        HGroup(
        Item(name='selector',show_label=False,style='custom',width=0.6),
        VGroup(
            VGroup(
            HGroup(
                Item(name='add_selected', show_label=False),
                Item(name='add_all', show_label=False),
                Item(name='use_subfolder',label='Subfolder',visible_when='show=="Folders"'),
                Item(name='subfolder', show_label=False, visible_when='show=="Folders"',
                     enabled_when='use_subfolder',width=-70),
                ),

            HGroup(
                Item(name='list_paths', show_label=False, springy=True, ),
                Item(name='clear_paths', show_label=False, springy=True, ),
                Item(name='index', show_label=False, width=-40),
                Item(name='rem_by_ind', show_label=False, springy=True, ),
                ),
                show_border=True, label='Import Queue'),
            UItem(name='import_tab',style='custom')
             ),

    ),
    resizable=True,
    )

    def __init__(self,parent):
        super(ImportToolBase, self).__init__()
        self.parent = parent

    def parent_changed(self):
        self.import_tab.destination = self.parent

    def _add_selected_fired(self):
        if self.selector.show == 'Folders':
            self.add_paths_from_folder_names(self.selector.dir,
                                             self.selector.selected_folders)
        elif self.selector.show == 'Files':
            self.add_paths_from_folder(self.selector.dir,self.selector.selected_files)

    def _add_all_fired(self):
        if self.selector.show == 'Folders':
            self.add_paths_from_folder_names(self.selector.dir,
                                         self.selector.filtered_folder_names)
        elif self.selector.show == 'Files':
            self.add_paths_from_folder(self.selector.dir,
                                       self.selector.filtered_file_names)

    def _selector_default(self):
        return SelectionToolBase()

    def _import_tab_default(self):
        return ImportTab(destination=self.parent)


    def _rem_by_ind_fired(self):
        if len(self.import_tab.path_list) > self.index:
            self.import_tab.remove_from_import_queue(self.import_tab.path_list[self.index])


    def _list_paths_fired(self):
        self.import_tab.list_import_queue()


    def _clear_paths_fired(self):
        self.import_tab.clear_import_queue()


    def add_paths_from_folder_names(self,dir,folder_names):

        for selected_folder in folder_names:
            folder_path = os.path.join(dir, selected_folder)
            if self.use_subfolder:
                folder_path = os.path.join(folder_path,self.subfolder)
            names = os.listdir(folder_path)
            self.add_paths_from_folder(folder_path, names)

    def add_paths_from_folder(self,folder_path,names):
        file_paths = []
        for name in names:
            path = os.path.join(folder_path, name)
            if os.path.isfile(path):
                file_paths.append(path)
        self.import_tab.extend_import_queue(file_paths)

if __name__ == '__main__':
    from experiment import SpectrumExperiment
    from project import Project
    experiment = SpectrumExperiment()
    project = Project()

    dialog = ImportToolBase(project)

    #fldr1 = '/home/joe/Dropbox/Color Centers/CC_measurments_140816/CalciumFluorite_IRR_CAF5_1_110816_VIS/ascii'
    #fldr2 = '/home/joe/Dropbox/Color Centers/CC_measurments_140816/CalciumFluorite_CAF5_1_110816_VIS/ascii'

#    dialog.path_list = [os.path.join(fldr1, name ) for name in os.listdir(fldr1)]
 #   dialog.path_list.extend([os.path.join(fldr2, name ) for name in os.listdir(fldr2)])
    dialog.configure_traits()
