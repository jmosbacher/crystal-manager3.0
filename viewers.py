import os
from traits.api import *
from traitsui.api import *
from traitsui.extras.checkbox_column import CheckboxColumn
import matplotlib
matplotlib.use('Qt4Agg')
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from traitsui.ui_editors.array_view_editor import ArrayViewEditor
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.colors import colorConverter
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import numpy as np
import random
import pandas as pd

class DFrameViewer(HasTraits):
    df = Instance(pd.DataFrame)
    array_view = Instance(HasTraits)

    view = View(Item(name='array_view',show_label=False,style='custom'))
    def _df_default(self):
        return pd.DataFrame(data=[[0,0,0,0]],columns=['s','t','u','v'])

    def _df_changed(self,old,new):
        ArrawViewClass = type('ArrawViewClass', (HasTraits,),
                              {
                                  'array': Array(),
                                  'view': View(
                                      HGroup(Item(name='array', show_label=False, style='custom', springy=True,
                                                  editor=ArrayViewEditor(
                                                      titles=[str(x) for x in new.columns.values],
                                                      format='%g',
                                                      show_index=False, )),
                                             scrollable=True),
                                      scrollable=True,
                                      resizable=True)
                              })

        array_view = ArrawViewClass()
        array_view.array = new.as_matrix()
        self.array_view = array_view


class DictEditor(HasTraits):

    Object = Instance( object )
    def __init__(self, obj, **traits):
        super(DictEditor, self).__init__(**traits)
        self.Object = obj

    def trait_view(self, name=None, view_elements=None):
        return View(
          VGroup(
            Item('Object',
                  label      = 'Debug',
                  id         = 'debug',
                  editor     = ValueEditor(), #ValueEditor()
                  style      = 'custom',
                  dock       = 'horizontal',
                  show_label = False),),
          title     = 'Dictionary Editor',
          width     = 800,
          height    = 600,
          resizable = True)


class _MPLFigureEditor(Editor):

   scrollable  = True

   def init(self, parent):
       self.control = self._create_canvas(parent)
       self.set_tooltip()

   def update_editor(self):
       pass

   def _create_canvas(self, parent):
       """ Create the MPL canvas. """
       # matplotlib commands to create a canvas
       mpl_canvas = FigureCanvas(self.value)
       return mpl_canvas

class MPLFigureEditor(BasicEditorFactory):

   klass = _MPLFigureEditor


#!/usr/bin/env python
# Copyright (C) 2011-2014 Swift Navigation Inc.
# Contact: Fergus Noble <fergus@swift-nav.com>
#
# This source is subject to the license found in the file 'LICENSE' which must
# be be distributed together with this source. All other rights reserved.
#
# THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
# EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.

"""Contains the class OutputStream, a HasTraits file-like text buffer."""

from traits.api import HasTraits, Str, Bool, Trait, Int, Instance, Button
from traitsui.api import View, UItem, TextEditor, Handler,CodeEditor, HTMLEditor, Group, Item, StatusItem,\
    ProgressEditor
from traits.etsconfig.api import ETSConfig
from pyface.api import GUI, ProgressDialog


DEFAULT_MAX_LEN = 80000


class _OutputStreamViewHandler(Handler):

    def object_text_changed(self, uiinfo):
        ui = uiinfo.ui
        if ui is None:
            return

        for ed in  ui._editors:
            if ed.name == 'text':
                break
        else:
            # Did not find an editor with the name 'text'.
            return

        if ETSConfig.toolkit == 'wx':
            # With wx, the control is a TextCtrl instance.
            ed.control.SetInsertionPointEnd()
        elif ETSConfig.toolkit == 'qt4':
            # With qt4, the control is a QtGui.QTextEdit instance.
            from pyface.qt.QtGui import QTextCursor
            ed.control.moveCursor(QTextCursor.End)


class OutputStream(HasTraits):
    """This class has methods to emulate an file-like output string buffer.
    It has a default View that shows a multiline TextEditor.  The view will
    automatically move to the end of the displayed text when data is written
    using the write() method.
    The `max_len` attribute specifies the maximum number of bytes saved by
    the object.  `max_len` may be set to None.
    The `paused` attribute is a bool; when True, text written to the
    OutputStream is saved in a separate buffer, and the display (if there is
    one) does not update.  When `paused` returns is set to False, the data is
    copied from the paused buffer to the main text string.
    """

    # The text that has been written with the 'write' method.
    text = Str

    # The maximum allowed length of self.text (and self._paused_buffer).
    max_len = Trait(DEFAULT_MAX_LEN, None, Int)

    # When True, the 'write' method appends its value to self._paused_buffer
    # instead of self.text.  When the value changes from True to False,
    # self._paused_buffer is copied back to self.text.
    paused = Bool(False)

    # String that holds text written while self.paused is True.
    _paused_buffer = Str

    #make a log file too
    printToFile = Bool(False)

    def write(self, s):
        if self.paused:
            self._paused_buffer = self._truncated_concat(self._paused_buffer, s)
        else:
            self.text = self._truncated_concat(self.text, s)
            if self.printToFile:
                f = open(self.outFile ,'a+')
                f.write(s)
                f.close()

    def flush(self):
        GUI.process_events()

    def close(self):
        pass

    def reset(self):
        self._paused_buffer = ''
        self.paused = False
        self.text = ''

    def _truncated_concat(self, text, s):
        if len(s) >= self.max_len:
            # s could be huge. Handle this case separately, to avoid the temporary
            # created by 'text + s'.
            result = s[-self.max_len:]
        else:
            result = (text + s)[-self.max_len:]
        return result

    def _paused_changed(self):
        if self.paused:
            # Copy the current text to _paused_buffer.  While the OutputStream
            # is paused, the write() method will append its argument to _paused_buffer.
            self._paused_buffer = self.text
        else:
            # No longer paused, so copy the _paused_buffer to the displayed text, and
            # reset _paused_buffer.
            self.text = self._paused_buffer
            self._paused_buffer = ''

    def traits_view(self):
        view = \
            View(
                UItem('text', editor=TextEditor(multi_line=True), style='custom'),
                handler = _OutputStreamViewHandler(),

            scrollable = True)
        return view
    '''indpntview = View(
                    UItem('text', editor=TextEditor(multi_line=True), style='custom'),
                handler = _OutputStreamViewHandler(),
                resizable = True,
                height = 700,
                width = 600,
                buttons = ['OK'],
                title = 'Fit Results'
            )    '''

    def set_out_file(self, outFile,header=' '):
        self.outFile = outFile
        self.printToFile = True
        f = open(self.outFile ,'w')
        f.write('\n'+header+'\n' )
        f.close()





class ProgressViewer(HasTraits):
    text_stream = Instance(OutputStream)
    finished = Int(0)
    stop = Button('Pause')
    user_wants_stop = Bool(False)
    working = Bool(False)

    view = View(Group(Item(name='text_stream', show_label=False, style='custom'),
                      show_border=True,label='Messages'),
                UItem(name = 'finished',editor=ProgressEditor(message='Progress',min=0,max=100),show_label=False),

                resizable=True,
                title='Working',
                kind='livemodal',
                buttons = ['OK',stop]

                )

    def _text_stream_default(self):
        return OutputStream()

    def _stop_fired(self):
        if not self.user_wants_stop and self.working:
            self.user_wants_stop = True