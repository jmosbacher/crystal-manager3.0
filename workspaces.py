from traits.api import *
from traitsui.api import *
from project import Project


class WorkSpace(HasTraits):
    name = Str('Worksapce')
    main = Any()
    projects = List(Project)
    selected = Instance(Project)
    view = View()
    #Item('name')
    def __init__(self, main=None):
        self.main=main
        super(WorkSpace,self).__init__()

    def _projects_default(self):
        return [Project(main=self.main)]