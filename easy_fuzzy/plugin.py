import os

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction

from .main_dock import EasyFuzzyDock


class EasyFuzzyPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.action = None
        self.dock = None

    def initGui(self):
        icon_path = os.path.join(os.path.dirname(__file__), "icon.png")

        self.action = QAction(
            QIcon(icon_path),
            "Easy Fuzzy",
            self.iface.mainWindow()
        )
        self.action.triggered.connect(self.run)

        self.iface.addPluginToMenu("&Easy Fuzzy", self.action)
        self.iface.addToolBarIcon(self.action)

    def unload(self):
        if self.action is not None:
            self.iface.removePluginMenu("&Easy Fuzzy", self.action)
            self.iface.removeToolBarIcon(self.action)

        if self.dock is not None:
            self.iface.removeDockWidget(self.dock)
            self.dock.deleteLater()
            self.dock = None

    def run(self):
        if self.dock is None:
            self.dock = EasyFuzzyDock(self.iface)
            self.iface.addDockWidget(Qt.RightDockWidgetArea, self.dock)

        self.dock.show()
        self.dock.raise_()