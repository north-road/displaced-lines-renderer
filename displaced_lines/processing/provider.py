# -*- coding: utf-8 -*-

"""
Displaced Lines processing provider
"""

# .. note:: This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

__author__ = '(C) 2021 by Nyall Dawson'
__date__ = '20/11/2021'
__copyright__ = 'Copyright 2021, North Road'
# This will get replaced with a git SHA1 when you do a git archive
__revision__ = '$Format:%H$'

from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (QgsProcessingProvider)

from .displace_lines import DisplaceLinesAlgorithm
from ..gui.gui_utils import GuiUtils


class DisplacedLinesProvider(QgsProcessingProvider):
    """
    Processing provider for displaced lines tools
    """

    # pylint: disable=missing-docstring
    def loadAlgorithms(self):
        for a in [DisplaceLinesAlgorithm]:
            self.addAlgorithm(a())

    def name(self):
        return self.tr('Displaced Lines')

    def id(self):
        return 'displacedlines'

    def versionInfo(self):
        return "0.0.1"

    # pylint: enable=missing-docstring

    def tr(self, string, context=''):
        """
        Translates a string
        """
        if context == '':
            context = 'DisplacedLines'
        return QCoreApplication.translate(context, string)
