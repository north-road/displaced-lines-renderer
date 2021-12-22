# -*- coding: utf-8 -*-

"""
Displace lines algorithm
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

import math
from collections import defaultdict

from qgis.PyQt.QtCore import QCoreApplication, QVariant
from qgis.core import (
    QgsField,
    QgsFields,
    QgsFeature,
    QgsGeometry,
    QgsWkbTypes,
    QgsProcessingAlgorithm,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterNumber,
    QgsProcessingParameterDistance,
    QgsProcessing,
    QgsProcessingException,
    QgsLineString,
    QgsSpatialIndex,
    QgsRectangle,
    QgsPointXY,
    QgsPoint,
    QgsGeometryUtils
)

from ..gui.gui_utils import GuiUtils


class DisplaceLinesAlgorithm(QgsProcessingAlgorithm):
    """
    Displace lines algorithm
    """

    INPUT = 'INPUT'
    TOLERANCE = 'TOLERANCE'
    ANGLE_THRESHOLD = 'ANGLE_THRESHOLD'
    DISPLACEMENT_DISTANCE = 'DISPLACEMENT_DISTANCE'
    OUTPUT = 'OUTPUT'

    def tr(self, string, context=''):
        """
        Translates a string
        """
        if context == '':
            context = 'DisplacedLines'
        return QCoreApplication.translate(context, string)

    # pylint: disable=missing-docstring

    def createInstance(self):
        return DisplaceLinesAlgorithm()

    def initAlgorithm(self, config=None):  # pylint: disable=unused-argument
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.INPUT,
                self.tr('Input layer'),
                [QgsProcessing.TypeVectorLine]
            )
        )

        self.addParameter(
            QgsProcessingParameterDistance(
                self.TOLERANCE,
                self.tr('Tolerance'),
                defaultValue=1200,
                minValue=0,
                parentParameterName=self.INPUT
            )
        )

        param = QgsProcessingParameterNumber(
            self.ANGLE_THRESHOLD,
            self.tr('Angle threshold (degrees)'),
            QgsProcessingParameterNumber.Double,
            defaultValue=10,
            minValue=0,
            maxValue=90
        )
        param.setMetadata({'widget_wrapper': {'decimals': 2}})
        self.addParameter(param)

        self.addParameter(
            QgsProcessingParameterDistance(
                self.DISPLACEMENT_DISTANCE,
                self.tr('Displacement distance'),
                defaultValue=2000,
                minValue=0,
                parentParameterName=self.INPUT
            )
        )

        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT,
                self.tr('Output layer'),
                type=QgsProcessing.TypeVectorLine
            )
        )

    def name(self):
        return 'displacelines'

    def displayName(self):
        return self.tr('Displace lines')

    def shortHelpString(self):
        return self.tr('Displaces overlapping lines')

    def shortDescription(self):
        return self.tr('Displaces overlapping lines')

    def processAlgorithm(self, parameters, context, feedback):

        source = self.parameterAsSource(
            parameters,
            self.INPUT,
            context
        )

        if source is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.INPUT))

        tolerance = self.parameterAsDouble(parameters, self.TOLERANCE, context)
        angle_threshold = self.parameterAsDouble(parameters, self.ANGLE_THRESHOLD, context)
        displacement_distance = self.parameterAsDouble(parameters, self.DISPLACEMENT_DISTANCE, context)

        segments = {}
        segment_id = 0

        # step 1 -- build up a spatial index of all line segments
        feedback.pushInfo('Building spatial index')
        segment_index = QgsSpatialIndex()

        step = 100 / source.featureCount() if source.featureCount() else 0
        for idx, f in enumerate(source.getFeatures()):
            feedback.setProgress(idx * step)
            if feedback.isCanceled():
                break

            if not f.hasGeometry():
                continue

            input_line = f.geometry()
            for part in input_line.constParts():
                if not isinstance(part, QgsLineString):
                    part = part.segmentize()

                assert isinstance(part, QgsLineString)

                num_points = part.numPoints()
                if num_points < 2:
                    continue

                p2 = part[0]
                for n in range(1, num_points):
                    if feedback.isCanceled():
                        break

                    p1 = p2
                    p2 = part[n]

                    segment_rect = QgsRectangle(QgsPointXY(p1.x(), p1.y()), QgsPointXY(p2.x(), p2.y()))
                    segments[segment_id] = (p1, p2, segment_rect, f.id(), n-1)
                    segment_index.insertFeature(segment_id, segment_rect)
                    segment_id += 1

        step = 100 / len(segments) if segments else 0
        idx = 0

        segment_group_index = QgsSpatialIndex()
        segment_group_index_id = 0
        segment_groups = {}

        split_segments = {}
        split_segment_id = 0

        feedback.pushInfo('Determining adjacent line segments')
        for _id, (p1, p2, segment_rect, fid, geometry_idx) in segments.items():
            feedback.setProgress(idx * step)
            idx += 1
            if feedback.isCanceled():
                break

            #feedback.pushInfo(f'{_id}')

            feature_line_angle = QgsGeometryUtils.lineAngle(p1.x(), p1.y(), p2.x(), p2.y())
            # feedback.pushInfo(f'  line angle {math.degrees(feature_line_angle)}')

            split_points = set()

            search_rect = segment_rect.buffered(tolerance)
            candidate_segments = segment_index.intersects(search_rect)
            out_segments = set()
            for candidate in candidate_segments:
                if candidate == _id:
                    continue

                candidate_p1, candidate_p2, _, candidate_fid, _ = segments[candidate]

                # actually ignoring all segments from same source feature!
                if fid == candidate_fid:
                    continue


                # ignore touching segments from same source feature
                if fid == candidate_fid and (
                        candidate_p1 == p1 or candidate_p1 == p2 or candidate_p2 == p1 or candidate_p2 == p2):
                    continue

                # feedback.pushInfo(f' testing candidate {candidate}')

                candidate_line_angle = QgsGeometryUtils.lineAngle(candidate_p1.x(), candidate_p1.y(), candidate_p2.x(),
                                                                  candidate_p2.y())
                # feedback.pushInfo(f'  candidate angle {math.degrees(candidate_line_angle)}')

                delta_angle = min(abs(candidate_line_angle - feature_line_angle),
                                  abs(2 * math.pi + candidate_line_angle - feature_line_angle),
                                  abs(candidate_line_angle - (2 * math.pi + feature_line_angle)),

                                  abs(math.pi + candidate_line_angle - feature_line_angle),
                                  abs(candidate_line_angle - (math.pi + feature_line_angle)),
                                  )

                # feedback.pushInfo(f'  delta angle {math.degrees(delta_angle)}')

                if math.degrees(delta_angle) > angle_threshold:
                    continue

                # next step -- find overlapping portion of segment
                # take the start/end of the candidate, and find the nearest point on segment to those
                dist1, pt_x1, pt_y1 = QgsGeometryUtils.sqrDistToLine(candidate_p1.x(), candidate_p1.y(), p1.x(), p1.y(),
                                                                     p2.x(), p2.y(), 0)
                dist2, pt_x2, pt_y2 = QgsGeometryUtils.sqrDistToLine(candidate_p2.x(), candidate_p2.y(), p1.x(), p1.y(),
                                                                     p2.x(), p2.y(), 0)

                if pt_x1 == pt_x2 and pt_y1 == pt_y2:
                    # lines are parallel which come close, but don't actually overlap at all
                    # i.e. ------ ----------
                    continue

                dist3, _, _ = QgsGeometryUtils.sqrDistToLine(p1.x(), p1.y(), candidate_p1.x(), candidate_p1.y(),
                                                             candidate_p2.x(), candidate_p2.y(), 0)
                dist4, _, _ = QgsGeometryUtils.sqrDistToLine(p2.x(), p2.y(), candidate_p1.x(), candidate_p1.y(),
                                                             candidate_p2.x(), candidate_p2.y(), 0)

                min_distance = math.sqrt(min(dist1, dist2, dist3, dist4))
                if min_distance > tolerance:
                    # lines are parallel, but minimum distance between them is larger than the tolerance
                    continue

                # calculate split points - the distance along the segment at which the overlap starts/ends
                split_1 = math.sqrt((p1.x() - pt_x1)**2 + (p1.y() - pt_y1)**2)
                split_points.add(split_1)
                split_2 = math.sqrt((p1.x() - pt_x2) ** 2 + (p1.y() - pt_y2) ** 2)
                split_points.add(split_2)

                # feedback.pushInfo(f' found candidate {candidate}')
                out_segments.add(candidate)

            split_points.discard(0)
            split_points.discard(math.sqrt((p1.x() - p2.x())**2 + (p1.y() - p2.y())**2))

            if split_points:
                #feedback.pushInfo(f' split points {split_points}')
                split_points = [p1] + [QgsGeometryUtils.pointOnLineWithDistance(p1, p2, d) for d in sorted(list(split_points))] + [p2]
                for i in range(len(split_points)-1):
                    pp1 = split_points[i]
                    pp2 = split_points[i+1]

                    centroid = QgsPointXY(0.5*(pp1.x() + pp2.x()), 0.5*(pp1.y() + pp2.y()))
                    associated_group = segment_group_index.nearestNeighbor(centroid, maxDistance=tolerance)
                    if associated_group:
                        # assigning to existing group
                        #assert len(associated_group) == 1
                        split_segments[split_segment_id] = (pp1, pp2, associated_group[0], len(segment_groups[associated_group[0]]), fid, geometry_idx, i)

                        segment_groups[associated_group[0]].append(split_segment_id)
                    else:
                        # making a new group
                        split_segments[split_segment_id] = (pp1, pp2, segment_group_index_id, 0, fid, geometry_idx, i)
                        segment_groups[segment_group_index_id] = [split_segment_id]
                        segment_group_index.addFeature(segment_group_index_id, QgsRectangle(centroid, centroid).buffered(0.0001))
                        segment_group_index_id += 1

                    split_segment_id += 1

            else:
                centroid = QgsPointXY(0.5*(p1.x() + p2.x()), 0.5*(p1.y() + p2.y()))
                associated_group = segment_group_index.nearestNeighbor(centroid, maxDistance=tolerance) if out_segments else None

                # TODO -- maybe if no out_segments we should store this elsewhere, since it will always be unchanged


                if associated_group:
                    # assigning to existing group
                    #assert len(associated_group) == 1
                    split_segments[split_segment_id] = (p1, p2, associated_group[0], len(segment_groups[associated_group[0]]), fid, geometry_idx, 0)

                    segment_groups[associated_group[0]].append(split_segment_id)
                else:
                    # making a new group
                    split_segments[split_segment_id] = (p1, p2, segment_group_index_id, 0, fid, geometry_idx,  0)
                    segment_groups[segment_group_index_id] = [split_segment_id]
                    segment_group_index.addFeature(segment_group_index_id, QgsRectangle(centroid, centroid).buffered(0.0001))
                    segment_group_index_id += 1

                split_segment_id += 1

        feedback.pushInfo('Generating geometries for displaced segment groups')

        processed_groups = {}
        # process segment groups to generate displaced segments
        for segment_group_index_id, segment_group in segment_groups.items():

            if len(segment_group) == 1:
                processed_groups[segment_group_index_id] = [split_segments[segment_group[0]]]

            else:
                p1 = split_segments[segment_group[0]][0]
                p2 = split_segments[segment_group[0]][1]

                # flip segment directions if needed so that all segments in group are the same direction
                aligned_directions = [(p1,p2)]
                was_reversed = [False]
                for i in range(1, len(segment_group)):
                    segment = split_segments[segment_group[i]]
                    if segment[0].distance(p1) < segment[0].distance(p2):
                        aligned_directions.append((segment[0], segment[1]))
                        was_reversed.append(False)
                    else:
                        aligned_directions.append((segment[1], segment[0]))
                        was_reversed.append(True)

                average_start_point_x = sum(p[0].x() for p in aligned_directions) / len(segment_group)
                average_start_point_y = sum(p[0].y() for p in aligned_directions) / len(segment_group)
                average_end_point_x = sum(p[1].x() for p in aligned_directions) / len(segment_group)
                average_end_point_y = sum(p[1].y() for p in aligned_directions) / len(segment_group)

                start_point = QgsPoint(average_start_point_x, average_start_point_y)
                end_point = QgsPoint(average_end_point_x, average_end_point_y)

                average_line_azimuth = start_point.azimuth(end_point)

                processed_group = []
                overall_width = displacement_distance* (len(segment_group)-1)
                current_offset = overall_width*-0.5
                for idx, _ in enumerate(segment_group):
                    if was_reversed[idx]:
                        processed_group.append((end_point.project(current_offset, average_line_azimuth+90),
                                            start_point.project(current_offset, average_line_azimuth+90)))
                    else:
                        processed_group.append((start_point.project(current_offset, average_line_azimuth+90),
                                            end_point.project(current_offset, average_line_azimuth+90)))

                    current_offset+=displacement_distance
                processed_groups[segment_group_index_id] = processed_group

        feedback.pushInfo('Concatenating linestrings')

        # concatenate segments back to linestrings
        displaced_segments = defaultdict(list)
        for _id, (_, _, segment_group, index_in_group, fid, index_in_geometry, index_in_split) in split_segments.items():
            out_segment = processed_groups[segment_group][index_in_group]
            displaced_segments[fid].append((index_in_geometry, index_in_split, out_segment))

        merged_linestrings = {}
        for fid, segments in displaced_segments.items():
            sorted_segments = sorted(segments, key=lambda x:[x[0], x[1]])

            out_points = []

            for _, _, segment in sorted_segments:
                if not out_points:
                    out_points.append(segment[0])
                    out_points.append(segment[1])
                else:
                    if out_points[-1] != segment[0]:
                        out_points.append(segment[0])
                    out_points.append(segment[1])

            merged_linestrings[fid] = QgsGeometry(QgsLineString(out_points))

        feedback.pushInfo('Writing features')

        fields = QgsFields()
        fields.append(QgsField('id', QVariant.Int))
        fields.append(QgsField('group', QVariant.Int))

        (sink, dest_id) = self.parameterAsSink(
            parameters,
            self.OUTPUT,
            context,
            source.fields(),
            QgsWkbTypes.LineString,
            source.sourceCrs()
        )

        for f in source.getFeatures():
            f.setGeometry(merged_linestrings[f.id()])
            sink.addFeature(f)

        return {
            self.OUTPUT: dest_id
        }

    # pylint: enable=missing-docstring
