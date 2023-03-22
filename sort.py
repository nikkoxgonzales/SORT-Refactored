"""
SORT: Simple Online and Realtime Tracker
Based on the code by Alex Bewley alex@bewley.ai (Copyright (C) 2016-2020)
Refactored and updated by Nikko Gonzales

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

For more information and updates, visit the official repository:
https://github.com/nikkoxgonzales/SORT-Refactored
"""

import numpy as np
from filterpy.kalman import KalmanFilter


class KalmanBoxTracker:
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox):
        """
        Initializes a tracker using initial bounding box.

        Args:
            bbox (list or np.array): A list or numpy array containing the initial bounding box coordinates.
        """
        # Define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0],
                              [0, 1, 0, 0, 0, 1, 0],
                              [0, 0, 1, 0, 0, 0, 1],
                              [0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.eye(4, 7)

        self.kf.R[2:, 2:] *= 10
        self.kf.P[4:, 4:] *= 1000  # Give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = self.convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.

        Args:
            bbox (list or np.array): A list or numpy array containing the updated bounding box coordinates.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self.convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.

        Returns:
            np.array: The predicted bounding box.
        """
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] *= 0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.

        Returns:
            np.array: The current bounding box estimate.
        """
        return self.convert_x_to_bbox(self.kf.x)

    @staticmethod
    def convert_bbox_to_z(bbox):
        """
        Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
        [x,y,s,r] where x,y is the center of the box, s is the scale/area, and r is
        the aspect ratio.

        Args:
            bbox (list or np.array): A list or numpy array containing bounding box coordinates [x1, y1, x2, y2].

        Returns:
            np.array: A numpy array containing the converted bounding box values [x, y, s, r].
        """
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.
        y = bbox[1] + h / 2.
        s = w * h  # scale is just area
        r = w / float(h)
        return np.array([x, y, s, r]).reshape((4, 1))

    @staticmethod
    def convert_x_to_bbox(x, score=None):
        """
        Takes a bounding box in the center form [x,y,s,r] and returns it in the form
        [x1,y1,x2,y2] where x1,y1 is the top-left corner and x2,y2 is the bottom-right corner.

        Args:
            x (np.array): A numpy array containing bounding box values [x, y, s, r].
            score (float, optional): The score associated with the bounding box. Defaults to None.

        Returns:
            np.array: A numpy array containing the converted bounding box coordinates [x1, y1, x2, y2, (score)].
        """
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        if score is None:
            return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
        else:
            return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))


class Sort:
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Initializes the Sort object with key parameters.

        :param max_age: Maximum number of frames to keep a tracker alive without matching a detection.
        :param min_hits: Minimum number of hits to consider a tracker valid.
        :param iou_threshold: Intersection over union threshold for matching detections to trackers.
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, detections=np.empty((0, 5))):
        """
        Updates the state of the Sort object for a given frame and its detections.

        :param detections: A numpy array of detections in the format [[x1,y1,x2,y2,score], ...].
                           Use np.empty((0, 5)) for frames without detections.
        :return: A numpy array similar to the input detections, but with the last column being the object ID.
                 Note that the number of objects returned may differ from the number of input detections.
        """
        self.frame_count += 1

        # Get predicted locations from existing trackers
        tracked_positions = np.array([tracker.predict()[0] for tracker in self.trackers])

        # Remove invalid trackers
        valid_indices = [i for i, trk in enumerate(tracked_positions) if not np.any(np.isnan(trk))]
        self.trackers = [self.trackers[i] for i in valid_indices]
        tracked_positions = tracked_positions[valid_indices]

        # Associate detections to trackers based on IoU threshold
        matched, unmatched_detections, _ = self.associate_detections_to_trackers(
            detections, tracked_positions, self.iou_threshold
        )

        # Update matched trackers with assigned detections
        for det_idx, trk_idx in matched:
            self.trackers[trk_idx].update(detections[det_idx, :])

        # Create and initialize new trackers for unmatched detections
        for i in unmatched_detections:
            self.trackers.append(KalmanBoxTracker(detections[i, :]))

        # Prepare the output list and update the list of alive trackers
        output = []
        alive_trackers = []
        for tracker in self.trackers:
            state = tracker.get_state()[0]
            if (tracker.time_since_update < 1) and (
                    tracker.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                output.append(np.concatenate((state, [tracker.id + 1])).reshape(1,
                                                                                -1))  # +1 as MOT benchmark requires
                # Positive IDs
            if tracker.time_since_update <= self.max_age:
                alive_trackers.append(tracker)
        self.trackers = alive_trackers

        return np.concatenate(output) if len(output) > 0 else np.empty((0, 5))

    def associate_detections_to_trackers(self, detections, _trackers, iou_threshold=0.3):
        """
        Assigns detections to tracked objects (both represented as bounding boxes).

        Args:
            detections: A list of bounding boxes representing the detections.
            _trackers: A list of bounding boxes representing the tracked objects.
            iou_threshold: The minimum intersection-over-union (IOU) threshold for a detection
                to be considered a match to a tracked object.

        Returns:
            A tuple of three arrays: matches, unmatched_detections, and unmatched_trackers.
            - matches: A 2D array of matched detection-tracker indices.
            - unmatched_detections: A 1D array of indices of unmatched detections.
            - unmatched_trackers: A 1D array of indices of unmatched tracked objects.
        """
        if len(_trackers) == 0:
            # If there are no tracked objects, all detections are unmatched.
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

        # Compute the IOU matrix between detections and tracked objects.
        iou_matrix = self.iou_batch(detections, _trackers)

        if min(iou_matrix.shape) == 0:
            # If the IOU matrix is empty, all detections are unmatched.
            matched_indices = np.empty(shape=(0, 2))
        else:
            # Find the matched indices using the IOU matrix.
            a = (iou_matrix > iou_threshold).astype(np.int32)
            if np.all(a.sum(axis=1) == 1) and np.all(a.sum(axis=0) == 1):
                # If there is only one match per detection and tracked object, use the matches.
                matched_indices = np.column_stack(np.where(a))
            else:
                # Otherwise, use the Hungarian algorithm to find the optimal matches.
                matched_indices = self.linear_assignment(-iou_matrix)

        # Compute the unmatched detections and tracked objects.
        unmatched_detections = set(range(len(detections))) - set(matched_indices[:, 0])
        unmatched_trackers = set(range(len(_trackers))) - set(matched_indices[:, 1])
        unmatched_detections = np.array(list(unmatched_detections))
        unmatched_trackers = np.array(list(unmatched_trackers))

        # Compute the matches that meet the IOU threshold.
        matches = matched_indices[iou_matrix[matched_indices[:, 0], matched_indices[:, 1]] >= iou_threshold]
        if len(matches) == 0:
            # If there are no matches, return an empty array.
            matches = np.empty((0, 2), dtype=int)
        else:
            # Otherwise, reshape the matches array to remove the unnecessary dimension.
            matches = matches.reshape(-1, 2)

        return matches, unmatched_detections, unmatched_trackers

    @staticmethod
    def linear_assignment(cost_matrix):
        """
        Solve the linear assignment problem using lapjv (if available) or scipy's linear_sum_assignment.

        Args:
            cost_matrix (np.array): A cost matrix for the linear assignment problem.

        Returns:
            np.array: A numpy array containing the assignment pairs.
        """
        try:
            import lap
            _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
            return np.array([[y[i], i] for i in x if i >= 0])  #
        except ImportError:
            from scipy.optimize import linear_sum_assignment
            x, y = linear_sum_assignment(cost_matrix)
            return np.array(list(zip(x, y)))

    @staticmethod
    def iou_batch(bb_test, bb_gt):
        """
        Computes IOU between two bounding boxes in the form [x1, y1, x2, y2].

        Args:
            bb_test (np.array): Test bounding boxes.
            bb_gt (np.array): Ground truth bounding boxes.

        Returns:
            np.array: An array containing the IOU values between the test and ground truth bounding boxes.
        """
        if bb_test.size == 0 or bb_gt.size == 0:
            return np.zeros((bb_test.shape[0], bb_gt.shape[0]))

        bb_gt = np.expand_dims(bb_gt, 0)
        bb_test = np.expand_dims(bb_test, 1)

        xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
        yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
        xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
        yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
                  + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
        return o
