import numpy as np
from typing import Dict, List, Optional, Union, Any
from scipy.spatial.transform import Rotation

from pycram.datastructures.pose import Pose, Vector3, Quaternion, GraspPose
from pycram.datastructures.grasp import GraspDescription
from pycram.datastructures.enums import ApproachDirection, VerticalAlignment, Arms

class GraspClassifier:
    """Classifies and manages grasps from YAML data structure"""

    def __init__(self, grasp_data: Dict):
        """
        Initialize with grasp data dictionary

        Args:
            grasp_data: Dictionary containing 'grasps' key with grasp definitions
        """
        self.grasp_data = grasp_data
        self.classified_grasps = self._classify_grasps()

    def _classify_grasps(self) -> Dict[ApproachDirection, List[Dict]]:
        """Classify grasps by approach direction with scores"""
        classified = {direction: [] for direction in ApproachDirection}

        for grasp in self.grasp_data['grasps']:
            pose = self._create_pose(grasp)
            approach_dir = self._determine_approach_direction(pose)

            grasp_info = {
                'id': grasp['id'],
                'pose': pose,
                'approach_direction': approach_dir,
                'vertical_alignment': self._determine_vertical_alignment(pose),
                'score': 1.0  # Default score, can be enhanced later
            }

            classified[approach_dir].append(grasp_info)

        return classified

    def _create_pose(self, grasp_data: Dict) -> Pose:
        position = Vector3(
            x=grasp_data['position']['x'],
            y=grasp_data['position']['y'],
            z=grasp_data['position']['z']
        )
        orientation = Quaternion(
            x=grasp_data['orientation']['x'],
            y=grasp_data['orientation']['y'],
            z=grasp_data['orientation']['z'],
            w=grasp_data['orientation']['w']
        )
        return Pose(position=position, orientation=orientation)

    def _determine_approach_direction(self, pose: Pose) -> ApproachDirection:
        """Determine approach direction from pose orientation"""
        quat = [pose.orientation.x, pose.orientation.y,
                pose.orientation.z, pose.orientation.w]
        r = Rotation.from_quat(quat)

        approach_vector = r.apply([0, 0, -1])

        abs_approach_xy = np.abs(approach_vector[:2])

        if abs_approach_xy[0] > abs_approach_xy[1]:
            return ApproachDirection.LEFT if approach_vector[0] < 0 else ApproachDirection.RIGHT
        else:
            return ApproachDirection.BACK if approach_vector[1] < 0 else ApproachDirection.FRONT


    def _determine_vertical_alignment(self, pose: Pose) -> VerticalAlignment:
        """Determine vertical alignment from pose"""
        quat = [pose.orientation.x, pose.orientation.y,
                pose.orientation.z, pose.orientation.w]
        r = Rotation.from_quat(quat)
        up_vector = r.apply([0, 0, 1])

        if up_vector[2] > 0.5:
            return VerticalAlignment.TOP
        elif up_vector[2] <= -0.5:
            return VerticalAlignment.BOTTOM

    def get_grasp_pose(self,
                               directions: Union[ApproachDirection, List[ApproachDirection]],
                               vertical_alignment: Optional[VerticalAlignment] = None,
                               arm: Arms = Arms.RIGHT) -> list[Any] | GraspPose:
        """
        Get GraspDescription objects for specified directions

        Args:
            directions: Single direction or list of directions
            vertical_alignment: Optional vertical alignment filter
            max_grasps: Maximum number of grasps to return

        Returns:
            List of GraspDescription objects sorted by score
        """
        if isinstance(directions, ApproachDirection):
            directions = [directions]

        # Combine grasps from all requested directions
        combined_grasps = []
        for direction in directions:
            grasps = self.classified_grasps.get(direction, [])

            if vertical_alignment:
                grasps = [g for g in grasps if g['vertical_alignment'] == vertical_alignment]

            combined_grasps.extend(grasps)
        if not combined_grasps:
            return []
        # Sort by score (highest first)
        combined_grasps.sort(key=lambda x: x['score'], reverse=True)
        best_grasp = combined_grasps[0]

        # Create GraspDescription
        grasp_description = GraspDescription(
            approach_direction=best_grasp['approach_direction'],
            vertical_alignment=best_grasp['vertical_alignment']
        )

        # Return PyCRAM's GraspPose object
        return GraspPose(
            pose=best_grasp['pose'],
            header=best_grasp['pose'].header if hasattr(best_grasp['pose'], 'header') else None,
            arm=arm,
            grasp_description=grasp_description
        )

    def get_grasp_with_pose(self,
                           directions: Union[ApproachDirection, List[ApproachDirection]],
                           vertical_alignment: Optional[VerticalAlignment] = None) -> Optional[Dict]:
        """
        Get grasp info including the actual pose data

        Returns:
            Dictionary with 'grasp_description' and 'pose' keys
        """
        if isinstance(directions, ApproachDirection):
            directions = [directions]

        # Combine grasps from all requested directions
        combined_grasps = []
        for direction in directions:
            grasps = self.classified_grasps.get(direction, [])
            if vertical_alignment:
                grasps = [g for g in grasps if g['vertical_alignment'] == vertical_alignment]
            combined_grasps.extend(grasps)

        if not combined_grasps:
            return None

        # Sort by score and get best
        combined_grasps.sort(key=lambda x: x['score'], reverse=True)
        best_grasp = combined_grasps[0]

        return {
            'grasp_description': GraspDescription(
                approach_direction=best_grasp['approach_direction'],
                vertical_alignment=best_grasp['vertical_alignment']
            ),
            'pose': best_grasp['pose'],
            'id': best_grasp['id']
        }

    def get_best_grasp_description(self,
                                   directions: Union[ApproachDirection, List[ApproachDirection]],
                                   vertical_alignment: Optional[VerticalAlignment] = None) -> Optional[
        GraspDescription]:
        """Get the best single grasp description"""
        grasps = self.get_grasp_descriptions(directions, vertical_alignment, max_grasps=1)
        return grasps[0] if grasps else None

    def update_scores(self, scoring_function):
        """Update grasp scores using custom scoring function"""
        for direction, grasps in self.classified_grasps.items():
            for grasp in grasps:
                grasp['score'] = scoring_function(grasp)