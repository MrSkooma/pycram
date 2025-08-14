import numpy as np
from typing import Dict, List, Optional, Union, Any
from scipy.spatial.transform import Rotation

from pycram.datastructures.pose import PoseStamped, Vector3, Quaternion, GraspPose
from pycram.datastructures.grasp import GraspDescription
from pycram.datastructures.enums import ApproachDirection, VerticalAlignment, Arms
from pycram.external_interfaces.ik import try_to_reach_with_grasp

class GraspClassifier:
    """Classifies and manages grasps from YAML data structure"""

    def __init__(self, grasp_data: Dict, robot=None):
        """
        Initialize with grasp data dictionary

        Args:
            grasp_data: Dictionary containing 'grasps' key with grasp definitions
            robot: Optional robot instance for reachability validation
        """
        self.grasp_data = grasp_data
        self.robot = robot
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

    def _create_pose(self, grasp_data: Dict) -> PoseStamped:
        position = [
            grasp_data['position']['x'],
            grasp_data['position']['y'],
            grasp_data['position']['z']
        ]
        orientation = [
            grasp_data['orientation']['x'],
            grasp_data['orientation']['y'],
            grasp_data['orientation']['z'],
            grasp_data['orientation']['w']
        ]
        return PoseStamped.from_list(position=position, orientation=orientation)

    def _determine_approach_direction(self, pose: PoseStamped) -> ApproachDirection:
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


    def _determine_vertical_alignment(self, pose: PoseStamped) -> VerticalAlignment:
        """Determine vertical alignment from pose"""
        quat = [pose.orientation.x, pose.orientation.y,
                pose.orientation.z, pose.orientation.w]
        r = Rotation.from_quat(quat)
        up_vector = r.apply([0, 0, 1])

        if up_vector[2] > 0.5:
            return VerticalAlignment.TOP
        elif up_vector[2] <= -0.5:
            return VerticalAlignment.BOTTOM

    def validate_grasp_reachability(self, pose: PoseStamped, arm: Arms) -> bool:
        """Check if grasp pose is kinematically reachable"""
        if not self.robot:
            return False

        gripper_name = "r_gripper_tool_frame" if arm == Arms.RIGHT else "l_gripper_tool_frame"
        result_pose = try_to_reach_with_grasp(
            pose, self.robot, gripper_name, [
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w
                ]
        )
        return result_pose is not None

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

    def get_n_best_reachable_grasps(self,
                                    n: int,
                                    directions: Union[ApproachDirection, List[ApproachDirection]] = None,
                                    vertical_alignment: Optional[VerticalAlignment] = None,
                                    arm: Arms = Arms.RIGHT,
                                    target_object=None) -> List[Dict]:
        """
        Get n best reachable grasps sorted by score

        Args:
            n: Number of best grasps to return
            directions: Approach direction(s) to filter by (if None, uses all directions)
            vertical_alignment: Vertical alignment to filter by (if None, uses all alignments)
            arm: Robot arm to check reachability for
            target_object: Target object for additional validation?

        Returns:
            List of grasp dictionaries with pose, description, and reachability info
        """
        # Determine which directions to consider
        if directions is None:
            search_directions = list(ApproachDirection)
        elif isinstance(directions, ApproachDirection):
            search_directions = [directions]
        else:
            search_directions = directions

        # Collect all candidate grasps
        candidate_grasps = []
        for direction in search_directions:
            grasps = self.classified_grasps.get(direction, [])

            # Filter by vertical alignment if specified
            if vertical_alignment:
                grasps = [g for g in grasps if g['vertical_alignment'] == vertical_alignment]

            candidate_grasps.extend(grasps)

        if not candidate_grasps:
            return []

        # Validate reachability and enrich grasp info
        reachable_grasps = []
        for grasp in candidate_grasps:
            # Check reachability
            is_reachable = self.validate_grasp_reachability(grasp['pose'], arm)

            if is_reachable:
                # Create enriched grasp info
                enriched_grasp = {
                    'id': grasp['id'],
                    'pose': grasp['pose'],
                    'score': grasp['score'],
                    'approach_direction': grasp['approach_direction'],
                    'vertical_alignment': grasp['vertical_alignment'],
                    'grasp_description': GraspDescription(
                        approach_direction=grasp['approach_direction'],
                        vertical_alignment=grasp['vertical_alignment']
                    )
                }

                reachable_grasps.append(enriched_grasp)

        # Sort by score (descending) and return top n
        reachable_grasps.sort(key=lambda x: x['score'], reverse=True)

        return reachable_grasps[:n]

    def update_scores(self, scoring_function):
        """Update grasp scores using custom scoring function"""
        for direction, grasps in self.classified_grasps.items():
            for grasp in grasps:
                grasp['score'] = scoring_function(grasp)