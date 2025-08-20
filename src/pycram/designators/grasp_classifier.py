import numpy as np
from typing import Dict, List, Optional, Union, Any

from mercurial.smartset import filteredset
from scipy.spatial.transform import Rotation

from pycram import tf_transformations
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

    def _normalize_quaternion(self, q):
        q = np.array(q, dtype=float)
        n = np.linalg.norm(q)
        return (q / n).tolist() if n > 0 else [0, 0, 0, 1]

    def relative_gripper_pose_to_world(self, object_frame_world: PoseStamped,
                                       gripper_frame_object: PoseStamped) -> PoseStamped:
        """
        Convert gripper pose expressed in object frame to world frame.

        Inputs:
          object_frame_world: PoseStamped (frame: world) -> T_world_object
          gripper_frame_object: PoseStamped (frame: object) -> T_object_gripper

        Output:
          PoseStamped (frame: world) -> T_world_gripper

        Math:
          q_world_gripper = q_world_object * q_object_gripper
          p_world_gripper = p_world_object + R(q_world_object) * p_object_gripper
        """

        # Object quaternion
        q_obj = object_frame_world.orientation.to_list()
        # Relative (object->gripper) quaternion
        q_grip = gripper_frame_object.orientation.to_list()

        r_obj = tf_transformations.quaternion_matrix(q_obj)[:3, :3]

        # Relative translation (object frame)
        t_grip = gripper_frame_object.position.to_list()

        # World translation
        t_world = object_frame_world.position.to_numpy() + np.dot(r_obj, t_grip)

        # Compose quaternions (world_object * object_gripper)
        q_world = tf_transformations.quaternion_multiply(q_obj, q_grip)
        q_world = self._normalize_quaternion(q_world)

        return PoseStamped().from_list(t_world.tolist(), q_world, object_frame_world.header.frame_id)

    def get_n_best_grasps(self,
                          n: int,
                          directions: Union[ApproachDirection, List[ApproachDirection]] = None,
                          vertical_alignment: Optional[VerticalAlignment] = None,
                          arm: Arms = Arms.RIGHT,
                          target_object=None,
                          object_world_pose: PoseStamped = None) -> List[Dict]:
        """
        Get n best reachable grasps sorted by score

        Args:
            n: Number of best grasps to return
            directions: Approach direction(s) to filter by (if None, uses all directions)
            vertical_alignment: Vertical alignment to filter by (if None, uses all alignments)
            arm: Robot arm to check reachability for
            target_object: Target object for additional validation??
            object_world_pose: Pose of the target object in world frame (needed for reachability)
                               reachability validation will be skipped if None

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
        candidate = []
        for direction in search_directions:
            grasps = self.classified_grasps.get(direction, [])

            # Filter by vertical alignment if specified
            if vertical_alignment:
                grasps = [g for g in grasps if g['vertical_alignment'] == vertical_alignment]

            candidate.extend(grasps)

        if not candidate:
            return []

        # Validate reachability and enrich grasp info
        filtered_grasps = []
        for grasp in candidate:
            enriched_grasp = {
                'id': grasp['id'],
                'pose_relative': grasp['pose'],
                'approach_direction': grasp['approach_direction'],
                'vertical_alignment': grasp['vertical_alignment'],
                'score': grasp['score']
            }

            # Reachability check
            if object_world_pose is not None:
                pose_world = self.relative_gripper_pose_to_world(object_world_pose, grasp['pose'])

                if self.validate_grasp_reachability(pose_world, arm):
                    # add world pose to grasp info, which is reachable
                    enriched_grasp['pose_world'] = pose_world

            filtered_grasps.append(enriched_grasp)

        # Sort by score (descending) and return top n
        filtered_grasps.sort(key=lambda x: x['score'], reverse=True)

        return filtered_grasps[:n]

    def update_scores(self, scoring_function):
        """Update grasp scores using custom scoring function"""
        for direction, grasps in self.classified_grasps.items():
            for grasp in grasps:
                grasp['score'] = scoring_function(grasp)

