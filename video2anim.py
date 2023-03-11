"""
Video to animation

This program makes use of OpenPose (https://github.com/CMU-Perceptual-Computing-Lab/openpose) pose detection to
transform a video into a 2D animation file in Unity's .anim format. Also, process
the results for a smooth animation. Multiple animations of different people can be obtained from a video.

This file contains the Video2Anim class, which allows the process.
More information can be found in the README.md or the specific methods docstrings.
"""


import os
import subprocess
from json import load as json_load
from math import atan2, degrees
from string import Template
from cv2 import VideoCapture, CAP_PROP_FPS
import numpy as np
from bisect import bisect
import time
import heapq


class Video2Anim:
	"""
	Main class of Video2Anim.

	The main methods are the constructor, set_settings and run.

	The program requires the video_path (input), output_folder (output),
	openpose_path (for execution) and bones_defs (settings) to work.
	These settings and more can be assigned at any of the main methods as keyword arguments,
	and it's description can be found in set_settings method docstring.

	Parallelism is not supported for now.
	"""

	###################### Constants ######################
	DEFAULT_BODY_ORIENTATION = 90
	DEFAULT_MIN_CONFIDENCE = 0.6
	DEFAULT_MIN_TREMBLING_FREQ = 7
	DEFAULT_MLF_MAX_ERROR_RATIO = 0.1
	DEFAULT_AVG_KEYS_PER_SEC = 0
	OPENPOSE_RELATIVE_EXE_PATH = os.path.normpath("bin/OpenPoseDemo.exe")
	NUM_BONE_DEF_VALUES = 4

	ANIM_FILE_TEMPLATE = '''%YAML 1.1
%TAG !u! tag:unity3d.com,2011:
--- !u!74 &7400000
AnimationClip:
  m_ObjectHideFlags: 0
  m_CorrespondingSourceObject: {fileID: 0}
  m_PrefabInstance: {fileID: 0}
  m_PrefabAsset: {fileID: 0}
  m_Name: $NAME
  serializedVersion: 6
  m_Legacy: 0
  m_Compressed: 0
  m_UseHighQualityCurve: 1
  m_RotationCurves: []
  m_CompressedRotationCurves: []
  m_EulerCurves:
$EULER_CURVES  m_PositionCurves: []
  m_ScaleCurves: []
  m_FloatCurves: []
  m_PPtrCurves: []
  m_SampleRate: 60
  m_WrapMode: 0
  m_Bounds:
    m_Center: {x: 0, y: 0, z: 0}
    m_Extent: {x: 0, y: 0, z: 0}
  m_ClipBindingConstant:
    genericBindings:
    - serializedVersion: 2
      path: 134607859
      attribute: 4
      script: {fileID: 0}
      typeID: 4
      customType: 4
      isPPtrCurve: 0
    pptrCurveMapping: []
  m_AnimationClipSettings:
    serializedVersion: 2
    m_AdditiveReferencePoseClip: {fileID: 0}
    m_AdditiveReferencePoseTime: 0
    m_StartTime: 0
    m_StopTime: $DURATION
    m_OrientationOffsetY: 0
    m_Level: 0
    m_CycleOffset: 0
    m_HasAdditiveReferencePose: 0
    m_LoopTime: 1
    m_LoopBlend: 0
    m_LoopBlendOrientation: 0
    m_LoopBlendPositionY: 0
    m_LoopBlendPositionXZ: 0
    m_KeepOriginalOrientation: 0
    m_KeepOriginalPositionY: 1
    m_KeepOriginalPositionXZ: 0
    m_HeightFromFeet: 0
    m_Mirror: 0
  m_EditorCurves:
$EDITOR_CURVES  m_EulerEditorCurves: []
  m_HasGenericRootTransform: 0
  m_HasMotionFloatCurves: 0
  m_Events: []
'''
	EULER_CURVE_TEMPLATE = '''  - curve:
      serializedVersion: 2
      m_Curve:
$CURVE      m_PreInfinity: 2    
      m_PostInfinity: 2
      m_RotationOrder: 4
    path: $PATH
'''
	EULER_CURVE_KEY_TEMPLATE = '''      - serializedVersion: 3
        time: $TIME
        value: $VALUE
        inSlope: $SLOPE
        outSlope: $SLOPE
        tangentMode: 0
        weightedMode: 0
        inWeight: {x: 0, y: 0, z: 0.5}
        outWeight: {x: 0, y: 0, z: 0.5}
'''
	EDITOR_CURVE_TEMPLATE = '''  - curve:
      serializedVersion: 2
      m_Curve:
$CURVE      m_PreInfinity: 2
      m_PostInfinity: 2
      m_RotationOrder: 4
    attribute: $ATTR
    path: $PATH
    classID: 4
    script: {fileID: 0}
'''
	EDITOR_CURVE_KEY_TEMPLATE = '''      - serializedVersion: 3
        time: $TIME
        value: $VALUE
        inSlope: $SLOPE
        outSlope: $SLOPE
        tangentMode: 0
        weightedMode: 0
        inWeight: 0.5
        outWeight: 0.5
'''

	###################### For external use ######################
	def __init__(self, **kwargs):
		"""
		Constructor of the class.
		All the settings can be assigned in it if desired.
		Undefined settings will have a default value (usually None).

		:param kwargs: Settings to assign. All the available settings are described in set_settings docstring.
		"""

		# Set default settings
		self.video_path = None
		self.video_name = None
		self.frame_rate = None
		self.time_per_frame = None
		self.output_folder = None		
		self.poses_folder = None
		self.anim_folder = None
		self.openpose_path = None
		self.bones_defs = None
		self.body_orientation = self.DEFAULT_BODY_ORIENTATION
		self.min_confidence = self.DEFAULT_MIN_CONFIDENCE
		self.min_trembling_freq = self.DEFAULT_MIN_TREMBLING_FREQ
		self.max_trembling_period = 1 / self.DEFAULT_MIN_TREMBLING_FREQ if self.DEFAULT_MIN_TREMBLING_FREQ > 0 else 0
		self.mlf_max_error_ratio = self.DEFAULT_MLF_MAX_ERROR_RATIO
		self.avg_keys_per_sec = self.DEFAULT_AVG_KEYS_PER_SEC

		# Set the settings of the arguments
		self.set_settings(**kwargs)

	def set_settings(self, **kwargs):
		"""
		Sets all the program settings defined in kwargs.
		Don't use it if other class method is running in parallel.

		:param kwargs: Settings to assign. The available settings are described below.
		:keyword video_path: Path to the video file. Video format must be compatible with OpenPose, such as .mp4 or .avi.
		:keyword output_folder: Folder path to put the results. A folder with the name of the video will be created there. 
			Inside that folder, a Poses folder is created for containing the OpenPose's .json files. Also a Animations folder for the output .anim files.
		:keyword openpose_path: Path to the OpenPose folder. The folder must contain bin and models folders.
		:keyword bones_defs: List of bones for the animation.
			Every position must have the following values:
				*1. Bone initial position in BODY_25 format.
				*2. Bone end position in BODY_25 format.
				*3. Index of the parent bone (-1 if don't has parent). Usually used for arms and legs.
				*4. GameObject path in hierarchy. For example: bone1/bone2/bone3.
			If a child bone if placed before the parent bone, the list will be copied and re-sorted.
		:keyword body_orientation: In which orientation is the body placed. By default is 90 degrees (vertical).
		:keyword min_confidence: Minimum confidence in a keypoint required for its use. In range [0, 1].
		:keyword min_trembling_freq: Used for trembling reduction.
			Minimum frequency of consecutive ups and downs required to delete such keypoints.
			See the README.md for more information.
			By default has a value of 7. A value of 0 disables the process.
		:keyword mlf_max_error_ratio: Multi-line Fitting maximum error ratio. In range [0, 1].
			The maximum permissible error in MLF in proportion to the difference between
			the minimum and maximum value of the bone animation.
			See the README.md for more information.
			By default has a value of 0.1. A value of 0 disables the process.
		:keyword avg_keys_per_sec: Average keypoints per second desired.
			If a bone has more keys during a second, the amount will be reduced computing averages.
			This is an alternative to trembling reduction and MLF, it is not recommended to use it at the same time.
			By default has a value of 0, which disables the process.

		:raises AssertionError: If any setting is incorrect.
		"""

		# If has any setting
		if len(kwargs):
			video_path = kwargs.get("video_path", None)
			output_folder = kwargs.get("output_folder", None)
			openpose_path = kwargs.get("openpose_path", None)
			bones_defs = kwargs.get("bones_defs", None)
			body_orientation = kwargs.get("body_orientation", None)
			min_confidence = kwargs.get("min_confidence", None)
			min_trembling_freq = kwargs.get("min_trembling_freq", None)
			mlf_max_error_ratio = kwargs.get("mlf_max_error_ratio", None)
			avg_keys_per_sec = kwargs.get("avg_keys_per_sec", None)

			# Input video
			if video_path is not None:
				video_path = os.path.normpath(video_path)
				if not os.path.exists(video_path):
					raise AssertionError(f"video_path {video_path} does not exist")
				self.video_path = video_path
				
				self.video_name = os.path.splitext(os.path.basename(self.video_path))[0]
				video = VideoCapture(self.video_path)
				if not video.isOpened():
					raise AssertionError(f"video_path {video_path} does not contain a video")

				self.frame_rate = video.get(CAP_PROP_FPS)
				self.time_per_frame = 1 / self.frame_rate
				video.release()

			# Output folder and subfolders
			if output_folder is not None:
				self.output_folder = os.path.normpath(output_folder)
				os.makedirs(os.path.dirname(self.output_folder), exist_ok=True)
			
				self.poses_folder = os.path.join(self.output_folder, self.video_name, "Poses")
				os.makedirs(os.path.dirname(self.poses_folder), exist_ok=True)

				self.anim_folder = os.path.join(self.output_folder, self.video_name, "Animations")
				os.makedirs(os.path.dirname(self.anim_folder), exist_ok=True)

			# OpenPose
			if openpose_path is not None:
				openpose_path = os.path.normpath(openpose_path)
				if not os.path.exists(openpose_path):
					raise AssertionError(f"openpose_path {openpose_path} does not exist")

				openpose_exe_path = os.path.join(openpose_path, self.OPENPOSE_RELATIVE_EXE_PATH)
				if not os.path.exists(openpose_exe_path):
					raise AssertionError(
						f"openpose_path {openpose_path} does not contain {self.OPENPOSE_RELATIVE_EXE_PATH}")

				self.openpose_path = openpose_path

			if bones_defs is not None:
				bones_defs = self.check_and_sort_bones_defs(bones_defs)
				self.bones_defs = bones_defs

			if body_orientation is not None:
				self.body_orientation = body_orientation % 360

			if min_confidence is not None:
				if min_confidence < 0 or min_confidence > 1:
					raise AssertionError("min_confidence must be in the range [0, 1]")
				self.min_confidence = min_confidence

			if min_trembling_freq is not None:
				if min_trembling_freq < 0:
					raise AssertionError("min_trembling_freq must be greater than or equal to 0")
				self.min_trembling_freq = min_trembling_freq
				self.max_trembling_period = 1 / min_trembling_freq if min_trembling_freq > 0 else 0

			if mlf_max_error_ratio is not None:
				if mlf_max_error_ratio < 0 or min_confidence > 1:
					raise AssertionError("mlf_max_error_ratio must be in the range [0, 1]")
				self.mlf_max_error_ratio = mlf_max_error_ratio

			if avg_keys_per_sec is not None:
				if avg_keys_per_sec < 0:
					raise AssertionError("avg_keys_per_sec must be greater than or equal to 0")
				self.avg_keys_per_sec = avg_keys_per_sec

	def run(self, person_idx=0, **kwargs):
		"""
		Executes the video to animation translation following the next steps:
		1. Assign the settings passed as **kwargs.
		2. Check if the required arguments (video_path, output_folder, openpose_path and bones_defs) are defined.
		3. Execute OpenPose with the video selected if output_folder contains no previous results,
			putting the resulting .json files in output_folder/<video name>/Poses.
		4. Read the poses of the specified person_idx from the .json files.
		5. Process the poses to get a smooth animation.
		6. Write the results in an .anim file in output_folder/<video name>/Animations, using as name the name of the video concatenated with the person index.

		:param person_idx: Index of the person to create the animation. By default is 0, the first person.
		:param kwargs: Settings to assign before execution. Can be None.
			All the available settings are described in the set_setting method docstring.

		:raises AssertionError: If any setting is incorrect or the required parameters are not defined.

		:return: The bones values of the poses obtained after the process step.
		"""
		self.set_settings(**kwargs)
		self.check_if_can_run()

		# Detect poses
		if not os.path.exists(self.poses_folder):
			self.detect_poses(self.video_path, self.poses_folder)

		# Read poses and process animation
		bones_values, duration = self.read_poses(self.poses_folder, self.bones_defs, person_idx)
		bones_values = self.process_animation(bones_values)

		# Write animation
		self.out_anim_path = os.path.join(self.anim_folder, f"{self.video_name}_{person_idx}.anim")
		self.out_anim_path = os.path.normpath(self.out_anim_path)
		self.write_anim(bones_values, self.bones_defs, duration, self.out_anim_path)

		return bones_values

	###################### Error checking ######################
	def check_and_sort_bones_defs(self, bones_defs):
		"""
		Checks the bones definitions list content and order.
		If a child bone if placed before the parent bone, the list will be copied and re-sorted.

		:param bones_defs: Bones definitions to check.

		:raises AssertionError: If the list has an error that can't be solved sorting.

		:return: The bones definitions list correctly sorted.
		"""
		if not isinstance(bones_defs, list) or len(bones_defs) == 0:
			raise AssertionError("bones_defs must be a not-empty list, one element per bone")
		else:
			idx = 0
			num_bones = len(bones_defs)
			is_sorted = True
			while idx < num_bones:
				bone_def = bones_defs[idx]
				if len(bone_def) < self.NUM_BONE_DEF_VALUES:
					raise AssertionError(
						f"Bone in position {idx} of bones_defs has less than {self.NUM_BONE_DEF_VALUES} values\n {bone_def}")
				parent_idx = bone_def[2]
				if parent_idx == idx:
					raise AssertionError(
						f"Bone in position {idx} of bones_defs has himself as parent \n {self.bones_defs}")
				# If the bone_def is positioned before its parent, solve the problem
				elif parent_idx >= 0 and idx < parent_idx:
					if is_sorted:
						is_sorted = False
						bones_defs = bones_defs.copy()  # Copy the list before sort, to avoid modification of the original

					# Move the bone_def to a correct position
					new_bone_idx = parent_idx
					self.bones_defs.insert(new_bone_idx, self.bones_defs.pop(idx))
					bone_def[2] -= 1  # Now the parent is just before

					# Update affected parents indexes of bones_defs
					upd_idx = idx
					while upd_idx < num_bones:
						if upd_idx != new_bone_idx:
							upd_bone = self.bones_defs[upd_idx]
							upd_parent_idx = upd_bone[2]
							# If the parent index is between the movement positions, needs to be changed
							if idx <= upd_parent_idx <= new_bone_idx:
								# If it's a child of the moved
								if upd_parent_idx == idx:
									upd_parent_idx = new_bone_idx
								# If the parent is before the moved
								else:
									upd_parent_idx -= 1

								upd_bone[2] = upd_parent_idx

						upd_idx += 1

					idx -= 1  # Compensate the movement
				idx += 1

		return bones_defs

	def check_if_can_run(self):
		"""Checks if all the required parameters are defined.
		:raises AssertionError: If any required parameter is missing.
		"""
		message = " setting must be assigned before run. " + \
		          "Add it as keyword argument when calling the constructor or the methods run or set_settings."

		if self.video_path is None:
			raise AssertionError("video_path" + message)
		if self.output_folder is None:
			raise AssertionError("output_folder" + message)
		if self.openpose_path is None:
			raise AssertionError("openpose_path" + message)
		if self.bones_defs is None:
			raise AssertionError("bones_defs" + message)

	###################### Detect poses (use OpenPose) ######################
	def detect_poses(self, in_path, out_path):
		"""Executes OpenPose with a video to obtain the poses data.
		The working directory will be moved to openpose_path during the process.

		:param in_path: Path of the video
		:param out_path: Path where OpenPose will output the .json result files

		:raises Exception: If any problem happens during the execution of OpenPose.
		"""
		current_path = os.getcwd()
		try:
			os.chdir(self.openpose_path)
			self.exe_openpose(in_path, out_path)
		except Exception as exc:
			raise Exception(f"Problem executing OpenPose: {exc}", exc)
		finally:
			os.chdir(current_path)

	def exe_openpose(self, in_path, out_path):
		"""Executes OpenPose as a command, using subprocess.run method.
		:param in_path: Path of the video
		:param out_path: Path where OpenPose will output the .json result files
		"""
		command = f"{self.OPENPOSE_RELATIVE_EXE_PATH} --keypoint_scale 3 --video {in_path} --write_json {out_path} --output_resolution 640x360"
		subprocess.run(command, shell=True, check=True)

	###################### Read poses files ######################
	def read_poses(self, poses_folder, bones_defs, person_idx=0):
		"""
		Reads and stores the poses from the .json files outputted by OpenPose.

		:param poses_folder: Folder which contains the .json files.
		:param bones_defs: Bones definitions to use.
		:param person_idx: Index of the person to create the animation. By default is 0, the first person.
		:return: A list of lists where every position belongs to a bone.
			Each bone has a list where every element is a keypoint (pair of timestamp and angle).
		"""

		bones_values = [[] for _ in range(len(bones_defs))]
		time = 0
		duration = 0
		num_frames = 0
		first_correct_frame = -1

		# Read bones values
		file_names = os.listdir(poses_folder)
		for file_name in file_names:
			if file_name.endswith(".json"):
				file_path = os.path.join(poses_folder, file_name)
				with open(file_path) as frame_file:
					frame_dict = json_load(frame_file)
					time = (num_frames - max(0, first_correct_frame)) * self.time_per_frame
					contains_data = self.get_bones_values(frame_dict, time, bones_defs, bones_values, person_idx)
					if contains_data:
						if first_correct_frame == -1:
							first_correct_frame = num_frames
						else:
							duration = time
					num_frames += 1

		return bones_values, duration

	def get_bones_values(self, frame_dict, time, bones_defs, bones_values, person_idx=0):
		"""
		Gets the bones angles of a frame and a person.
		If a bone has a parent, it decreases the parent angle of the obtained one.

		:param frame_dict: Dictionary extracted from the .json file of a frame.
		:param time: Timestamp of the frame.
		:param bones_defs: Bones definitions to use.
		:param bones_values: Current list of bones values. The new values will be appended to it.
		:param person_idx: Index of the person to create the animation. By default is 0, the first person.
		:return: If the frame contains at least one keypoint.
		"""
		contains_data = False

		people = frame_dict["people"]
		if len(people) > person_idx:
			keypoints = people[person_idx]["pose_keypoints_2d"]
			for i, bone_def in enumerate(bones_defs):
				# Manage parent value
				parent_idx = bone_def[2]
				needs_parent = parent_idx != -1
				has_parent = False
				parent_values = None
				if needs_parent:
					parent_bone = bones_values[parent_idx]
					has_parent = len(parent_bone) > 0
					if has_parent:
						parent_values = parent_bone[-1]
						has_parent = parent_values[0] == time  # Last frame is the current

				is_correct = has_parent or not needs_parent
				if is_correct:
					ini = self.get_kp(keypoints, bone_def[0])
					end = self.get_kp(keypoints, bone_def[1])

					if ini and end:
						offset = self.kp_sub(end, ini)
						angle = atan2(offset[1], offset[0])
						angle = degrees(angle) - self.body_orientation
						if has_parent:
							angle -= parent_values[1]
						angle = angle % 360

						# Use the more similar angle to the previous
						if len(bones_values[i]) > 0:
							previous_angle = bones_values[i][-1][1]
							possible_angles = [angle + 360, angle - 360]
							min_diff = abs(angle - previous_angle)
							for possible_angle in possible_angles:
								diff = abs(possible_angle - previous_angle)
								if diff < min_diff:
									min_diff = diff
									angle = possible_angle

						bones_values[i].append([time, angle])

				if is_correct and not contains_data:
					contains_data = True

		return contains_data

	def get_kp(self, keypoints, idx):
		"""
		Gets a keypoint of the array only if is confidence is greater or equal to the minimum required.
		Exceptionally, if the index is not an integer, gets the average of the current keypoint and next one.

		:param keypoints: Array of keypoints of a frame.
		:param idx: Index of the keypoint in BODY_25 format.
		:return: The keypoint of the index or, if the confidence isn't enough, None.
		"""
		kp = None

		if idx % 1 == 0:
			ini = idx * 3
			val = keypoints[ini: ini + 3]

			if val[2] >= self.min_confidence:
				kp = val
				kp[1] = 1 - kp[1]
		else:
			a = self.get_kp(keypoints, int(idx))
			b = self.get_kp(keypoints, int(idx + 1))
			if a and b:
				kp = self.kp_average(a, b)

		return kp

	def kp_average(self, a, b):
		"""
		Computes the average of two keypoints time and angle.
		:param a: First keypoint.
		:param b: Second keypoint.
		:return: A keypoint with the average time and angle of both.
		"""
		return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]

	def kp_sub(self, a, b):
		"""
		Computes the difference between two keypoints time and angle.
		:param a: First keypoint.
		:param b: Second keypoint.
		:return: A keypoint with the difference of time and angle.
		"""
		return [a[0] - b[0], a[1] - b[1]]

	########### Post-process animation ###########
	def process_animation(self, bones_values):
		"""
		Process the poses keypoints to smooth the animation.
		In order to smooth it uses multi_line_fitting, reduce_trembling and check_avg_keys_per_sec methods.
		The methods used depends on the current settings.
		Finally, computes the slope of each keypoint.
		See the README.md for more information.

		:param bones_values: A list of lists where every position belongs to a bone.
			Each bone has a list where every element is a keypoint (pair of timestamp and angle).
		:return: A list of lists where every position belongs to a bone.
			Each bone has a list where every element is a keypoint (timestamp, angle and slope).
		"""
		num_execs = 0
		num_keys = 0
		for bone_idx in range(len(bones_values)):
			bone_keys = bones_values[bone_idx]
			num_bone_keys = len(bone_keys)
			if num_bone_keys != 0:
				# Remove high frequency trembling
				if self.min_trembling_freq > 0:
					bones_values[bone_idx] = bone_keys = self.reduce_trembling(bone_keys)

				# Fit the line to reduce redundancy and noise
				if self.mlf_max_error_ratio > 0:
					num_execs += 1
					num_keys += len(bone_keys)
					bones_values[bone_idx] = bone_keys = self.multi_line_fitting(bone_keys)

				# Compute averages if is needed
				if self.avg_keys_per_sec > 0:
					bones_values[bone_idx] = bone_keys = self.check_avg_keys_per_sec(bone_keys)

				# Compute slopes
				for key_idx, key in enumerate(bone_keys):
					slope = self.compute_slope(bone_keys, key_idx)
					key.append(slope)

		return bones_values

	def reduce_trembling(self, bone_keys):
		"""
		Removes the keypoints that makes consecutive ups and downs at high frequency (greater or equal to min_trembling_freq).
		See the README.md for more information.

		:param bone_keys: Keypoints of a bone.
		:return: Keypoints of a bone with trembling reduced.
		"""

		num_bone_keys = len(bone_keys)
		new_bone_keys = [bone_keys[0]]

		key_idx = 1
		previous_kp = bone_keys[0]
		keypoint = bone_keys[key_idx]
		ini_wave_idx = -1
		while key_idx < num_bone_keys - 1:
			next_kp = bone_keys[key_idx + 1]
			has_wave_period = (next_kp[0] - previous_kp[
				0]) <= self.max_trembling_period  # Time smaller than the maximum trembling period
			has_wave_values = (keypoint[1] - previous_kp[1] > 0) != (
					next_kp[1] - keypoint[1] > 0)  # It's a wave if the slope changes
			is_trembling_wave = has_wave_period and has_wave_values
			if is_trembling_wave:
				# If is the start of a trembling wave
				if ini_wave_idx == -1:
					ini_wave_idx = key_idx
			else:
				# If is end of the wave, ignore the trembling
				if ini_wave_idx != -1:
					ini_wave_idx = -1

				# Always add if it don't belongs to a trembling wave
				new_bone_keys.append(keypoint)

			# Advance to next
			previous_kp = keypoint
			keypoint = next_kp
			key_idx += 1

		# Always add last keypoint
		new_bone_keys.append(bone_keys[-1])

		return new_bone_keys

	def multi_line_fitting(self, bone_keys):
		"""
		Applies the Multi-Line Fitting algorithm (implemented in the MultiLineFitting class) to bone_keys.
		An explanation of MLF can be found in the README.md.

		:param bone_keys: A list which contains the keypoints (pair of timestamp and angle) of a bone.
		:return: The new bone keys list processed by the MLF algorithm.
		"""
		# Perform MultiLineFitting
		estimated_keypoints_arr = MultiLineFitting(self.mlf_max_error_ratio)(bone_keys)

		# Transform to list of lists
		estimated_keypoints = [list(x) for x in estimated_keypoints_arr]	

		return estimated_keypoints

	def check_avg_keys_per_sec(self, bone_keys):
		"""
		Checks if bone_keys has more keys than the desired in a second.
		Then, computes the averages of some keypoints to reduce the amount to the defined one.
		See the README.md for more information.

		:param bone_keys: Keypoints list of a bone.
		:return: Processed bone_keys with the corresponding averages.
		"""
		num_bone_keys = len(bone_keys)
		ini_second_time = bone_keys[1][0]
		ini_second_key_idx = 1
		key_idx = 2
		new_keypoint_idx = 1

		while key_idx < num_bone_keys - 1:
			time = bone_keys[key_idx][0]
			elapsed_time = time - ini_second_time
			is_penultimate = key_idx == (num_bone_keys - 2)

			if is_penultimate:
				key_idx += 1  # Needed to include it in averages computation

			# If a second passed
			if elapsed_time > 1 or is_penultimate:
				keys_count = key_idx - ini_second_key_idx
				# If the key count is less than or equal to the desired
				if keys_count <= self.avg_keys_per_sec:
					new_keypoint_idx += keys_count  # Don't modify these keys
				# If the key count is grater than the desired
				else:
					num_keys_to_avg = keys_count // self.avg_keys_per_sec
					ini = ini_second_key_idx
					end = ini + num_keys_to_avg
					# Compute the averages
					for i in range(0, self.avg_keys_per_sec - 1):
						keys_for_average = bone_keys[ini:end]
						bone_keys[new_keypoint_idx] = self.bone_keys_average(keys_for_average)
						new_keypoint_idx += 1
						ini = end
						end = ini + num_keys_to_avg

					# Add the last average
					keys_for_average = bone_keys[ini:key_idx]
					bone_keys[new_keypoint_idx] = self.bone_keys_average(keys_for_average)
					new_keypoint_idx += 1

				ini_second_time = time
				ini_second_key_idx = key_idx

			key_idx += 1

		# Add the last keypoint
		bone_keys[new_keypoint_idx] = bone_keys[-1]
		new_keypoint_idx += 1

		return bone_keys[:new_keypoint_idx]

	def bone_keys_average(self, bone_keys):
		"""
		Computes the time and angle average of a list of keypoints.

		:param bone_keys: List of keypoints to average.
		:return: A unique keypoint with the time and angle average.
		"""

		average = [0, 0]

		# Sum values
		for time, value in bone_keys:
			average[0] = average[0] + time
			average[1] = average[1] + value

		# Compute averages
		num_values = float(len(bone_keys))
		average[0] /= num_values
		average[1] /= num_values

		return average

	def compute_slope(self, bone_keys, key_idx):
		"""
		Computes the slope of a keypoint using the previous and next keypoints.
		If is the first or last keypoint the slope will be zero.

		:param bone_keys: List of bone keypoints.
		:param key_idx: Index of the keypoint to compute the slope.
		:return: The corresponding slope.
		"""

		slope = 0

		if key_idx != 0 and key_idx != len(bone_keys) - 1:
			previous_kp = bone_keys[key_idx - 1]
			next_kp = bone_keys[key_idx + 1]
			slope = (next_kp[1] - previous_kp[1]) / (next_kp[0] - previous_kp[0])

		return slope

	########### Write animation ###########
	def write_anim(self, bones_values, bones_defs, duration, file_path):
		"""
		Writes the animation to the file using the previously read and processed bones keypoints.
		For that, it uses template strings of the .anim file which it fills with the values.
	    The generated file is in YAML format, and defines the animation as euler_curves and editor_curves (both with the same data).
	    The only modified attribute of the bones is the rotation of the z-axis, because it is a 2D animation.

		:param bones_values: List of bones keypoints with timestamp, angle and slope.
		:param bones_defs: Bones definitions that contains the path of each bone.
		:param duration: Total duration in seconds of the animation.
		:param file_path: Path of the file to write. If the file exists, it will be overwritten.
		"""

		# Prepare templates
		euler_curve_tmpl = Template(self.EULER_CURVE_TEMPLATE)
		euler_curves_str = ''
		euler_keys_tmpl = Template(self.EULER_CURVE_KEY_TEMPLATE)
		euler_keys_str = ''
		editor_curve_tmpl = Template(self.EDITOR_CURVE_TEMPLATE)
		editor_curves_str = ''
		editor_keys_tmpl = Template(self.EDITOR_CURVE_KEY_TEMPLATE)
		editor_keys_str = ''
		curve_tmpl_values = {'CURVE': '', 'PATH': '', 'ATTR': 'localEulerAnglesRaw.z'}
		key_tmpl_values = {'TIME': 0, 'VALUE': 0, 'SLOPE': 0}

		# Process all bone values
		for bone_idx, bone_values in enumerate(bones_values):
			if len(bone_values):
				for time, value, slope in bone_values:
					key_tmpl_values['TIME'] = '%.2f' % time
					key_tmpl_values['VALUE'] = '{x: 0, y: 0, z: ' + value.__str__() + '}'
					key_tmpl_values['SLOPE'] = '{x: 0, y: 0, z: ' + slope.__str__() + '}'
					euler_keys_str += euler_keys_tmpl.substitute(key_tmpl_values)
					key_tmpl_values['VALUE'] = value
					key_tmpl_values['SLOPE'] = slope
					editor_keys_str += editor_keys_tmpl.substitute(key_tmpl_values)

				curve_tmpl_values['PATH'] = bones_defs[bone_idx][3]
				curve_tmpl_values['CURVE'] = euler_keys_str
				euler_curves_str += euler_curve_tmpl.substitute(curve_tmpl_values)
				curve_tmpl_values['CURVE'] = editor_keys_str
				editor_curves_str += editor_curve_tmpl.substitute(curve_tmpl_values)
				euler_keys_str = ''
				editor_curves_str = ''

		# Substitue file regions
		file_tmpl = Template(self.ANIM_FILE_TEMPLATE)
		anim_name = os.path.splitext(os.path.basename(file_path))[0]
		tmpl_values = {'NAME': anim_name,
		               'EULER_CURVES': euler_curves_str,
		               'DURATION': duration,
		               'EDITOR_CURVES': editor_curves_str,
		               'ATTR': 'localEulerAnglesRaw.z'}
		file_content = file_tmpl.substitute(tmpl_values)

		# Write animation file
		os.makedirs(os.path.dirname(file_path), exist_ok=True)
		with open(file_path, "w") as out_file:
			out_file.write(file_content)


class MultiLineFitting():
	def __init__(self, max_error_ratio) -> None:
		self.max_error_ratio = max_error_ratio

	def __call__(self, keypoints):
		keypoints = np.array(keypoints)
		timestamps = keypoints[:, 0]
		values = keypoints[:, 1]
		max_error_thld = (values.max() - values.min()) * self.max_error_ratio

		# Compute initial estimation and errors
		estimation_idxs = [0, len(keypoints)-1]
		values_estimation = np.empty_like(values)
		self.estimate_line(keypoints[0], keypoints[-1], timestamps, values_estimation)
		errors = values - values_estimation
		errors = np.abs(errors, out=errors)
		max_error_idx = errors.argmax()

		# If some error is too high, perform fitting
		if errors[max_error_idx] > max_error_thld:
			top_errors_idxs = [max_error_idx]
			heapq.heapify(top_errors_idxs)

			# Iterative fitting			
			while len(top_errors_idxs):
				# Get maximum error
				max_error_idx = heapq.heappop(top_errors_idxs)
				error = errors[max_error_idx]

				# If error is too high, adapt estimation
				if error > max_error_thld:
					max_error_est_idx = bisect(estimation_idxs, max_error_idx)
					estimation_idxs.insert(max_error_est_idx, max_error_idx)
					est_idx_left = estimation_idxs[max_error_est_idx-1]
					est_idx_right = estimation_idxs[max_error_est_idx+1]

					# New estimation line at left
					self.estimate_line(keypoints[est_idx_left], keypoints[max_error_idx],
						timestamps[est_idx_left:max_error_idx+1],
						values_estimation[est_idx_left:max_error_idx+1])
					
					# New estimation line at right
					self.estimate_line(keypoints[max_error_idx], keypoints[est_idx_right],
						timestamps[max_error_idx:est_idx_right+1],
						values_estimation[max_error_idx:est_idx_right+1])
					
					# Recompute errors
					np.subtract(values[est_idx_left:est_idx_right+1], values_estimation[est_idx_left:est_idx_right+1], out=errors[est_idx_left:est_idx_right+1])
					np.abs(errors[est_idx_left:est_idx_right+1], out=errors[est_idx_left:est_idx_right+1])

					# Get maximum error at left and store it if too high
					new_max_error_idx = errors[est_idx_left:max_error_idx].argmax()
					new_max_error_idx += est_idx_left
					if errors[new_max_error_idx] > max_error_thld:
						heapq.heappush(top_errors_idxs, new_max_error_idx)
					
					# Get maximum error at right and store it if too high
					new_max_error_idx = errors[max_error_idx:est_idx_right+1].argmax()
					new_max_error_idx += max_error_idx
					if errors[new_max_error_idx] > max_error_thld:
						heapq.heappush(top_errors_idxs, new_max_error_idx)
			
		# Gather keypoints for final estimation
		estimated_keypoints = keypoints[estimation_idxs]

		return estimated_keypoints

	def estimate_line(self, ini_keypoint, end_keypoint, timestamps, out):
		time_deltas = (timestamps - ini_keypoint[0])
		slope = (end_keypoint[1] - ini_keypoint[1]) / (end_keypoint[0] - ini_keypoint[0])
		np.multiply(time_deltas, slope, out=out)
		np.add(out, ini_keypoint[1], out=out)
	

if __name__=="__main__":
	raise Exception("Main not implemented")