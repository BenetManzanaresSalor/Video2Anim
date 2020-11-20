import os
import json
import math
from string import Template

############################################ CONSTANTS ############################################
FRAME_RATE = 25
MIN_CONFIDENCE = 0.6
MAX_KEYS_PER_SECOND = 5
BODY_ORIENTATION = 90

OPENPOSE_PATH = "openpose"

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


############################################ METHODS ############################################
###################### Detect poses ######################
def detect_poses(in_path, out_path):
	current_path = os.getcwd()
	os.chdir(OPENPOSE_PATH)
	exe_OpenPose(in_path, out_path)
	os.chdir(current_path)


def exe_OpenPose(in_path, out_path):
	command = f"start bin/OpenPoseDemo.exe --keypoint_scale 3 --video {in_path} --write_json {out_path} --write_video Benet.avi"
	print(command)
	stream = os.popen(command)
	stream.read()


###################### Process poses ######################
def process_poses(in_path, out_path, bones_settings, person_idx):
	bones_values = [[] for _ in range(len(bones_settings))]
	time = 0
	num_frames = 0
	first_correct_frame = -1
	num_correct_frames = 0
	second_per_frame = 1 / FRAME_RATE

	# Read bones values
	file_names = os.listdir(in_path)
	for file_name in file_names:
		if file_name.endswith(".json"):
			file_path = os.path.join(in_path, file_name)
			with open(file_path) as frame_file:
				frame_values = json.load(frame_file)
				time = (num_frames - max(0, first_correct_frame)) * second_per_frame
				contains_data = get_bones_values(frame_values, bones_settings, person_idx, time, bones_values)
				if contains_data:
					num_correct_frames += 1
					if first_correct_frame == -1:
						first_correct_frame = num_frames
				num_frames += 1

	# Process bones_values, computing averages and slopes
	process_bones_values(bones_values)

	# Write animation
	file_name = os.path.splitext(os.path.basename(in_path))[0]
	file_path = os.path.join(out_path, f"{file_name}{person_idx}.anim")
	write_anim(bones_values, bones_settings, time, file_path)

	print(f"{num_correct_frames} / {num_frames}")


######## Compute bones values ########
def get_bones_values(frame_values, bones_settings, person_idx, time, bones_values):
	contains_data = False
	default_slope = 0

	people = frame_values["people"]
	if len(people) > person_idx:
		keypoints = people[person_idx]["pose_keypoints_2d"]
		for i, bone_setting in enumerate(bones_settings):
			# Manage parent value
			parent_idx = bone_setting[2]
			needs_parent = parent_idx != -1
			has_parent = False
			if needs_parent:
				parent_bone = bones_values[parent_idx]
				has_parent = len(parent_bone) > 0
				if has_parent:
					parent_values = parent_bone[-1]
					has_parent = parent_values[0] == time   # Last frame is the current

			is_correct = not needs_parent or has_parent
			if is_correct:
				ini = get_kp(keypoints, bone_setting[0])
				end = get_kp(keypoints, bone_setting[1])

				if ini and end:
					offset = kp_sub(end, ini)
					angle = math.atan2(offset[1], offset[0])
					angle = math.degrees(angle) - BODY_ORIENTATION
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

					bones_values[i].append([time, angle, default_slope])

			if is_correct and not contains_data:
				contains_data = True

	return contains_data


def get_kp(keypoints, idx):
	kp = None

	if idx % 1 == 0:
		ini = idx * 3
		val = keypoints[ini: ini + 3]

		if val[2] >= MIN_CONFIDENCE:
			kp = val
			kp[1] = 1 - kp[1]
	else:
		a = get_kp(keypoints, int(idx))
		b = get_kp(keypoints, int(idx + 1))
		if a and b:
			kp = kp_average(a, b)

	return kp


def kp_average(a, b):
	return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]


def kp_sub(a, b):
	return [a[0] - b[0], a[1] - b[1]]


########### Process bones values ###########
def process_bones_values(bones_values):
	for i in range(len(bones_values)):
		bone_keys = bones_values[i]
		num_bone_keys = len(bone_keys)

		if num_bone_keys != 0:
			# Compute averages if are needed
			if MAX_KEYS_PER_SECOND != 1:
				previous_time = int(bone_keys[0][0])
				ini_key_idx = 0
				key_idx = 1
				num_keys_in_second = 1
				result_idx = 0
				is_last = key_idx >= num_bone_keys

				while key_idx < num_bone_keys:
					time = bone_keys[key_idx][0]
					num_keys_in_second += 1
					is_last = (key_idx + 1) >= num_bone_keys
					# If a second has passed or more keys than MAX_KEYS_PER_SECOND or is_last => compute the average
					if time >= previous_time + 1 or num_keys_in_second == MAX_KEYS_PER_SECOND or is_last:
						previous_time = time
						num_keys_in_second = 0
						bone_keys[result_idx] = bone_values_average(bone_keys[ini_key_idx:key_idx + 1], is_last)
						ini_key_idx = key_idx + 1
						result_idx += 1

					key_idx += 1

				# Update bone keys, only catching the averages values
				bones_values[i] = bone_keys = bone_keys[:result_idx]

			# Compute slopes
			for key_idx, key_value in enumerate(bone_keys):
				key_value[2] = compute_slope(bone_keys, key_idx)


def bone_values_average(bone_values, is_last=False):
	average = [0, 0, 0]

	# Set time and check if is_first or is_last
	is_first = bone_values[0][0] == 0
	compute_time_average = not(is_first or is_last)
	if is_first:
		average[0] = 0
	elif is_last:
		average[0] = bone_values[-1][0]

	# Sum values
	for time, value, slope in bone_values:
		if compute_time_average:
			average[0] = average[0] + time
		average[1] = average[1] + value

	# Compute averages
	num_values = float(len(bone_values))
	if compute_time_average:
		average[0] /= num_values
	average[1] /= num_values

	return average


def compute_slope(bone_values, key_idx):
	slope = 0

	if key_idx != 0 and key_idx != len(bone_values) - 1:
		previous_time, previous_value, previous_slope = bone_values[key_idx - 1]
		next_time, next_value, next_slope = bone_values[key_idx + 1]
		slope = (next_value - previous_value) / (next_time - previous_time)

	return slope


########### Write animation ###########
def write_anim(bones_values, bones_settings, duration, file_path):
	euler_curve_tmpl = Template(EULER_CURVE_TEMPLATE)
	euler_curves_str = ''
	euler_keys_tmpl = Template(EULER_CURVE_KEY_TEMPLATE)
	euler_keys_str = ''
	editor_curve_tmpl = Template(EDITOR_CURVE_TEMPLATE)
	editor_curves_str = ''
	editor_keys_tmpl = Template(EDITOR_CURVE_KEY_TEMPLATE)
	editor_keys_str = ''
	curve_tmpl_values = {'CURVE': '', 'PATH': '', 'ATTR': 'localEulerAnglesRaw.z'}
	key_tmpl_values = {'TIME': 0, 'VALUE': 0, 'SLOPE': 0}

	for bone_idx, bone_values in enumerate(bones_values):
		if len(bone_values) != 0:
			for time, value, slope in bone_values:
				key_tmpl_values['TIME'] = '%.2f' % time
				key_tmpl_values['VALUE'] = '{x: 0, y: 0, z: ' + value.__str__() + '}'
				key_tmpl_values['SLOPE'] = '{x: 0, y: 0, z: ' + slope.__str__() + '}'
				euler_keys_str += euler_keys_tmpl.substitute(key_tmpl_values)
				key_tmpl_values['VALUE'] = value
				key_tmpl_values['SLOPE'] = slope
				editor_keys_str += editor_keys_tmpl.substitute(key_tmpl_values)

			curve_tmpl_values['PATH'] = bones_settings[bone_idx][3]
			curve_tmpl_values['CURVE'] = euler_keys_str
			euler_curves_str += euler_curve_tmpl.substitute(curve_tmpl_values)
			curve_tmpl_values['CURVE'] = editor_keys_str
			editor_curves_str += editor_curve_tmpl.substitute(curve_tmpl_values)
			euler_keys_str = ''
			editor_curves_str = ''

	file_tmpl = Template(ANIM_FILE_TEMPLATE)
	anim_name = os.path.splitext(os.path.basename(file_path))[0]
	tmpl_values = {'NAME': anim_name,
	               'EULER_CURVES': euler_curves_str,
	               'DURATION': duration,
	               'EDITOR_CURVES': editor_curves_str,
	               'ATTR': 'localEulerAnglesRaw.z'}
	file_content = file_tmpl.substitute(tmpl_values)

	os.makedirs(os.path.dirname(file_path), exist_ok=True)
	with open(file_path, "w") as out_file:
		out_file.write(file_content)


############################################ MAIN ############################################
def main(in_path, out_path, bones_settings):
	file_name = os.path.splitext(os.path.basename(in_path))[0]
	out_poses_path = os.path.join(out_path, file_name)
	out_poses_path = os.path.normpath(out_poses_path)

	if not os.path.exists(out_poses_path):
		detect_poses(in_path, out_poses_path)

	process_poses(out_poses_path, out_path, bones_settings, 0)


if __name__ == "__main__":
	IN_PATH = "E:/PROYECTOS/Pose2Anim/Input/Body.mp4"
	OUT_PATH = "E:/PROYECTOS/Pose2Anim/Pose2AnimUnity/Assets/Animations"
	BONES_SETTINGS = [(8, 1, -1, 'bone_1/bone_2', 0),
	                  (0, 15.5, 0, 'bone_1/bone_2/bone_3', 0),
	                  (2, 3, 0, 'bone_1/bone_2/bone_6', 180),
	                  (3, 4, 2, 'bone_1/bone_2/bone_6/bone_7', 0),
	                  (5, 6, 0, 'bone_1/bone_2/bone_4', 180),
	                  (6, 7, 4, 'bone_1/bone_2/bone_4/bone_5', 0),
	                  (9, 10, 0, 'bone_1/bone_8', 0),
	                  (10, 11, 6, 'bone_1/bone_8/bone_9', 0),
	                  (12, 13, 0, 'bone_1/bone_10', 0),
	                  (13, 14, 8, 'bone_1/bone_10/bone_11', 0)]

	main(IN_PATH, OUT_PATH, BONES_SETTINGS)
