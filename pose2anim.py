import os
import json
import math
from string import Template

############################################ CONSTANTS ############################################
FRAME_RATE = 30
MIN_CONFIDENCE = 0.75
NUM_FRAMES_AVERAGE = 4
ZERO_DEGREE_OFFSET = 90

OPENPOSE_PATH = "OpenPose"
RESULTS_FOLDER_NAME = "Results"

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
  - curve:
      serializedVersion: 2
      m_Curve:
$EULER_CURVE      m_PreInfinity: 2    
      m_PostInfinity: 2
      m_RotationOrder: 4
    path: $PATH
  m_PositionCurves: []
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
  - curve:
      serializedVersion: 2
      m_Curve:
$EDITOR_CURVE      m_PreInfinity: 2
      m_PostInfinity: 2
      m_RotationOrder: 4
    attribute: $ATTR
    path: $PATH
    classID: 4
    script: {fileID: 0}
  m_HasGenericRootTransform: 0
  m_HasMotionFloatCurves: 0
  m_Events: []
'''

EULER_CURVE_TEMPLATE = '''      - serializedVersion: 3
        time: $TIME
        value: $VALUE
        inSlope: $SLOPE
        outSlope: $SLOPE
        tangentMode: 0
        weightedMode: 0
        inWeight: {x: 0, y: 0, z: 0.5}
        outWeight: {x: 0, y: 0, z: 0.5}
'''

EDITOR_CURVE_TEMPLATE = '''      - serializedVersion: 3
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
	command = f"start bin/OpenPoseDemo.exe --keypoint_scale 3 --video {in_path} --write_json {out_path}"
	print(command)
	stream = os.popen(command)
	stream.read()


###################### Process poses ######################
def process_poses(in_path, out_path, person_idx):
	bones_values = [[]] * len(BONES_SETTINGS)
	ini_time = -1
	time = 0
	num_frames = 0
	num_correct_frames = 0

	# Read bones values
	file_names = os.listdir(in_path)
	for file_name in file_names:
		if file_name.endswith(".json"):
			file_path = os.path.join(in_path, file_name)
			with open(file_path) as frame_file:
				frame_values = json.load(frame_file)
				time = num_correct_frames * (1 / FRAME_RATE)
				contains_data = compute_bones_values(frame_values, person_idx, time, bones_values)
				if contains_data and ini_time == -1:
					num_correct_frames += 1
				num_frames += 1

	# Post-process
	bones_values = postprocess_bones_values(bones_values)

	# Write animation
	file_name = os.path.splitext(os.path.basename(in_path))[0]
	file_path = os.path.join(out_path, f"{file_name}{person_idx}.anim")
	write_anim(bones_values, time, file_path)

	print(f"{num_correct_frames} / {num_frames}")


########### Compute bones values ###########
def compute_bones_values(frame_values, person_idx, time, bones_values):
	contains_data = False

	people = frame_values["people"]
	if len(people) > person_idx:
		keypoints = people[person_idx]["pose_keypoints_2d"]
		for i, bone_setting in enumerate(BONES_SETTINGS):
			ini = get_kp(keypoints, bone_setting[0])
			end = get_kp(keypoints, bone_setting[1])

			if ini and end:
				contains_data = True
				offset = kp_sub(end, ini)
				angle = math.atan2(offset[1], offset[0])
				angle = math.degrees(angle) - ZERO_DEGREE_OFFSET

				bones_values[i].append((time, angle))

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


########### Post-process bones values ###########
def postprocess_bones_values(bones_values):
	output = [[]] * len(BONES_SETTINGS)
	ini = 0
	end = NUM_FRAMES_AVERAGE
	for i, bone_values in enumerate(bones_values):
		num_bone_values = len(bone_values)
		# Compute average
		while end < num_bone_values:
			average = bone_values_average(bone_values[ini:end])
			output[i].append(average)
			ini += NUM_FRAMES_AVERAGE
			end += NUM_FRAMES_AVERAGE

		# Last average
		if ini < num_bone_values - 1:
			average = bone_values_average(bone_values[ini:num_bone_values])
			output[i].append(average)

		# Get slope
		for key_idx, values in enumerate(output[i]):
			values[2] = get_anim_slope(output[i], key_idx)

	return output


def bone_values_average(bone_values):
	num_values = float(len(bone_values))
	first_time = bone_values[0][0]
	average = [first_time, 0, 0]

	for time, value in bone_values:
		average[1] = average[1] + value
	average[1] /= num_values

	return average


def get_anim_slope(bone_values, key_idx):
	slope = 0

	if key_idx != 0 and key_idx != len(bone_values) - 1:
		previous_time, previous_value, previous_slope = bone_values[key_idx - 1]
		next_time, next_value, next_slope = bone_values[key_idx + 1]
		slope = (next_value - previous_value) / (next_time - previous_time)

	return slope


########### Write animation ###########
def write_anim(bones_values, duration, file_path):
	euler_curve_str = ''
	euler_tmpl = Template(EULER_CURVE_TEMPLATE)
	editor_curve_str = ''
	editor_tmpl = Template(EDITOR_CURVE_TEMPLATE)
	tmpl_values = {'TIME': 0, 'VALUE': 0, 'SLOPE': 0}

	for bone_values in bones_values:
		# TODO : Multiple bones
		for time, value, slope in bone_values:
			tmpl_values['TIME'] = '%.2f' % time
			tmpl_values['VALUE'] = '{x: 0, y: 0, z: ' + value.__str__() + '}'
			tmpl_values['SLOPE'] = '{x: 0, y: 0, z: ' + slope.__str__() + '}'
			euler_curve_str += euler_tmpl.substitute(tmpl_values)
			tmpl_values['VALUE'] = value
			tmpl_values['SLOPE'] = slope
			editor_curve_str += editor_tmpl.substitute(tmpl_values)

	file_tmpl = Template(ANIM_FILE_TEMPLATE)
	anim_name = os.path.splitext(os.path.basename(file_path))[0]
	tmpl_values = {'NAME': anim_name,
	               'EULER_CURVE': euler_curve_str,
	               'PATH': BONES_SETTINGS[0][3],  # TODO : Multiple bones
	               'DURATION': duration,
	               'EDITOR_CURVE': editor_curve_str,
	               'ATTR': 'localEulerAnglesRaw.z'}
	file_content = file_tmpl.substitute(tmpl_values)

	os.makedirs(os.path.dirname(file_path), exist_ok=True)
	with open(file_path, "w") as out_file:
		out_file.write(file_content)


############################################ MAIN ############################################
if __name__ == "__main__":
	IN_PATH = "E:/PROYECTOS/Pose2Anim/Input/Benet.mp4"
	OUT_PATH = "E:/PROYECTOS/1-UNITY/Pose2Anim/Assets/Cop"
	BONES_SETTINGS = [(0, 15.5, -1, 'bone_1/bone_2/bone_3')]

	file_name = os.path.splitext(os.path.basename(IN_PATH))[0]
	OUT_PATH = os.path.join(OUT_PATH, file_name)

	if not os.path.exists(OUT_PATH):
		detect_poses(IN_PATH, OUT_PATH)

	result_path = os.path.join(OUT_PATH, RESULTS_FOLDER_NAME)
	process_poses(OUT_PATH, result_path, 0)
