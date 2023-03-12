"""
This file exemplifies the use of Video2Anim.
See more information in the README.md.
"""

from video2anim import Video2Anim

if __name__ == "__main__":
	# Required settings
	video_path = "Body.mp4"
	output_folder = "Animations"
	openpose_path = "OpenPose"
	bones_defs = [[8, 1, -1, 'bone_1/bone_2'],  # Backbone
	              [1, 0, 0, 'bone_1/bone_2/bone_3'],  # Head
	              [5, 6, 0, 'bone_1/bone_2/bone_4'],  # Left upper arm
	              [6, 7, 2, 'bone_1/bone_2/bone_4/bone_5'],  # Left lower arm
	              [2, 3, 0, 'bone_1/bone_2/bone_6'],  # Right upper arm
	              [3, 4, 4, 'bone_1/bone_2/bone_6/bone_7'],  # Right lower arm
	              [9, 10, 0, 'bone_1/bone_8'],  # Right upper leg
	              [10, 11, 6, 'bone_1/bone_8/bone_9'],  # Right lower leg
	              [12, 13, 0, 'bone_1/bone_10'],  # Left upper leg
	              [13, 14, 8, 'bone_1/bone_10/bone_11']]  # Left lower leg

	# Optional settings
	body_orientation = 90
	min_confidence = 0.6
	min_trembling_frequency = 7
	mlf_max_error_ratio = 0.1
	avg_keys_per_second = 0

	# Creation
	v2a = Video2Anim(video_path=video_path, output_folder=output_folder, openpose_path=openpose_path,
	                 bones_defs=bones_defs, body_orientation=body_orientation,
	                 min_confidence=min_confidence, min_trembling_freq=min_trembling_frequency,
	                 mlf_max_error_ratio=mlf_max_error_ratio, avg_keys_per_sec=avg_keys_per_second)

	# Execution
	v2a.run()