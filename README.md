# Video2Anim
![BodyVsAnim2](https://user-images.githubusercontent.com/47823656/101023935-b8086080-3573-11eb-9f05-42624219e150.png)

This program makes use of [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) pose detection to transform a video into a 2D animation file in Unity's .anim format.
Also, it process the results to smooth the animation and is able to generate animations of different people from one video.
Requires download OpenPose and configure it (execute getModels.bat script) separately. Only tested with the Windows demo of version 1.6.0.

This project is free to use and modify with attribution.

# How to use
Currently, the only way to use Video2Anim is by code.
Check the [example](example.py) file for a simple sample or keep reading for a complete explanation.

First, install the requirements (using pip and [requirements.txt](requirements.txt)) and import the main class.
```python
from video2anim import Video2Anim
```

After, instance the class and pass the required arguments as keywords.
<br>The required arguments are:
* **video_path**: Path to the video file. Video format must be compatible with OpenPose, such as .mp4 or .avi.
* **anim_path**: Folder path to put the results. The results include the animation file and
    a folder with OpenPose's .json files.
* **openpose_path**: Path to the OpenPose folder. The folder must contain bin and models folders.
* **bones_defs**: List of bones for animation.
    <br>Every position must have the following values:
    1. Bone initial position in BODY_25 format.
    2. Bone end position in BODY_25 format.
    3. Index of the parent bone (-1 if don't has parent). Usually used for lower bones of arms and legs.
    <br>If a child bone if placed before the parent bone, the list will be copied and re-sorted.
    4. GameObject path in hierarchy. For example: bone1/bone2/bone3.
```python
video_path = "Videos/Jump.mp4"
anim_path = "Animations"
openpose_path = "OpenPose"
bones_defs = [[8, 1, -1, 'bone_1/bone_2'],                  # Backbone
              [1, 0, 0, 'bone_1/bone_2/bone_3'],            # Head
              [5, 6, 0, 'bone_1/bone_2/bone_4'],            # Left upper arm
              [6, 7, 2, 'bone_1/bone_2/bone_4/bone_5'],     # Left lower arm
              [2, 3, 0, 'bone_1/bone_2/bone_6'],            # Right upper arm
              [3, 4, 4, 'bone_1/bone_2/bone_6/bone_7'],     # Right lower arm
              [9, 10, 0, 'bone_1/bone_8'],                  # Right upper leg
              [10, 11, 6, 'bone_1/bone_8/bone_9'],          # Right lower leg
              [12, 13, 0, 'bone_1/bone_10'],                # Left upper leg
              [13, 14, 8, 'bone_1/bone_10/bone_11']]        # Left lower leg

v2a = Video2Anim(video_path=video_path, anim_path=anim_path, openpose_path=openpose_path,
	                      bones_defs=bones_defs)
```
An AssertionError will be raised if an incorrect parameter is passed.

Finally, the program can be executed using the **run** method, that receives the person index as optional argument (zero by default).
Also, you can vary the arguments previously defined passing them as keywords arguments.
The run method follows the next steps:
1. Assign the settings passed as keyword arguments.
2. Check if the required arguments (video_path, anim_path, openpose_path and bones_defs) are defined.
3. Execute OpenPose with the video selected if anim_path contains no previous results, putting the resulting .json files in a folder in anim_path with the same name as the video.
4. Read the poses of the specified person_idx from the .json files.
5. Process the poses to get a smooth animation.
6. Write the results in an .anim file in anim_path, using as name the name of the video concatenated with the person index.
```python
# Only one person
v2a.run()

# For N people
N = 5
for i in range(N):
    v2a.run(person_idx=i)
```
In the code above OpenPose will only run on the first call, using the calculated results for the next calls.

In addition, you can change more settings (and the previously defined) at **constructor**, **set_settings** and **run** methods as keywords.
<br>The additional (and not-required) parameters are:
* **body_orientation**: In which orientation is the body placed. By default is 90 degrees (vertical).
* **min_confidence**: Minimum confidence in a keypoint required for its use. In range [0, 1].
* **min_trembling_freq**: Used for trembling reduction.
    Minimum frequency of consecutive ups and downs required to delete such keypoints.    
    By default has a value of 7. A value of 0 disables the process.
    See the **Implementation** section for more information about trembling reduction.
* **mlf_max_error_ratio**: Multi-line Fitting maximum error ratio. In range [0, 1].
    The maximum permissible error in MLF in proportion to the difference between
    the minimum and maximum value of the bone animation.
    By default has a value of 0.1. A value of 0 disables the process.
    See the **Implementation** section for more information about Multi-line Fitting.
* **avg_keys_per_sec**: Average keypoints per second desired.
    If a bone has more keys during a second this, the amount will be reduced computing averages.
    It modifies fast movements as kicks or jumps, so is not recommended to use (and disabled by default) unless the previous processes return trembling results.
    It is not recommended to use it at the same time as the trembling reduction and MLF.
    By default has a value of 0, which disables the process.
    See the **Implementation** section for more information about keypoint averaging.
```python
# Multiple videos with different minimum confidences
videos = ["Idle.mp4", "Run.mp4", "Jump.mp4"]
confidences = [0.1, 0.6, 0.8]

# Define the required settings and min_trembling_freq at constructor
v2a = Video2Anim(video_path=video_path, anim_path=anim_path, openpose_path=openpose_path,
	                      bones_defs=bones_defs, min_trembling_freq=5)

# Compute first person animation of all the videos
for idx, video in enumerate(videos):
    v2a.run(video_path=video, min_confidence=confidences[idx])

# Change the minimum trembling frequency
v2a.set_settings(min_trembling_freq=10)

# Compute second person animation of all the videos
for idx, video in enumerate(videos):
    v2a.run(person_idx=1, video_path=video, min_confidence=confidences[idx])
```

# Implementation
The program is implemented in the [video2anim.py](video2anim.py) file as a namesake class (Video2Anim).
The class is divided into the following parts:
* **External use**: Methods that the programmer will use for execution (constructor, set_settings and run). Explained in the **How to use** section.
* **Error detection**: Settings checking. Specifically if all required paths exist and the bones definitions are correct.
* **Detect poses**: Execution of OpenPose using subprocess.run with the corresponding command.
    The working directory will be moved to openpose_path during the process.
    The program uses the OpenPoseDemo.exe of the bin folder.
* **Read poses**: Reading of the .json files generated by OpenPose.
    The animation data of each bone (timestamp and angle pairs) is obtained only if the required keypoints have the minimum confidence.
    If a bone has a parent, it must be detected. Also, its angle will be subtracted from that of the child, having a relative motion.
* **Process animation**: Modifies the poses data to obtain a smoother and simpler animation.
    Also, it computes the slopes of each keypoint, necessary for the animation file.    
* **Write animation**: Writing of the .anim file using the animation data and string templates.
    The generated file is in YAML format, and defines the animation as euler_curves and editor_curves (both with the same data).
    The only modified attribute of the bones is the rotation of the z-axis, because it is a 2D animation.

All the methods have its corresponding docstring.
<br>The process animation is the more complex and will therefore be explained in more depth in the next subsection.

## Process animation
If the poses detected by OpenPose are directly translated to an animation the result will be shaky and contain a lot of redundant information.
To solve this, each set of keypoints of a bone is treated with the following processes:
* **Trembling reduction**: Deletes the high frequency movements.
* **Multi-Line Fitting**: Simplifies the animation keeping only the most defining keypoints.
* **Keypoints averaging**: Smooths the animation curve.
    It modifies fast movements as kicks or jumps, so is not recommended to use (and disabled by default) unless the previous processes return trembling results.
    It is not recommended to use it at the same time as the trembling reduction and MLF.
* **Slope computation**: Computes the slope value of each keypoint based on its surrounding keypoints.
    This process is always necessary and determines the animation curve.

The execution of first three processes can be affected or disabled by the settings, as it's described in the **How to use** section.
<br>All the processes are described in more detail below.

### Trembling reduction
The poses returned by OpenPose have usually little variations in its positions, resulting in trembling animations.
It has been observed that many of these variations are consecutive peaks and valleys in the values, which can be seen as high frequency waves.
This high frequencies could be avoided using a not-uniform fast fourier transform (and inverse), but that can be too time consuming and complex.
Instead, the trembling is reduced deleting consecutive peaks and valleys that have a frequency greater of equal to the **min_trembling_freq** setting.
The process is avoided if **min_trembling_freq** is 0.

### Multi-Line Fitting
Multi-Line Fitting is an algorithm specifically created for this project that iteratively approximates the keypoints set detecting its most defining keypoints.
Because it uses the existing keypoints in its result, it does not affect quick movements like kicks and jumps (as keypoints averaging does).
Following, its procedure will be described.

First, is necessary compute an maximum tolerable error for the approximation.
Because the ranges of values can vary, this maximum must be based on the keypoints set.
For that, the minimum and maximum values of the keypoints are calculated.
Then, the maximum tolerable error is described as: (maximum - minimum) * **mlf_max_error_ratio**.
**mlf_max_error_ratio** is a setting that has a value between 0 and 1. If the value is 0 all MLF is skipped.
![MLF_Kick_MaxMin](https://user-images.githubusercontent.com/47823656/101020417-c86a0c80-356e-11eb-97a2-d430b7de3b71.png)

Then, the approximation/result is initialized only with the initial and end keypoints.
The consequent flat line from start to finish is probably an oversimplified approximation, so the errors will be calculated.
For each keypoint, the projection on that line is computed by linear interpolation.
The distance/difference between the keypoint and its projection is interpreted as the error.
Any error minor than the maximum tolerable error is ignored.
Using this information, the maximum error keypoint is found to add to the approximation.
![MLF_Kick_Start](https://user-images.githubusercontent.com/47823656/101021358-221f0680-3570-11eb-9c0f-6653033f88e9.png)

Because the approximation has changed, the errors must be recalculated.
But now the maximum error is computed separately for the two parts and compared later, selecting the maximum error of both.
The nonmaximum error is stored, because that part of the approximation will not change now and can therefore be used in future iterations.
![MLF_Kick_Continue](https://user-images.githubusercontent.com/47823656/101021894-d91b8200-3570-11eb-9560-9b6386853102.png)

This process is repeated until there are no more errors greater than the tolerable, searching in smaller sets of keypoints each time.
![MLF_Kick_Final](https://user-images.githubusercontent.com/47823656/101021926-ec2e5200-3570-11eb-8fa1-456a18069373.png)

The cost of this algorithm is between *O(n)* and *O(n^2)*.

### Keypoints averaging
An alternative simplification process to MLF.
This procedure checks if during a second exists more than **avg_keys_per_sec** keypoints.
If **avg_keys_per_sec** setting has a value of 0 (the default), keypoints averaging is skipped.
Instead, if an excessive amount of keypoints are found in a second, the average is used to reduce the count to the desired.
For example, if a second contains 10 keypoints and **avg_keys_per_sec** has a value of 5, every 2 keypoints will substituted by their average (of timestamp and value).
If the number of keypoints is not divisible, the last will be an average of fewer keypoints.
This process does not affect to initial and final keypoints.

The problem of this algorithm is that, if the animation has fast movements (like kicks or jumps) the corresponding peak keypoints can be smoothed out by the averaging.

### Slope computation
In Unity (and other game engines), transitions between the keypoints in an animation do not run as linear interpolations.
Each keypoint has a slope value that allows to transform the animation into a smooth curve.
To calculate it, has been seen that a good slope is the one between the previous keypoint and the next.
For the first and the last keypoint the slope value is zero.
Therefore, this process compute the slope for each keypoint, transforming the structure from [timestamp, angle] to [timestamp, angle, slope].

# Future improvements
In the future, the following improvements will be implemented:
* **User Interface**
* **Support for parallelization**