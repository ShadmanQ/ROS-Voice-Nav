# ROS Voice Nav

A project designed to take advantage of Google's Speech-to-Text API and make a Fetch Robotics Mobile Manipulator perform basic commands.

A large portion of this project relies on Pedro Oliviera's speech_recog_uc ROS Package. [Click here](https://github.com/jopedroliveira/speech_recog_uc) for more information and detailed set up on the project. Basic instructions are as follows

1. extract the contents of this package to your catkin_ws
2. Create a Google Cloud Platform account and create a new Project
3. Enable Google Cloud's Speech-to-Text API and obtain the credentials json file.
4. Save and name it something convenient.

In a terminal do the following
- cd ~/catkin_ws
- catkin_make
- source devel/setup.bash
- roscore

To start the speech recognition node enter rosrun speech_recog_uc speech_recog_uc node. Then to start the robot manipulator, enter rosrun speech_recog_uc listener.py
