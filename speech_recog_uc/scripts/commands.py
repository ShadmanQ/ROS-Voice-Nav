import rospy
import cv2 as cv
from moveit_msgs.msg import MoveItErrorCodes
from moveit_python import MoveGroupInterface, PlanningSceneInterface
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion, Twist, Image
from nobridge import msg_to_RGB, RGB_to_msg, msg_to_depth_f32, depth_f32_to_msg
import tf_conversions
import tf2_ros
import darknetrgbanddepth


class allCommands():
    def __init__(self):
        rospy.loginfo("Commands have started")
        self.image_Sub = rospy.Subscriber("/image_bbox", Image, self.callback)
        self.DNet = darknetrgbanddepth()

    def command_move(self, input, mode='m'):
        directions = {
            'forward': (1, 0, 0, 0),
            'backward': (-1, 0, 0, 0),
            'left': (0, 0, 0, -1),
            'right': (0, 0, 0, 1)
        }
        rospy.loginfo("Move or turn detected, now executing....")
        number = 0

        twist = Twist()
        twist.linear.x = 0
        twist.linear.y = 0
        twist.linear.z = 0
        twist.angular.x = 0
        twist.angular.y = 0
        twist.angular.z = 0

        if mode == 't':
            print("turn mode is on")
            for element in input.split():
                if element == 'left' or element == 'right':
                    if (element == "left"):
                        print("turning left")
                        twist.angular.y = 1
                    if (element == "right"):
                        print("turning right")
                        twist.angular.y = -1

                    print(directions[element])
                    break

        for element in input.split():
            if element.isnumeric():
             #   rospy.loginfo(element)
                number = int(element)
        pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)

        if ("forward" in input):
            twist.linear.y = 1
        if ("left" in input):
            twist.angular.y = -1
        if ("right" in input):
            twist.angular.y = 1
        print(twist)
        # work that out in a second
        pub.publish(twist)

    def command_locate(self, input=None, img=None):

        img_d32 = msg_to_depth_f32(img)
        outs = self.DNet.net.forward(self.DNet.get_Outputs_Names())
        self.DNet.postprocess(img, outs, img_d32)

        print("Now finding this")
        twist = Twist()
        twist.linear.x = 0; twist.linear.y = -1; twist.linear.z = 0
        twist.angular.x = 0; twist.angular.y = 0; twist.angular.z = 0
        print(twist)
        # work that out in a second
        pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        pub.publish(twist)

    def command_retrieve(self, input):

        if len(input.split()) > 1 and input.split()[1] == "Cube":
            self.command_locate(input)
            print("I will now go and retrieve the " + input.split()[1])
            self.command_locate(input)

            rospy.loginfo("I will attempt to grab this object")

            # Define ground plane
            # This creates objects in the planning scene that mimic the ground
            # If these were not in place gripper could hit the ground

            move_group = MoveGroupInterface("arm_with_torso", "base_link")
            planning_scene = PlanningSceneInterface("base_link")
            planning_scene.removeCollisionObject("my_front_ground")
            planning_scene.removeCollisionObject("my_back_ground")
            planning_scene.removeCollisionObject("my_right_ground")
            planning_scene.removeCollisionObject("my_left_ground")
            planning_scene.addCube("my_front_ground", 2, 1.1, 0.0, -1.0)
            planning_scene.addCube("my_back_ground", 2, -1.2, 0.0, -1.0)
            planning_scene.addCube("my_left_ground", 2, 0.0, 1.2, -1.0)
            planning_scene.addCube("my_right_ground", 2, 0.0, -1.2, -1.0)
            # Construct a "pose_stamped" message as required by moveToPose
            gripper_pose_stamped = PoseStamped()
            gripper_pose_stamped.header.frame_id = 'base_link'

            while not rospy.is_shutdown():
                for pose in gripper_poses:
                    # Finish building the Pose_stamped message
                    # If the message stamp is not current it could be ignored
                    gripper_pose_stamped.header.stamp = rospy.Time.now()
                    # Set the message pose
                    gripper_pose_stamped.pose = pose

                    # Move gripper frame to the pose specified
                    move_group.moveToPose(gripper_pose_stamped, gripper_frame)
                    result = move_group.get_move_action().get_result()

                    if result:
                        # Checking the MoveItErrorCode
                        if result.error_code.val == MoveItErrorCodes.SUCCESS:
                            rospy.loginfo("Hello there!")
                        else:
                            # If you get to this point please search for:
                            # moveit_msgs/MoveItErrorCodes.msg
                            rospy.logerr("Arm goal in state: %s",
                                         move_group.get_move_action().get_state())
                    else:
                        rospy.logerr("MoveIt! failure no result returned.")

            # This stops all arm movement goals
            # It should be called when a program is exiting so movement stops
            move_group.get_move_action().cancel_all_goals()

    def callback(self, image):
        cv.imshow(msg_to_RGB(image))
        command_locate(img=image)
