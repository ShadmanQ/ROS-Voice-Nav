#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from speech_recog_uc.msg import SpeechResult
from commands import allCommands


class State():

    def __init__(self):
        rospy.loginfo("listener has started")
        self.commands_module = allCommands()

    def callback(self, data):
      #  rospy.loginfo((data.result))
        self.parse(data.result)

    def parse(self, command):
        print(command)
        if "move" in command:
            self.commands_module.command_move(command)
        if "turn" in command:
            self.commands_module.command_move(command, mode='t')
        if "locate" in command:
            self.commands_module.command_locate(command)
        if "retrieve" in command:
            self.commands_module.command_retrieve(command)

    def listener(self):

        # In ROS, nodes are uniquely named. If two nodes with the same
        # name are launched, the previous one is kicked off. The
        # anonymous=True flag means that rospy will choose a unique
        # name for our 'listener' node so that multiple listeners can
        # run simultaneously.
        rospy.init_node('listener', anonymous=True)

        rospy.Subscriber("speech_recog_uc/words", SpeechResult, self.callback)

        # spin() simply keeps python from exiting until this node is stopped
        rospy.spin()


if __name__ == '__main__':
    state = State()
    state.listener()
