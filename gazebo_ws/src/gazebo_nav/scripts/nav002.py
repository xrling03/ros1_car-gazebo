#! /usr/bin/env python3
# -*- coding: utf-8 -*-
from time import sleep
import rospy
import actionlib
import os
from actionlib_msgs.msg import *
from geometry_msgs.msg import Pose, PoseWithCovarianceStamped, PoseStamped, Quaternion, Twist
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from std_msgs.msg import Int8MultiArray, Int8, Int32
import multiprocessing as mp
import io
import sys
#import pyttsx3
import io
import sys
# import recongize
# from playsound import playsound

# 


global broadcast_word
 

#rospy.init_node('ucar_nav')  # 创建了一个节点
#S_detect = rospy.Publisher("S_detect",Int8,queue_size=1)

#speed = ros
# bc_flag = 0
# plant_room = []
# fruit_num = []
det_res = []
#是否到达
arrive = 0

    

    
    

# 发布目标点函数
def move():

    # 设置目标点
    #if arrive == 1:
    send_goal(0)
    sleep(2)
    # print("目标点1成功")
    # terrorist_num = recongize.recongize_terrorist()
    # print("恐怖分子编号：{}".format(terrorist_num))
    # if terrorist_num == 3:
    #     playsound("/home/iflytek/ucar_ws/src/ucar_nav/kongbu_1.mp3")
    # elif terrorist_num == 4:
    #     playsound("/home/iflytek/ucar_ws/src/ucar_nav/kongbu_2.mp3")
    # else:
    #     playsound("/home/iflytek/ucar_ws/src/ucar_nav/kongbu_3.mp3")

    # prop_num = recongize.calculate_num(terrorist_num)
    # print("待拿物品编号：{}".format(prop_num))
    send_goal(1)
    # print("目标点2成功")
    
    sleep(2)
    send_goal(2)
    
    # sleep(5) 
    # send_goal(3)
    # playsound("/home/iflytek/ucar_ws/src/ucar_nav/jijiubao.mp3")
    # sleep(1)
    # send_goal(4) 

    # #开启多线程 视觉模块
    # fg = [0]
    # recongize.recongize_prop(prop_num,fg)
    # if terrorist_num == 3:
    #     playsound("/home/iflytek/ucar_ws/src/ucar_nav/jingun.mp3")
    # elif terrorist_num == 4:
    #     playsound("/home/iflytek/ucar_ws/src/ucar_nav/fangdanyi.mp3")
    # else:
    #     playsound("/home/iflytek/ucar_ws/src/ucar_nav/cuileiwasi.mp3")
    
    # sleep(1)
    # send_goal(5) 
    
    # send_goal(8) 
    # #巡线多线程
    # send_goal(6) 
    # recongize.line_follow()
    # playsound("/home/iflytek/ucar_ws/src/ucar_nav/renzhijiuyuan.mp3")
    # # send_goal(7) 



def send_goal(num):
    #初始化坐标点
    locations =[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0]]

    # 任务点1 ！！！！！！！！！！！！！！！！！！！！！！！！！
    locations[0][0] = rospy.get_param("~point_1_x", 3.736)
    locations[0][1] = rospy.get_param("~point_1_y", 0.973)
    locations[0][2] = rospy.get_param("~point_1_z", 0.712)
    locations[0][3] = rospy.get_param("~point_1_w", 0.702)

    # 任务点2
    locations[1][0] = rospy.get_param("~point_1_x", 2.319)
    locations[1][1] = rospy.get_param("~point_1_y", 1.464)
    locations[1][2] = rospy.get_param("~point_1_z",  0.709)
    locations[1][3] = rospy.get_param("~point_1_w", 0.705)
        # 任务点3
    locations[2][0] = rospy.get_param("~point_1_x", 0.698)
    locations[2][1] = rospy.get_param("~point_1_y", 1.109)
    locations[2][2] = rospy.get_param("~point_1_z", 0.701)
    locations[2][3] = rospy.get_param("~point_1_w", 0.713)
    #rospy.Subscriber("angle", Int32, vo_start)

    locations[3][0] = rospy.get_param("~point_1_x", 3.049)
    locations[3][1] = rospy.get_param("~point_1_y", 4.007)
    locations[3][2] = rospy.get_param("~point_1_z",  0.703)
    locations[3][3] = rospy.get_param("~point_1_w", 0.712)

    locations[4][0] = rospy.get_param("~point_1_x", -1.803)
    locations[4][1] = rospy.get_param("~point_1_y", 0.381)
    locations[4][2] = rospy.get_param("~point_1_z", 0.721)
    locations[4][3] = rospy.get_param("~point_1_w", 0.693) 

    locations[5][0] = rospy.get_param("~point_1_x", -2.076)
    locations[5][1] = rospy.get_param("~point_1_y", 0.061)
    locations[5][2] = rospy.get_param("~point_1_z", 0.002)
    locations[5][3] = rospy.get_param("~point_1_w", 1.000)

    locations[8][0] = rospy.get_param("~point_1_x", -0.027)
    locations[8][1] = rospy.get_param("~point_1_y", -0.025)
    locations[8][2] = rospy.get_param("~point_1_z", 0.002)
    locations[8][3] = rospy.get_param("~point_1_w", 1.000) 
    #Position(-0.097, -0.345, 0.000), Orientation(0.000, 0.000, -0.697, 0.717) = Angle: -1.543
    #巡线
    locations[6][0] = rospy.get_param("~point_1_x", -0.097)
    locations[6][1] = rospy.get_param("~point_1_y", -0.345)
    locations[6][2] = rospy.get_param("~point_1_z", -0.710)
    locations[6][3] = rospy.get_param("~point_1_w", 0.704) 

    #slam
    # locations[6][0] = rospy.get_param("~point_1_x", -0.023)
    # locations[6][1] = rospy.get_param("~point_1_y", -0.018)
    # locations[6][2] = rospy.get_param("~point_1_z", 0.009)
    # locations[6][3] = rospy.get_param("~point_1_w", 1.000) 

    locations[7][0] = rospy.get_param("~point_1_x", 2.026)
    locations[7][1] = rospy.get_param("~point_1_y", 0.031)
    locations[7][2] = rospy.get_param("~point_1_z", -0.008)
    locations[7][3] = rospy.get_param("~point_1_w", 1.000)
    # 调用发布目标点函数
# 订阅move_base服务
    move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)
    print("Waiting for move_base action server...")
    # 等待服务
    while move_base.wait_for_server(rospy.Duration(5)) == 0:
        print("Connected to move base server")
    print("准备发布目标点")
    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = 'map'
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose.position.x = locations[num][0]
    goal.target_pose.pose.position.y = locations[num][1]
    goal.target_pose.pose.position.z = 0
    goal.target_pose.pose.orientation.x = 0
    goal.target_pose.pose.orientation.y = 0
    goal.target_pose.pose.orientation.z = locations[num][2]
    goal.target_pose.pose.orientation.w = locations[num][3]
    
    # 前往下一个目标
    
    
    #sleep(0.3)
    #S_detect.publish(num)
    #sleep(0.3)

    move_base.send_goal(goal)

    # 设置一个运行限制时间
    finished_within_time = move_base.wait_for_result(rospy.Duration(500))


    # 查看是否成功到达
    if not finished_within_time:
        move_base.cancel_goal()
        #rospy.loginfo("Timed out achieving goal")
    else:
        state = move_base.get_state()
        # if state == GoalStatus.SUCCEEDED:
            #rospy.loginfo("Goal succeeded!")
            # if num == 4:
            #     play_mp3("file1 我已取到急救包")

            # elif num == 5 and speed == 0 :
            #     play_mp3("boardcast_word")

            # elif num == 8:
            #     play_mp3("file5 已完成人质营救工作，请快速增派支援进行人质转运.mp3")

        #else:
            #rospy.loginfo("Goal failed！")  
    # sleep(0.2)
    # S_detect.publish(num)
    # sleep(0.2)

                    


if __name__ == '__main__':
    rospy.init_node('ucar_nav', anonymous=True)  # 创建了一个节点
    # from rknnlite.api import RKNNLite
    print("import成功")
    move()

