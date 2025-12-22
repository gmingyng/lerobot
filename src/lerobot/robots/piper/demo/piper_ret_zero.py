import time
from piper_sdk import *

# 测试代码
if __name__ == "__main__":
    piper = C_PiperInterface_V2()
    piper.ConnectPort()
    time.sleep(0.5)

    piper.EnableArm()
    time.sleep(0.5)
    piper.MotionCtrl_2(0x01, 0x01, 20, 0x00)
    Joint_1=3826
    Joint_2=-1896
    Joint_3=-2069
    Joint_4=34321
    Joint_5=14022
    Joint_6=114063

    # piper.JointCtrl(Joint_1, Joint_2, Joint_3, Joint_4, Joint_5, Joint_6)
    try:
        # 2. 将所有可能被中断的操作都放在 try 块中
        # 这是一个无限循环，模拟机器人持续运行的主程序
        while True:
            print(piper.GetArmJointMsgs())
            print(piper.GetArmGripperMsgs())
            # print(piper.GetArmJointCtrl())
            # print(piper.GetArmGripperCtrl())
            time.sleep(0.005)

    except KeyboardInterrupt:
        # 3. (可选但推荐) 捕获 KeyboardInterrupt 异常
        # 这样可以在程序退出前打印一个友好的提示信息
        print("\n捕获到键盘中断 (Ctrl+C)... 准备退出。")
        piper.DisableArm()
        time.sleep(1)
        piper.DisconnectPort()

    finally:
        # 4. 无论上面发生了什么，这里的代码都将被执行！
        # 这是放置清理代码最安全的地方。
        print("--- 进入 finally 清理环节 ---")
        piper.DisableArm()
        time.sleep(1)
        piper.DisconnectPort()
