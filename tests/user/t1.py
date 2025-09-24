import cv2
import time

# --- 参数设置 ---
# 根据 v4l2-ctl 的输出 /dev/video4, 对应的索引通常是 4
CAMERA_INDEX = 4 

# 从 v4l2-ctl 输出中选择一个支持的分辨率和帧率
# 例如: Size: Discrete 640x480, Interval: Discrete 0.033s (30.000 fps)
WIDTH = 640
HEIGHT = 480
FPS = 30

# 摄像头支持的像素格式 'YUYV'
# 我们需要将其转换为OpenCV的FOURCC代码
FOURCC = cv2.VideoWriter_fourcc(*'YUYV')

# --- 主程序 ---
def main():
    # 1. 初始化摄像头
    # 使用 cv2.CAP_V4L2 可以强制使用 V4L2 后端，在Linux下更稳定
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)

    # 2. 检查摄像头是否成功打开
    if not cap.isOpened():
        print(f"错误: 无法打开摄像头索引 {CAMERA_INDEX}")
        return

    # 3. 设置摄像头的参数
    cap.set(cv2.CAP_PROP_FOURCC, FOURCC)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    # 4. 验证实际应用的参数 (摄像头可能不会完全接受设置)
    actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join([chr((actual_fourcc >> 8 * i) & 0xFF) for i in range(4)])
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    print("--- 摄像头设置 ---")
    print(f"请求 -> 格式: YUYV, 尺寸: {WIDTH}x{HEIGHT}, 帧率: {FPS}")
    print(f"实际 -> 格式: {fourcc_str}, 尺寸: {actual_width}x{actual_height}, 帧率: {actual_fps:.2f}")
    print("--------------------")

    # 5. 循环读取并显示视频帧
    while True:
        # cap.read() 返回一个元组 (是否成功读取, 图像帧)
        ret, frame = cap.read()

        # 如果读取失败 (例如摄像头断开), 则退出循环
        if not ret:
            print("错误: 无法接收视频帧。退出...")
            break

        # 显示图像帧
        cv2.imshow('Camera Feed (Press "q" to exit)', frame)

        # 等待按键，如果按下 'q' 键则退出循环
        # cv2.waitKey(1) 表示等待 1 毫秒
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 6. 释放资源
    print("正在释放资源...")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
