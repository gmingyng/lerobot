import cv2

def display_all_cameras():
    """
    查找所有可用的摄像头并同时显示它们的视频流。
    """
    # 先用上面的方法找到所有可用的摄像头索引
    camera_indexes = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            camera_indexes.append(i)
            cap.release()

    if not camera_indexes:
        print("未找到任何摄像头。")
        return

    print(f"找到 {len(camera_indexes)} 个摄像头: {camera_indexes}")

    # 为每个摄像头创建一个捕捉对象
    caps = [cv2.VideoCapture(i) for i in camera_indexes]

    while True:
        frames = []
        # 从每个摄像头读取一帧
        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            if ret:
                # 在帧上绘制索引号，方便识别
                cv2.putText(frame, f"Index: {camera_indexes[i]}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # 显示在不同的窗口中
                cv2.imshow(f"Camera {camera_indexes[i]}", frame)

        # 按 'q' 键退出
        if cv2.waitKey(1) == ord('q'):
            break

    # 释放所有资源
    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    display_all_cameras()
