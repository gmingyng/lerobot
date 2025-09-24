import cv2

# 根据你的 v4l2-ctl 输出，这些是可能的 ID
camera_indices = [2, 3, 4, 5, 6, 7] 

caps = {}

# 尝试初始化所有相机
for index in camera_indices:
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        print(f"成功打开相机 ID: {index}")
        caps[index] = cap
    else:
        print(f"无法打开相机 ID: {index}")

if not caps:
    print("错误：一个相机也打不开！请检查驱动或权限。")
else:
    try:
        while True:
            # 循环读取并显示每一路视频流
            for index, cap in caps.items():
                ret, frame = cap.read()
                if ret:
                    # 在窗口标题栏显示 ID
                    cv2.imshow(f'Camera ID: {index}', frame)

            # 按 'q' 键退出所有窗口
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # 释放所有资源
        for cap in caps.values():
            cap.release()
        cv2.destroyAllWindows()
        print("程序已退出。")
