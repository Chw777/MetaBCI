import time
def control_robot(self, label):
    """控制机械手运动"""
    # 确保串口已连接
    if label == 0 and self.serial_left:  # 左手
        try:
            # 机械手控制命令（根据实际协议调整）
            self.serial_left.write(b'A3')  # 初始化命令
            time.sleep(1)
            self.serial_left.write(b'G5')  # 运动命令
            time.sleep(4.5)
            self.serial_left.write(b'G5')  # 复位命令
            time.sleep(2)
            print("✅ 左手机械手运动完成")
        except Exception as e:
            print(f"❌ 左手机械手控制失败: {e}")

    elif label == 1 and self.serial_right:  # 右手
        try:
            self.serial_right.write(b'A3')  # 初始化命令
            time.sleep(1)
            self.serial_right.write(b'G5')  # 运动命令
            time.sleep(4.5)
            self.serial_right.write(b'G5')  # 复位命令
            time.sleep(2)
            print("✅ 右手机械手运动完成")
        except Exception as e:
            print(f"❌ 右手机械手控制失败: {e}")