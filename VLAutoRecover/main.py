import rclpy

class VLANode(rclpy.node.Node):
    def __init__(self):
        super().__init__('vla_node')
        self.create_timer(0.02, self.step)

    def step(self):
        # VLA 処理
        pass


class ADNode(rclpy.node.Node):
    def __init__(self):
        super().__init__('vla_node')
        self.create_timer(0.02, self.step)

    def step(self):
        # 異常検知 & VLM 処理
        pass

def main():
    rclpy.init()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(ADNode())
    executor.add_node(VLANode())
    executor.spin()

if __name__ == "__main__":
    main()