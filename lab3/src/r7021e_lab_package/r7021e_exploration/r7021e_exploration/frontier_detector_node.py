#!/usr/bin/env python3
# frontier_detector_node.py

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import numpy as np


class FrontierDetector(Node):
    def __init__(self):
        super().__init__('frontier_detector')

        self.map_sub = self.create_subscription(
            OccupancyGrid, 'map', self.map_callback, 1)

        self.map_pub = self.create_publisher(
            OccupancyGrid, 'frontiers', 1)

    def map_callback(self, msg: OccupancyGrid):
        if msg.info.width == 0 or msg.info.height == 0:
            return

        # Convert map data stored as a list to a 2D grid
        grid = np.array(msg.data, dtype=np.int16).reshape(
            (msg.info.height, msg.info.width))

        free = (grid >= 0) & (grid <= 50)
        unk = (grid < 0)

        h, w = grid.shape
        neigh_unknown = np.zeros_like(unk, dtype=bool)

        def acc_shift(dst, src, dy, dx):
            if dy == -1 and dx == -1:
                dst[:-1, :-1] |= src[1:, 1:]
            elif dy == -1 and dx == 0:
                dst[:-1, :] |= src[1:, :]
            elif dy == -1 and dx == 1:
                dst[:-1, 1:] |= src[1:, :-1]
            elif dy == 0 and dx == -1:
                dst[:, :-1] |= src[:, 1:]
            elif dy == 0 and dx == 1:
                dst[:, 1:] |= src[:, :-1]
            elif dy == 1 and dx == -1:
                dst[1:, :-1] |= src[:-1, 1:]
            elif dy == 1 and dx == 0:
                dst[1:, :] |= src[:-1, :]
            elif dy == 1 and dx == 1:
                dst[1:, 1:] |= src[:-1, :-1]

        # 4-neighborhood, creat mask using np
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            acc_shift(neigh_unknown, unk, dy, dx)

        frontier = free & neigh_unknown
        frontier_map = np.zeros_like(unk, dtype=np.int8)
        frontier_map[frontier] = 100

        # Publish frontier map
        out_msg = OccupancyGrid()
        out_msg.header = msg.header
        out_msg.info = msg.info
        out_msg.data = frontier_map.astype(np.int8).flatten().tolist()
        self.map_pub.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = FrontierDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
