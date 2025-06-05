#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RosCommunicator 用於管理各種 ROS 主題之間的通信，用於處理映像、IMU 和偵測資料。
初始化預先定義主題的訂閱者和發布者，儲存最新收到的訊息，並提供檢索和發布資料的方法。

Subscribers:
    - /camera/image/compressed (sensor_msgs/CompressedImage): RGB camera images (compressed)
    - /imu/data (sensor_msgs/Imu): IMU sensor data
    - /camera/depth/compressed (sensor_msgs/CompressedImage): Depth images (compressed)
    - /camera/depth/image_raw (sensor_msgs/Image): Raw depth images
    - /target_label (std_msgs/String): Target label information

Publishers:
    - /yolo/detection/compressed (sensor_msgs/CompressedImage): YOLO detection images (compressed)
    - /yolo/detection/position (geometry_msgs/PointStamped): Detected object position
    - /yolo/object/offset (std_msgs/String): 目標物體和畫面中心的 3D 空間偏移
    - /yolo/detection/status (std_msgs/Bool): Detection status

Attributes:
    subscriber_dict (dict): Configuration for all subscribers (topic, message type, callback)
    publisher_dict (dict): Configuration for all publishers (topic, message type)
    latest_data (dict): Stores the latest message for each subscribed topic
    publisher_instances (dict): Stores publisher instances for each topic

Methods:
    _image_sub_callback(msg): Callback for RGB image subscriber
    _imu_sub_callback(msg): Callback for IMU data subscriber
    _depth_image_sub_callback(msg): Callback for raw depth image subscriber
    _depth_image_compress_sub_callback(msg): Callback for compressed depth image subscriber
    _target_label_sub_callback(msg): Callback for target label subscriber
    get_latest_data(key): Returns the latest message for the given key
    publish_data(key, data): Publishes data to the topic associated with the given key
"""

from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Imu, Image
from std_msgs.msg import String, Bool, Float32MultiArray
from geometry_msgs.msg import PointStamped


class RosCommunicator(Node):
    def __init__(self):
        super().__init__("RosCommunicator")

        # --- Subscriber and Publisher Initialization ---
        self.subscriber_dict = {
            "rgb_compress": {
                "topic": "/camera/image/compressed",
                "msg_type": CompressedImage,
                "callback": self._image_sub_callback,
            },
            "imu": {
                "topic": "/imu/data",
                "msg_type": Imu,
                "callback": self._imu_sub_callback,
            },
            "depth_image_compress": {
                "topic": "/camera/depth/compressed",
                "msg_type": CompressedImage,
                "callback": self._depth_image_compress_sub_callback,
            },
            "depth_image": {
                "topic": "/camera/depth/image_raw",
                "msg_type": Image,
                "callback": self._depth_image_sub_callback,
            },
            "target_label": {
                "topic": "/target_label",
                "msg_type": String,
                "callback": self._target_label_sub_callback,
            },
        }

        self.publisher_dict = {
            "yolo_image": {
                "topic": "/yolo/detection/compressed",
                "msg_type": CompressedImage,
            },
            "point": {
                "topic": "/yolo/detection/position",
                "msg_type": PointStamped,
            },
            "object_offset": {
                "topic": "/yolo/object/offset",
                "msg_type": String,
            },
            "detection_status": {
                "topic": "/yolo/detection/status",
                "msg_type": Bool,
            },
            "yolo_target_info": {
                "topic": "/yolo/target_info",
                "msg_type": Float32MultiArray,
            },
            "camera_x_multi_depth_values": {
                "topic": "/camera/x_multi_depth_values",
                "msg_type": Float32MultiArray,
            },
        }

        # Initialize Subscribers
        self.latest_data = {}
        for key, sub in self.subscriber_dict.items():
            self.latest_data[key] = None
            msg_type = sub["msg_type"]
            topic = sub["topic"]
            callback = sub["callback"]
            self.create_subscription(msg_type, topic, callback, 10)

        # Initialize Publishers
        self.publisher_instances = {}
        for key, pub in self.publisher_dict.items():
            self.publisher_instances[key] = self.create_publisher(
                pub["msg_type"], pub["topic"], 10
            )

    # --- Callback Functions ---
    def _image_sub_callback(self, msg):
        self.latest_data["rgb_compress"] = msg

    def _imu_sub_callback(self, msg):
        self.latest_data["imu"] = msg

    def _depth_image_sub_callback(self, msg):
        self.latest_data["depth_image"] = msg

    def _depth_image_compress_sub_callback(self, msg):
        self.latest_data["depth_image_compress"] = msg

    def _target_label_sub_callback(self, msg):
        self.latest_data["target_label"] = msg

    # --- Getter Functions ---
    def get_latest_data(self, key):
        return self.latest_data.get(key)

    # --- Publisher Functions ---
    def publish_data(self, key, data):
        """
        Publishes data to the topic associated with the given key.
        範例: (in BoundingBoxVisualizer) self.ros_communicator.publish_data("yolo_image", ros_image)
        "yolo_image" 對應到 "/yolo/detection/compressed" topic 和 "sensor_msgs/CompressedImage" msg type。
        """
        try:
            publisher = self.publisher_instances.get(key)
            if publisher:
                publisher.publish(data)
            else:
                self.get_logger().error(f"No publisher found for key: {key}")
        except Exception as e:
            self.get_logger().error(f"Could not publish data for {key}: {e}")
