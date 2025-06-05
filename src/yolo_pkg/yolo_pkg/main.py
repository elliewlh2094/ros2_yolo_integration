#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.executors import MultiThreadedExecutor
from yolo_pkg.ros_communicator import RosCommunicator
from yolo_pkg.image_processor import ImageProcessor
from yolo_pkg.yolo_depth_extractor import YoloDepthExtractor
from yolo_pkg.yolo_bounding_box import YoloBoundingBox
from yolo_pkg.boundingbox_visaulizer import BoundingBoxVisualizer
from yolo_pkg.camera_geometry import CameraGeometry
from yolo_pkg.darth_vader_detect import YoloDetectionNode

import threading
import json
from std_msgs.msg import String, Float32MultiArray
from yolo_pkg.load_params import LoadParams


def _init_ros_node():
    """
    Initialize the ROS 2 node with MultiThreadedExecutor for efficient handling of multiple subscribers.
    """
    rclpy.init()
    node = RosCommunicator()  # Initialize the ROS node
    """MultiThreadedExecutor 支援同時多個 callback 並行執行，這對於同時處理多種 sensor input 或高頻訊息的場景很有用，也能提升效能"""
    executor = MultiThreadedExecutor()  # Use MultiThreadedExecutor
    executor.add_node(node)  # Add the node to the executor
    thread = threading.Thread(
        target=executor.spin
    )  # Start the executor in a separate thread
    thread.start()
    return node, executor, thread  # Return the node, executor, and thread


def menu():
    print("Select mode:")
    print("1: Draw bounding boxes without screenshot.")
    print("2: Draw bounding boxes with screenshot.")
    print("3: 5 fps screenshot.")
    print("4: segmentation.")
    print("5: Custom detection. (For final project)")
    print("Press Ctrl+C to exit.")

    user_input = input("Enter your choice (1-5): ")
    return user_input


def main():
    """
    Main function to initialize the node and run the bounding box visualizer.
    """
    load_params = LoadParams("yolo_pkg")
    ros_communicator, executor, ros_thread = _init_ros_node()
    image_processor = ImageProcessor(ros_communicator, load_params)
    yolo_boundingbox = YoloBoundingBox(image_processor, load_params)
    yolo_depth_extractor = YoloDepthExtractor(
        yolo_boundingbox, image_processor, ros_communicator
    )
    boundingbox_visualizer = BoundingBoxVisualizer(
        image_processor, yolo_boundingbox, ros_communicator
    )
    camera_geometry = CameraGeometry(yolo_depth_extractor)

    user_input = menu()

    try:
        while True:
            if user_input == "1":
                offsets_3d = camera_geometry.calculate_offset_from_crosshair_2d()
                boundingbox_visualizer.draw_bounding_boxes(
                    draw_crosshair=True,
                    screenshot=False,
                    segmentation_status=True,
                    bounding_status=True,
                    offsets_3d_json=offsets_3d,
                )

                offset_msg = String()
                offset_msg.data = offsets_3d
                ros_communicator.publish_data("object_offset", offset_msg)

            elif user_input == "2":
                boundingbox_visualizer.draw_bounding_boxes(
                    draw_crosshair=True,
                    screenshot=True,
                    segmentation_status=False,
                    bounding_status=True,
                )
            elif user_input == "3":
                boundingbox_visualizer.draw_bounding_boxes(
                    draw_crosshair=True,
                    screenshot=False,
                    segmentation_status=False,
                    bounding_status=True,
                )

                # store 5fps unity camera picture
                boundingbox_visualizer.save_fps_screenshot()

            elif user_input == "4":
                boundingbox_visualizer.draw_bounding_boxes(
                    draw_crosshair=True,
                    screenshot=False,
                    segmentation_status=True,
                    bounding_status=False,
                )

            # =======================================
            # custom detection mode for final project
            # =======================================
            elif user_input == "5":
                # list: 每個物體的 {'label': str, 'offset_flu': np.ndarray([x, y, z])} 列表。
                # 但是是 JSON 格式的字串
                offsets_3d_json = camera_geometry.calculate_offset_from_crosshair_2d()
                offsets_3d = json.loads(offsets_3d_json)
                
                # np.ndarray([x_f, y_f, z_f]):x_f = +Z_cam （前方）,y_f = -X_cam （左方）,z_f = -Y_cam （上方）
                found = 1 if yolo_boundingbox.get_tags_and_boxes() else 0
                distance = offsets_3d[0]['offset_flu'][0] if offsets_3d else 0.0
                delta_x = offsets_3d[0]['offset_flu'][1] if offsets_3d else 0.0

                # dict: 包含 'center' (x,y) 座標, 'depth', 'depth_values' (list of depth values at x_num_splits points).
                depth_camera_center_value = yolo_depth_extractor.get_depth_camera_center_value()
                if depth_camera_center_value is not None:
                    camera_multi_depth = depth_camera_center_value["depth_values"]
                else:
                    camera_multi_depth = [-1.0 for _ in range(yolo_depth_extractor.x_num_splits)]
                boundingbox_visualizer.draw_bounding_boxes(
                    draw_crosshair=True,
                    screenshot=False,
                    segmentation_status=False,
                    bounding_status=True,
                    offsets_3d_json=offsets_3d_json,
                )
                offset_msg = String()
                offset_msg.data = offsets_3d_json
                ros_communicator.publish_data("object_offset", offset_msg)
                
                yolo_target_info_msg = Float32MultiArray()
                yolo_target_info_msg.data = [float(found), float(distance), float(delta_x)]
                ros_communicator.publish_data("yolo_target_info", yolo_target_info_msg)

                camera_x_multi_depth_values_msg = Float32MultiArray()
                camera_x_multi_depth_values_msg.data = camera_multi_depth
                ros_communicator.publish_data("camera_x_multi_depth_values", camera_x_multi_depth_values_msg)
            else:
                print("Invalid input.")

            # Example action for yolo_depth_extractor (can be removed if not needed)
            depth_data = yolo_depth_extractor.get_yolo_object_depth()
            print(f"Object Depth: {depth_data}")

    except KeyboardInterrupt:
        print("Shutting down gracefully...")
    finally:
        # Shut down the executor and ROS
        executor.shutdown()
        rclpy.shutdown()
        ros_thread.join()


if __name__ == "__main__":
    main()
