#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
結合 YOLO 偵測與深度資料，推算出 bounding box 對應目標的深度值
"""

import numpy as np


class YoloDepthExtractor:
    def __init__(self, yolo_boundingbox, image_processor, ros_communication):
        self.yolo_boundingbox = yolo_boundingbox
        self.ros_communicator = ros_communication
        self.image_processor = image_processor
        # 相機畫面中央高度上切成 n 個等距水平點。
        self.x_num_splits = 20

    def get_yolo_object_depth(self, radius_increment=2, max_iterations=10):
        """
        計算每個偵測到的物體的深度。如果中心像素深度無效 (<=0 or NaN)，則從中心向外搜尋，
        直到找到有效深度（視窗中有效像素的平均值）或視窗達到 bbox 限制或最大迭代次數

        Args:
            radius_increment (int): 每次迭代中搜尋半徑的增量 (e.g., 2 means radius grows 2, 4, 6...).
            max_iterations (int): 最大迭代步驟數，以防止無限迴圈。

        Returns:
            list: 一個 dictionary list，每個 dict 包含 'label', 'box', and 'depth'. Depth is in meters (float) 如果未找到有效深度，則傳回原本的 invalid value
        """
        depth_cv_image = self.image_processor.get_depth_cv_image()
        if depth_cv_image is None or not isinstance(depth_cv_image, np.ndarray):
            print("Depth image is invalid.")
            return []

        detected_objects = self.yolo_boundingbox.get_tags_and_boxes()
        if not detected_objects:
            return []

        objects_with_depth = []
        img_h, img_w = depth_cv_image.shape[:2]

        for obj in detected_objects:
            label = obj["label"]
            x1, y1, x2, y2 = map(int, obj["box"])  # Ensure box coords are int

            # Clamp box coordinates to image dimensions
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w - 1, x2), min(img_h - 1, y2)

            if x1 >= x2 or y1 >= y2:
                print(f"Invalid box dimensions after clamping for {label}. Skipping.")
                continue

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # --- Initial depth check at the center ---
            depth_value = depth_cv_image[center_y, center_x]
            is_invalid = depth_value <= 0 or np.isnan(depth_value)

            # --- Iterative Expansion Search (if center is invalid) ---
            if is_invalid:
                found_valid_depth = False
                current_radius = 0
                for iteration in range(max_iterations):
                    current_radius += radius_increment

                    # Define search window boundaries, clamped by bounding box
                    min_x = max(x1, center_x - current_radius)
                    max_x = min(x2, center_x + current_radius)
                    min_y = max(y1, center_y - current_radius)
                    max_y = min(y2, center_y + current_radius)

                    # Check if window covers the entire bounding box - stop if it does
                    if (
                        min_x == x1
                        and max_x == x2
                        and min_y == y1
                        and max_y == y2
                        and iteration > 0
                    ):
                        # print(f"Window reached bbox limits for {label} at radius {current_radius}. Stopping search.") # Optional debug
                        break  # No point expanding further

                    # Extract the window (ensure valid slice indices)
                    if (
                        min_y > max_y or min_x > max_x
                    ):  # Should not happen with clamping, but safety check
                        continue
                    depth_window = depth_cv_image[min_y : max_y + 1, min_x : max_x + 1]

                    # Find valid depths within the window
                    valid_depths = depth_window[
                        (depth_window > 0) & (~np.isnan(depth_window))
                    ]

                    if valid_depths.size > 0:
                        # Calculate the mean of valid depths
                        depth_value = np.mean(valid_depths)
                        found_valid_depth = True
                        # print(f"Found valid depth for {label} at radius {current_radius}: {depth_value:.3f}m") # Optional debug
                        break  # Exit loop once valid depth is found

                # if not found_valid_depth: # Optional debug
                # print(f"No valid depth found for {label} after {max_iterations} iterations.")

            # --- Append Result ---
            # depth_value will be the original center value (valid or invalid)
            # or the mean from the first successful window expansion.
            # If expansion failed, it remains the original invalid value.
            objects_with_depth.append(
                {
                    "label": label,
                    "box": (x1, y1, x2, y2),
                    "depth": (
                        float(depth_value) if not np.isnan(depth_value) else np.nan
                    ),  # Handle potential NaN persistence
                }
            )

        return objects_with_depth

    # ... rest of the class (get_depth_camera_center_value) ...
    # Note: get_depth_camera_center_value still uses a fixed window search
    def get_depth_camera_center_value(self):
        """
        傳回深度相機中心點的深度值及其中心座標。如果中心無效，則使用固定視窗搜尋。

        Returns:
            dict: 包含 'center' (x,y) 座標, 'depth', 'depth_values' (list of depth values at x_num_splits points).
            如果深度影像無效或在中心附近未找到有效深度，則傳回 None。
        """
        depth_cv_image = self.image_processor.get_depth_cv_image()
        is_invalid_depth_image = depth_cv_image is None or not isinstance(
            depth_cv_image, np.ndarray
        )
        if is_invalid_depth_image:
            print("Depth image is invalid.")
            return None

        height, width = depth_cv_image.shape[:2]
        center_x = width // 2
        center_y = height // 2
        lower_y = height // 2 - height // 4
        segment_length = width // self.x_num_splits

        # 取得中心點的深度值
        center_depth = depth_cv_image[center_y, center_x]

        # 如果中心點深度無效 (<=0 or NaN)，則使用固定視窗搜尋
        if center_depth <= 0 or np.isnan(center_depth):
            window_size = 5  # Fixed window size (e.g., 11x11) for camera center
            min_r, max_r = max(0, center_y - window_size), min(
                height, center_y + window_size + 1
            )
            min_c, max_c = max(0, center_x - window_size), min(
                width, center_x + window_size + 1
            )
            window = depth_cv_image[min_r:max_r, min_c:max_c]

            non_zero_values = window[(window > 0) & (~np.isnan(window))]
            if non_zero_values.size > 0:
                center_depth = np.mean(non_zero_values)
            else:
                print("No valid depth value found near the camera center point.")
                return None

        # 確保返回值為浮點數，處理潛在的 NaN
        final_depth = float(center_depth) if not np.isnan(center_depth) else np.nan
        if np.isnan(final_depth):
            print("Final center depth is NaN.")
            return None  # Return None if depth remains NaN

        # 取得 n 個等分點的深度值
        points = [(i * segment_length, lower_y) for i in range(self.x_num_splits)]
        depth_values_list = [depth_cv_image[lower_y, x] for x, _ in points]
        if not depth_values_list:
            print("No valid depth values found in the horizontal segments.")
            return None

        # 確保返回值為浮點數，過濾掉無效的深度值 (<=0 or NaN)
        depth_values_list = [
            float(depth) if depth > 0 and not np.isnan(depth) else -1.0
            for depth in depth_values_list
        ]

        return {"center": (center_x, center_y), 
                "depth": final_depth, 
                "depth_values": depth_values_list}
