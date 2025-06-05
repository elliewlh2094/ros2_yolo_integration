# ros2_yolo_integration
## yolo_pkg
### Usage
1. 執行提供的 shell script 來啟動容器並準備環境
```
./yolo_activate.sh
```
2. 執行 `colcon buiild` 和 `source ./install/setup.bash`
```
r
```
3. 執行 yolo 節點
```
ros2 run yolo_pkg yolo_detection_node
```
### Mode
YOLO 節點可以以不同的模式運作：
- Mode 1: Draw bounding boxes without screenshot
    - 執行 Detection 畫出 bounding boxes，不截圖。在輸出 topic `/yolo/detection/compressed` 上顯示 YOLO 偵測到的 bounding boxes。
- Mode 2: Draw bounding boxes with screenshot
    - 執行 Detection 畫出 bounding boxes。對每一個偵測到的物件截圖儲存，儲存位置在 `/fps_screenshots`
- Mode 3: 5 fps screenshot
    - 以 5 fps 頻率從截取螢幕截圖
- Mode 4: segmentation
    - 啟用 segmentation 模式，在輸出 topic `/yolo/detection/compressed` 顯示分割 mask。
### Function
- draw_bounding_boxes
    - `draw_crosshair`(bool) :
        - 如果設定為 `True`，會在影像的中心繪製一個十字線。
    - `screenshot`(bool) :
        - 如果為 `True`，將在畫出 bounding boxes 後擷取影像的螢幕截圖。
    - `segmentation_status`(bool) :
        - 控制分割 mask 是否顯示在輸出影像上。
    - `bounding_status`(bool) :
        - 控制是否在輸出影像上顯示 bounding boxes。
- save_fps_screenshot
    - 以每秒 5 幀的固定速率擷取螢幕截圖。
### class diagram
![Logo](https://github.com/alianlbj23/ros2_yolo_integration/blob/main/img/image_deal.jpeg?raw=true)
## yolo_example_pkg
This is a ROS 2 project for integrating YOLO with ROS 2, providing functionality for real-time object detection and bounding box visualization.
### Features
- Receive compressed images from the /yolo/detection/compressed topic.
- Process images to convert them into OpenCV format.
- Draw bounding boxes around detected objects on the images.
- Publish processed images back to the /yolo/detection/compressed topic, which includes the drawn bounding boxes.
### Usage
1. Run the provided activation script to start the container and prepare the environment
```
./yolo_activate.sh
```
2. Do colcon build and source ./install/setup.bash
```
r
```
3. Run yolo node
```
ros2 run yolo_example_pkg yolo_node
```

## How to put your yolo model
To use the YOLO model in this package, follow these steps:
1. Place the Model File

    Download or obtain the YOLO .pt file (e.g., yolov8n.pt) and place it inside the models directory within the pkg folder:

    ```
    <your_ros2_workspace>/
    ├── src/
    │   ├── yolo_example_pkg/
    │   │   ├── models/
    │   │   │   ├── yolov8n.pt
                ├── yolov8n-seg.pt
    ```
    You can update the model used in the following files inside `yolo_pkg`:
    - `yolo_segmentation_model.py`
    - `yolo_detect_model.py`

2. Model Path Configuration

    The script dynamically loads the model from the package's shared directory using the following code:

    ```
    import os
    from ament_index_python.packages import get_package_share_directory

    model_path = os.path.join(
        get_package_share_directory("yolo_example_pkg"), "models", "yolov8n.pt"
    )
    ```
