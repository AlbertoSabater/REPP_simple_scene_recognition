# REPP - Simple scene recognition

Simple scene recognition based on the kind of objects detected for each frame. The present script loads object detections (COCO format), evaluates the presence of different objects in each frame and classifies the per-frame scene accordingly into 4 different categories (kitchen, dining room, bathroom, bedroom).

For the experiments, [Yolov3 + REPP](https://github.com/AlbertoSabater/Robust-and-efficient-post-processing-for-video-object-detection) can be used ([COCO weights](https://github.com/AlbertoSabater/Robust-and-efficient-post-processing-for-video-object-detection/blob/master/demos/YOLOv3/README.md#download-pretrained-models-and-convert-to-keras)) to perform the object detection following the [instructions for custom videos](https://github.com/AlbertoSabater/Robust-and-efficient-post-processing-for-video-object-detection#repp-applied-to-custom-videos). 

To perform the Simple scene recognition and render a final video with the detections and classifications results, execute:

`python REPP_scene_recognition.py  --preds_filename path_to_coco_predictions.json --min_score 0.1 --video_filename path_to_original_video.mp4`
