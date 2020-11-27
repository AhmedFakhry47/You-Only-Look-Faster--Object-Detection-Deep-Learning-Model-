# You-Only-Look-Faster--Object-Detection-Deep-Learning-Model-
Object Detection Model- Graduation Project Thesis

![Model Output](https://github.com/AhmedFakhry47/You-Only-Look-Faster--Object-Detection-Deep-Learning-Model-/blob/master/Model.gif)


## Motivation:
Modern self-driving cars require expensive processing units to run their algorithms. These pieces of hardware are expensive, which raises the production cost of these cars and in turn, puts another obstacle to them becoming more mainstream. Our contribution is to focus on optimizing one essential algorithm for self-driving cars which is the Object Detection Algorithm. We aim to achieve satisfactory accuracy (which we will measure by mAP) and low enough latency so that it can run on mid-end to low-end GPUs. Our original plan was to further test our model on low-end processing units but the delays and hinders due to the Covid-19 pandemic prevented us from seeing this through. 

## Design:
Our model combines two groundbreaking papers in the field of computer vision, namely YOLOv2 and MobileNet. Combining the powerful detection capabilities of the YOLOv2 algorithm with the low latency and high accuracy of the MobileNet classifier as a feature extractor would achieve our goals. Continuing on YOLO’s naming scheme (You Only Look Once), we name our model YOLF (You Only Look Faster). 

## Design Criteria:
When it came to designing our model, we set our priorities to optimize for speed first and accuracy second. Meaning that for our purposes, we didn’t need to maximize the model accuracy as long as it surpassed a certain baseline value. We didn’t need the best accuracy, only one that was good enough. But when it came for speed, we were limited by the Nvidia Tegra X1 chip of the Jetson Nano, so optimizing for speed and latency had priority over accuracy.

![](https://github.com/AhmedFakhry47/You-Only-Look-Faster--Object-Detection-Deep-Learning-Model-/blob/master/DesignCriteria.png)

We utilized the powerful design of YOLOv2 and the factorizational power of depthwise separable convolution. We use MobileNet as a feature extractor with αset to 0.25 or 0.5 to benefit from the speed-up inferred from Table (the previous table). This gave us more flexibility in designing the classification and localization portion of the model.
For the classification and localization portion of the model (called model head from here on), we experimented with different settings. We wanted a powerful design, that can be factorized with depthwise separable convolution. We redesigned the model head from YOLOv2 to incorporate depthwise separable convolution. We also experimented with using α> 1 to increase the model head’s power. 

