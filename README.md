# You-Only-Look-Faster--Object-Detection-Deep-Learning-Model-
Object Detection Model- Graduation Project Thesis

![Model Output](https://github.com/AhmedFakhry47/You-Only-Look-Faster--Object-Detection-Deep-Learning-Model-/blob/master/Model.gif)

Motivation:
Modern self-driving cars require expensive processing units to run their algorithms. These pieces of hardware are expensive, which raises the production cost of these cars and in turn, puts another obstacle to them becoming more mainstream. Our contribution is to focus on optimizing one essential algorithm for self-driving cars which is the Object Detection Algorithm. We aim to achieve satisfactory accuracy (which we will measure by mAP) and low enough latency so that it can run on mid-end to low-end GPUs. Our original plan was to further test our model on low-end processing units but the delays and hinders due to the Covid-19 pandemic prevented us from seeing this through. 

Design:
Our model combines two groundbreaking papers in the field of computer vision, namely YOLOv2 and MobileNet. Combining the powerful detection capabilities of the YOLOv2 algorithm with the low latency and high accuracy of the MobileNet classifier as a feature extractor would achieve our goals. Continuing on YOLO’s naming scheme (You Only Look Once), we name our model YOLF (You Only Look Faster). 

MobileNets:
 MobileNets is the name of a series of papers published by Google’s Deep Learning team. They were aiming for the MobileNets family of classifiers to be the architecture of choice for embedded applications as is evident from the name. They aim to achieve the balance between satisfactory classification accuracy and low latency. 
As deep learning classifiers are usually used as decoders as feature extractors in Object Detection algorithms, the authors argued that their model would be useful in building object detection algorithms for mobile application. Below are tables from the MobileNetV1 paper. 
 Table ( ):MobileNetV1 classifier in comparison to popular models on ImageNet dataset..4

 Table ( ):COCO object detection results comparison using different frameworks and network architectures.3
As evident from the aforementioned tables, MobileNetV1 performs almost as well as other -then- state of the art models while being much smaller in size, and thus less complex. In table () it is shown that for SSD, MobileNet has sacrifices 1.8% mAP in comparison to VGG while being 35 times less complex (fewer Mult-Adds operations) and 5 times smaller in size (fewer parameters). Similar results are seen in two-step detectors like Faster-RCNN. 
The basis of MobileNetV1 is replacing traditional convolutional layers with Depthwise Separable convolutions, which will be explained in section 2.2.1.2.2. 
2.2.1.2.2  Convolutional Neural Networks and Depthwise Convolution:
The main layer used in computer vision applications is the convolutional layer. The convolutional layer allows for feature sharing among different areas of the image, meaning that the same set of weights is applied to different parts of the image. 
Consider an input of shape DF x DF xM where DF   is the height or width of the input and cis the number of channels in said input, and N convolutional kernels of shape DK x DK xM with DK<DF   . The convolutional kernels will scan the image, overlapping with different parts of the image. The weights of the kernel will be multiplied by the pixel values of the input and then the resulting matrix will be added. This allows larger parts of the input to be represented by a single image, effectively compressing the input. This method is much faster and effective than vanilla fully connected layers because a smaller set of weights is used. Also matrix operations are vectorizable and concurrent which further increases the speed of the operation.    Figure (): A typical classification NN. A series of convolutional layers followed by fully connected layers and a softmax nonlinearity.
To further clarify the power of convolutional neural networks we present a primitive example. Consider the following image which shows a sharp transition between black and white. With only a few parameters, we can detect exactly where the edge occurs. 


Consider the matrix representation of the image, where 1 represents a black pixel and 0 represents a white pixel. Using the kernel shown, with only 9 weights we can detect the edge in the output. 

In the output matrix, the presence of the non-zero values signifies the presence of an edge, and the sign of these values represents the type of the transition. If the values are positive, the input had a positive edge (a transition from black to white) and were they negative, the input would have had a negative edge (a transition from black to white) .
For more complex kernels, each kernel recognizes a certain feature. For early layers, the features are general and straight forward like vertical, horizontal and diagonal edges, and deeper layers are more specialized to the training set, like facial features. 

However, convolutional layers are not the best we can do. For a kernel of size DK x DK  x M x N, we have a total of  DK 2* M * N  parameters. This kernel is across all M input channels and the convolution is done in one step. However, the authors of this paper relied on Depthwise Separable convolution. In this layer, standard convolution is split into two steps, M separate DK  x DK  depthwise convolution, where we have M output feature maps that are yet to be combined together, followed by N 1 x 1 x M separable convolutions to combine these M separate feature maps into one feature map N time. Figure () is from MobileNet paper and explains the difference between standard convolution and depthwise separable convolution. For depthwise separable convolution, we will have DK 2 * M +M*N parameters. Thus we’d have a saving factor of 1N+1 DK2 which is a substantial saving factor for large networks and it accounts for the massive decrease in computational overhead in MobileNets.

Figure (). The standard convolutional kernels in (a) are replaced by two kernels: depthwise convolution in (b) and pointwise convolution in (c) to build a depthwise separable kernel.
Guided by this new powerful layer, the authors designed the MobileNet classifier. The architecture of MobileNet is shown in Figure()


Figure(): MobileNet Architecture.
2.2.1.2.3 Width Multiplier:
Although MobileNet was already a very powerful model, the authors aimed to further increase the model’s speed by introducing the width multiplier. Basically, for a given layer and width multiplier α, the number of input channels M becomes αM and the number of output channels N becomes αN. The number of parameters then becomes DK 2 * α*M +α2*M*N and the saving factor becomesαN+α2 DK2. Where α ∈ (0, 1] with typical settings of 1, 0.75, 0.5 and 0.25. α = 1 is the baseline MobileNet and α < 1 are reduced MobileNets. Width multiplier has the effect of reducing computational cost and the number of parameters quadratically by roughlyα . Width multiplier can be applied to any model structure to define a new smaller model with a reasonable accuracy, speed and size trade off. Figure () shows the trade-off between speed and size and accuracy as αdecreases. 
Figure(): The effect of αon classification accuracy
As evident from the table αhas a clear effect on the number of parameters of the classifier. The accuracy drops smoothly with the decrease of α. However for object detection with MobileNet as the feature extractor, the effect on accuracy is less evident. Figure () shows that effect. 
Figure(): COCO object detection results comparison using different frameworks and network architectures.
By decreasing αto 0.5, one 1% of accuracy was lost but the model shrank 25%. Setting αsaves even more parameters. 
An additional benefit to the width multiplier is the decrease in number of parallel operations, which can decrease the bottleneck of speed in CPU computation. This feature made our design suitable for CPU computational with sufficient speed.
2.2.1.2.4 Design Criteria:
When it came to designing our model, we set our priorities to optimize for speed first and accuracy second. Meaning that for our purposes, we didn’t need to maximize the model accuracy as long as it surpassed a certain baseline value. We didn’t need the best accuracy, only one that was good enough. But when it came for speed, we were limited by the Nvidia Tegra X1 chip of the Jetson Nano, so optimizing for speed and latency had priority over accuracy.

Model Design Map
We utilized the powerful design of YOLOv2 and the factorizational power of depthwise separable convolution. We use MobileNet as a feature extractor with αset to 0.25 or 0.5 to benefit from the speed-up inferred from Table (the previous table). This gave us more flexibility in designing the classification and localization portion of the model.
For the classification and localization portion of the model (called model head from here on), we experimented with different settings. We wanted a powerful design, that can be factorized with depthwise separable convolution. We redesigned the model head from YOLOv2 to incorporate depthwise separable convolution. We also experimented with using α> 1 to increase the model head’s power. 
In the following sections, we will talk about our choices for hyperparameters, training process, and finally, our final model design.
