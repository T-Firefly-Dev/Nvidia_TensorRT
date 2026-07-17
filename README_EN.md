# NVIDIA AIBOX + TensorRT: A High-Performance Solution for Real-Time Semantic Segmentation

---

## NVIDIA Series AIBOX

The AIBOX-OrinNano and AIBOX-OrinNX are both equipped with original NVIDIA Jetson Orin core modules. They come standard with industrial-grade all-metal casings and aluminum alloy structures for heat dissipation. The top cover features a strip grille design on the side for efficient cooling, ensuring computational performance and stability even under high-temperature operating conditions, meeting various industrial application requirements.

| Category | AIBOX-OrinNX | AIBOX-OrinNano |
| ---- | ---- | ---- |
| Module | Jetson Orin NX 16GB | Jetson Orin Nano 8GB |
| AI Performance | 100 TOPS | 40 TOPS |
| GPU | 1024-core NVIDIA Ampere architecture GPU with 32 Tensor Cores | 1024-core NVIDIA Ampere architecture GPU with 32 Tensor Cores |
| CPU | 8-core 64-bit Arm Cortex-A78 CPU<br>2MB L2 Cache + 4MB L3 Cache | 6-core 64-bit Arm Cortex-A78 CPU<br>1.5MB L2 Cache + 4MB L3 Cache |
| DDR | 16GB 128-bit LPDDR5, 102.4GB/s Bandwidth | 8GB 128-bit LPDDR5, 68GB/s Bandwidth |
| HDMI | 4K @ 60Hz | 4K @ 30Hz |


## NVIDIA TensorRT

The NVIDIA series AIBOX supports the deep learning framework TensorRT. TensorRT is an API ecosystem for high-performance deep learning inference, including inference runtime and model optimization, providing low latency and high throughput for production applications.

The TensorRT ecosystem includes TensorRT, TensorRT-LLM, TensorRT Model Optimizer, and TensorRT Cloud.

**Advantages of NVIDIA TensorRT**

1.  Inference speed increased by up to 36x
2.  Optimized inference performance
3.  Accelerated various workloads
4.  Deploy, run, and scale using Triton

## Semantic Segmentation

Semantic segmentation is based on image recognition, but classification is performed at the pixel level rather than on the entire image. This is achieved by convolving a pre-trained image recognition backbone network, converting the model into a Fully Convolutional Network (FCN) capable of performing per-pixel annotation. Semantic segmentation is particularly useful for environmental perception, as it allows for dense per-pixel classification of many different potential objects (including foreground and background) in each scene.

![Semantic Segmentation Concept](./res/2.webp)

### SegNet Model

The novelty of SegNet lies in the way the decoder upsamples its lower-resolution input feature maps. Specifically, the decoder uses the pooling indices computed in the corresponding encoder's max-pooling steps to perform non-linear upsampling. The upsampled feature maps are sparse, so trainable convolution kernels are subsequently used to perform convolution operations to generate dense feature maps. SegNet's architecture is compared with the widely adopted FCN and the well-known DeepLab-LargeFOV and DeconvNet architectures. The comparison results reveal the trade-off between memory and accuracy involved in achieving good segmentation performance.

![SegNet Architecture](./res/3.webp)

### Download Source Code

```bash
$ git clone --recursive --depth=1 https://github.com/T-Firefly-Dev/Nvidia_TensorRT
```

### Compile / Install

> Reference: [Building the Project from Source](https://github.com/T-Firefly-Dev/Nvidia_TensorRT/blob/master/docs/building-repo-2.md)

### Run Example

```bash
$ ./segnet.py --network=fcn-resnet18-cityscapes city_0.jpg output_city_0.jpg
```

![Segmentation Result](./res/4.webp)
