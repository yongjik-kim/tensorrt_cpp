# Simple C++ TensorRT inference library and example (WIP)

Testing and demonstrating functionalities of tensorRT framework in a C++ application. In particular, I tackle how NMS plugin can be used.

The goal of this repo is to look into how one can implement deep learning based detection inference workloads (For example, those using YOLO) The main library is largely based on [cyrusbehr's repo](https://github.com/cyrusbehr/tensorrt-cpp-api), but I removed building related functions. Use trtexec to build the engine.

This repo aims to provide three things:

1. Lay out how to prepare a onnx based detection model, add NMS TensorRT plugin `add_nms_to_graph.py` (You should edit this file to fit your model)
2. Provide a C++ TensorRT execution library `trt_infer_engine.h, trt_infer_engine.cpp`
3. Demonstrate how this library can be used in real scenario (WIP)

## Applying NMS(Non Maximum Suppresion) in TensorRT

In `add_nms_to_graph.py`, onnxsim is used to simplify the graph and apply some basic operation fusion(conv+BN -> conv, trim out constants). Although running onnxsim doesn't actually improve the end-result trt engine, it makes visualization way simpler so I consider it as best practice.

Then, [EfficientNMS](https://github.com/NVIDIA/TensorRT/tree/release/8.6/plugin/efficientNMSPlugin) plugin is attached to appropriate output nodes. TensorRT plugins such as [batchedNMS](https://github.com/NVIDIA/TensorRT/tree/release/8.6/plugin/batchedNMSPlugin) and [efficientNMS](https://github.com/NVIDIA/TensorRT/tree/release/8.6/plugin/efficientNMSPlugin) enable NMS inside TensorRT model. Unfortunately, other popular NMS methods such as [matrix NMS](https://arxiv.org/abs/2003.10152) and [soft NMS](https://openaccess.thecvf.com/content_ICCV_2017/papers/Bodla_Soft-NMS_--_Improving_ICCV_2017_paper.pdf) do not have plugins yet, as far as I know. As softNMS is widely used in the industry, one can try implementing softNMS plugin.

Although devs claim that EfficientNMS plugin will be deprecated and users should be using INMSLayer instead,[(source)](https://github.com/NVIDIA/TensorRT/issues/3000) it's way simpler to just use EfficientNMS plugin. So I use it here.

## Generating engine using trtexec

An example of this would look something like this:
```sh
trtexec --onnx=ppyoloe_plus_crn_l_80e_coco_w_trt_nms.onnx --saveEngine=ppyoloe_plus_crn_l_80e_coco_w_trt_nms.trt --fp16 --infStreams=1 --memPoolSize=workspace:2048 --iterations=100
```

You are encouraged to try multiple values of infStreams or memPoolSize, to see which maximizes the inference speed in your system.