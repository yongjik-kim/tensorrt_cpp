# Note: onnx_graphsurgeon library should be built manually, from TensorRT repo
# https://github.com/NVIDIA/TensorRT

import onnx_graphsurgeon as gs
import numpy as  np
import onnx
import os
from onnxsim import simplify

batch_size = 1
simp_model_path = "ppyoloe_plus_crn_l_80e_coco_wo_nms_simplified.onnx"

if os.path.exists(simp_model_path):
    # Load the model
    onnx_model = onnx.load("ppyoloe_plus_crn_l_80e_coco_wo_nms.onnx")

    # Explicitly set the input batch size as constant (usually, 1)
    for input_tensor in onnx_model.graph.input:
        dim1 = input_tensor.type.tensor_type.shape.dim[0]
        dim1.dim_value=batch_size
    onnx_model_simp, check = simplify(onnx_model)

    # Save the simplified model
    onnx.save(onnx_model_simp, simp_model_path)
else:
    onnx_model_simp = onnx.load(simp_model_path)

onnx_graph = gs.import_onnx(onnx_model_simp)

keep_top_k = 100
batch_size = 1

attributes= {
    "score_threshold": 0.25,
    "iou_threshold": 0.6,
    "max_output_boxes": keep_top_k,
    "background_class": -1,
    "score_activation": False,
    "class_agnostic": True,
    "box_coding": 1
}

# 1. Add transpose node

tensors = onnx_graph.tensors()

transpose_node = gs.Node(
    op="Transpose",
    inputs=[tensors["concat_14.tmp_0"]],
    outputs=[gs.Variable(name="concat_14.tmp_0_transposed", dtype=np.float32)],
    attrs={"perm": [0, 2, 1]}
)

onnx_graph.nodes.append(transpose_node)

# 2. 

tensors = onnx_graph.tensors()

boxes_input = tensors["tmp_128"]
scores_input = tensors["concat_14.tmp_0_transposed"]

num_detections = gs.Variable(name="num_detections").to_variable(dtype=np.int32, shape=[batch_size,1])
detection_boxes = gs.Variable(name="detection_boxes").to_variable(dtype=np.float32, shape=[batch_size,keep_top_k,1])
detection_scores = gs.Variable(name="detection_scores").to_variable(dtype=np.float32, shape=[batch_size,keep_top_k])
detection_classes = gs.Variable(name="detection_classes").to_variable(dtype=np.int32, shape=[batch_size,keep_top_k])

new_outputs = [num_detections, detection_boxes, detection_scores, detection_classes]

nms_node = gs.Node(
    name = "Efficient_NMS",
    op = "EfficientNMS_TRT",
    attrs = attributes,
    inputs = [boxes_input, scores_input],
    outputs = new_outputs
)

onnx_graph.nodes.append(nms_node)
onnx_graph.outputs = new_outputs

onnx_graph = onnx_graph.cleanup().toposort()

onnx.save(gs.export_onnx(onnx_graph), 'ppyoloe_plus_crn_l_80e_coco_w_trt_nms.onnx')