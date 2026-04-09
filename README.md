# MoE-RelationNet++
## Introduction 
MoE-RelationNet++ is a relation-enhanced framework for object detection that leverages a **Mixture-of-Experts (MoE)** architecture to model heterogeneous feature relationships between keypoints. Traditional relation-based detection methods often rely on fixed keypoint selection and shared feature transformations, which struggle to handle diverse spatial and semantic patterns in complex scenes.

To address this limitation, MoE-RelationNet++ introduces adaptive keypoint selection and conditional feature transformation. The framework consists of three key components:

* **Lightweight Key Selector** – adaptively selects informative keypoints from dense feature maps.
* **MoE Enhancement Module** – dynamically routes keypoints to specialized experts for conditional feature transformation.
* **Energy Verification Mechanism** – evaluates the reliability of enhanced features and filters noisy responses.

The proposed module can be seamlessly integrated into common object detection frameworks such as RetinaNet, FCOS, ATSS, and Faster R-CNN. Experiments on MS COCO and VisDrone-DET2019 demonstrate consistent improvements in detection performance.

## How to use it
- ### Install it
```shell
cd ${your_code_dir}
git clone https://github.com/hani-yu/MoE-RelationNet-.git
```
where `your_code_dir` is your code path. For more information, you may refer to [getting started](https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md)

- ### For testing
```shell
bash tools/dist_test.sh ${selected_config} 8
```
where `selected_config` is one of provided script under the `configs` folder.
- ### For training
```shell
bash tools/dist_train.sh ${selected_config} 8
```
where `selected_config` is one of provided script under the `configs` folder.
