# DexGraspNet-X2 Collection

A dexterous grasp synthesis and validation pipeline for our custom-designed X2 hand.

This project extends DexGraspNet with X2-specific grasp generation, including front/back palm grasping strategies and PhysX-based validation. It covers the full loop from grasp strategy design and optimization to simulation validation and dataset curation.

## Highlights

- Designed X2-specific dexterous grasp strategies
- Supported front-palm and back-palm grasping
- Supported 1–5 finger grasp configurations
- Adapted grasp optimization to X2 kinematics
- Generated and validated 5,000 physically feasible grasps
- Built Isaac Sim / PhysX validation pipeline

## Pipeline

- **Grasp strategy design.** Front-palm and back-palm grasping modes with 1–5 finger configurations, including non-overlapping finger masks.
- **Grasp optimization.** Side-conditioned mesh-grasp generation adapted to X2 kinematics, covering finger count and palm orientation.
- **Simulation validation.** Isaac Sim / PhysX v7 protocol that accepts a grasp only after it passes six-direction physical checks.
- **Dataset curation.** The formal collector targets exactly 5,000 audited valid records across 12 deterministic primitives and 30 selected general meshes, with balanced front/back and finger-count quotas.

## Demo

(Add grasping videos here)

## Documentation

- [X2 mesh grasp generator](docs/x2_mesh_grasp_generator.md)
- [PhysX validation protocol](docs/x2_physx_grasp_validation.md)
- [Dataset protocol](docs/x2_primitive_dataset.md)
- [Collection and recovery runbook](docs/x2_collection_runbook.md)
- [Experiment log](docs/x2_experiment_log.md)

## Repository contents

The public repository contains source code, tests, configuration, operational documentation, and sanitized sample visualizations. Raw/valid/failed records, simulator caches, checkpoints, the general-object meshes, and X2 USD/geometry assets are intentionally excluded. To run the pipeline, provide the locally licensed X2 asset at `x2_mujoco/x2_keypoints.usda` (including its payloads) and the general meshes under `data/meshdata/<object_id>/coacd/decomposed.obj`. Generated raw optimizer poses are not valid data until they pass the documented six-direction PhysX protocol.

## Attribution

DexGraspNet is the foundation of this project. If you use this work, please cite:

```bibtex
@article{wang2022dexgraspnet,
  title={DexGraspNet: A Large-Scale Robotic Dexterous Grasp Dataset for General Objects Based on Simulation},
  author={Wang, Ruicheng and Zhang, Jialiang and Chen, Jiayi and Xu, Yinzhen and Li, Puhao and Liu, Tengyu and Wang, He},
  journal={arXiv preprint arXiv:2210.02697},
  year={2022}
}
```

Paper: https://arxiv.org/abs/2210.02697 · Project page: https://pku-epic.github.io/DexGraspNet/

## License

This work and the dataset are licensed under [CC BY-NC 4.0][cc-by-nc]. This repository remains subject to the upstream DexGraspNet attribution and CC BY-NC 4.0 terms; no robot or object asset license is granted here.

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
