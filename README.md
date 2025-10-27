<div align="center">
  <img src="assets/merge.jpeg" alt="WorldGrow merge" height="150">
  <h1>WorldGrow: Generating Infinite 3D World</h1>

  <p>
    <a href="https://world-grow.github.io/">
      <img src="https://img.shields.io/badge/Project%20Page-online-brightgreen.svg" alt="Project Page">
    </a>
    <a href="https://arxiv.org/abs/2510.21682">
      <img src="https://img.shields.io/badge/arXiv-2510.21682-b31b1b.svg" alt="arXiv">
    </a>
    <a href="https://huggingface.co/datasets/your-org/worldgrow">
      <img src="https://img.shields.io/badge/Data-HuggingFace-yellow.svg" alt="Dataset">
    </a>
    <a href="https://world-grow.github.io/play">
      <img src="https://img.shields.io/badge/Game-Play%20Now-blueviolet.svg" alt="Play the game">
    </a>
  </p>
</div>


<p align="center">
  <a href="https://scholar.google.com/citations?user=2dCJlg4AAAAJ">Sikuang Li</a><sup>1*</sup>,
  <a href="https://chensjtu.github.io/">Chen Yang</a><sup>2*</sup>,
  <a href="https://jaminfong.cn/">Jiemin Fang</a><sup>2✉</sup>,
  <a href="https://taoranyi.com/">Taoran Yi</a><sup>3</sup>,
  <a href="https://hustvl.github.io/Snap-Snap/">Jia Lu</a><sup>3</sup>,<br/>
  <a href="https://scholar.google.com/citations?user=JZIKCF0AAAAJ&hl=en">Jiazhong Cen</a><sup>1</sup>,
  <a href="http://lingxixie.com/Home.html">Lingxi Xie</a><sup>2</sup>,
  <a href="https://shenwei1231.github.io/">Wei Shen</a><sup>1</sup>,
  <a href="https://www.qitian1987.com/">Qi Tian</a><sup>2✉</sup>
</p>

<p align="center">
  <sup>1</sup>Shanghai Jiao Tong University 
  <sup>2</sup>Huawei 
  <sup>3</sup>Huazhong University of Science and Technology<br/>
  <sup>*</sup>Equal contribution   <sup>✉</sup>Corresponding author
</p>

---

We propose **WorldGrow** — a generative method which creates **infinite EXPLICIT 3D** worlds, an alternative to the extensible, realistic, interactive world simulator.

[![Watch the video](https://img.youtube.com/vi/blVXMwGHQO4/0.jpg)](https://www.youtube.com/watch?v=blVXMwGHQO4)

## Overview

**WorldGrow** is a hierarchical framework for **infinite (open-ended) 3D world generation**. Starting from a single seed block, the system **grows** large environments via **block-wise synthesis** and **coarse-to-fine refinement**, producing coherent global layouts and detailed local geometry/appearance. The generated scenes are **walkable** and suitable for **navigation/planning** evaluation.

> If you use any part of this repository, please consider starring ⭐ the project and citing our paper.

## News

- **2025-10-27** — 🚧 Paper released and repository initialized. The code is being prepared for public release; pretrained weights and full training/inference pipelines are planned.



## Getting Started

> ⚠️ The repo is under active development. Interfaces may change. Placeholders below will be updated as components are released.

### Requirements (placeholder)


### Installation (placeholder)


### Visualization (placeholder)


## Results

* **Gallery**: diverse generated scenes at multiple scales.
* **Large-scale example**: a 19x39 indoor world (~1,800 m²) with reconstructed mesh and textured rendering.

Please visit the **[project page](https://world-grow.github.io/)** for more figures, videos, and metrics.


## License

TBD (to be finalized before full code release).

## Citation

```bibtex
@article{worldgrow2025,
  title   = {WorldGrow: Generating Infinite 3D World},
  author  = {Li, Sikuang and Yang, Chen and Fang, Jiemin and Yi, Taoran and Lu, Jia and Cen, Jiazhong and Xie, Lingxi and Shen, Wei and Tian, Qi},
  journal = {arXiv preprint arXiv:2510.21682},
  year    = {2025}
}
```
