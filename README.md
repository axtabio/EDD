# Expandable and Differentiable Dual Memories with Orthogonal Regularization (EDD)

[![AAAI 2026](https://img.shields.io/badge/AAAI-2026-blue.svg)](https://arxiv.org/abs/2511.09871)
[![arXiv](https://img.shields.io/badge/arXiv-2511.09871-b31b1b.svg)](https://arxiv.org/abs/2511.09871)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Official PyTorch Implementation** for the paper:  
> **"Expandable and Differentiable Dual Memories with Orthogonal Regularization for Exemplar-free Continual Learning"** > Accepted to **AAAI 2026**.

[**Hyung-Jun Moon**](https://github.com/axtabio) and [**Sung-Bae Cho**](http://sclab.yonsei.ac.kr/)  
Yonsei University, Seoul, Korea

---

## 📢 News
* **[2025.12]** 🎉 This paper has been accepted to **AAAI 2026**!
* **[2025.11]** The preprint is available on [arXiv](https://arxiv.org/abs/2511.09871).
* **[2025.11]** Code release.

---

## 📖 Abstract

<p align="center">
  <img src="figure.pdf" width="800" title="EDD Architecture">
</p>

Continual learning (CL) aims to learn a sequence of tasks while maintaining performance on previous tasks, but catastrophic forgetting remains a fundamental challenge. To address this, we propose a novel **Expandable and Differentiable Dual Memory (EDD)** method. 

**Key Features:**
- **Dual Memory Architecture**: Composed of a *Shared Memory* ($M^s$) for common features and a *Task-specific Memory* ($M^t$) for discriminative characteristics.
- **Fully Differentiable**: Enables end-to-end learning of latent representations without external buffers.
- **Orthogonal Regularization**: Enforces geometric separation between preserved and newly learned memory components to prevent interference.
- **Performance**: Outperforms 14 state-of-the-art methods on CIFAR-10, CIFAR-100, and Tiny-ImageNet benchmarks.

---

## 🛠️ Installation

This code is built on top of the [Mammoth](https://github.com/aimagelab/mammoth) framework (A PyTorch Framework for Benchmarking Continual Learning).

### Prerequisites
* Python 3.8+
* PyTorch
* Mammoth dependencies

### Setup
```bash
# Clone this repository
git clone [https://github.com/axtabio/EDD.git](https://github.com/axtabio/EDD.git)
cd EDD

# Install dependencies (Example)
pip install -r requirements.txt
