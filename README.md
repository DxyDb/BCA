# Block Circulant Adapter for Large Language Models

The official implementation of "Block Circulant Adapter for Large Language Models" (BCA).

## Introduction
Fine-tuning large language models (LLMs) is exceptionally difficult due to their huge model size and the high storage and computation costs involved. 

In this work, we propose a **block circulant matrix-based fine-tuning method (BCA)** with a stable training heuristic. By leveraging the elegant properties of circulant matrices and one-dimensional Fourier transforms, our method significantly reduces both storage and computational overhead. Experimental results demonstrate that our approach maintains close or even better task performance while using **14× fewer parameters than VeRA, 16× fewer than LoRA, and 32× fewer FLOPs than FourierFT**, presenting a highly promising direction for frequency-domain LLM fine-tuning.

## Citing our work

If you find this work helpful for your research, please consider citing our paper:

```bibtex
@inproceedings{ding2025block,
  title={Block circulant adapter for large language models},
  author={Ding, Xinyu and Wang, Meiqi} and Liao, Siyu and Wang, Zhongfeng},
  booktitle={Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence},
  pages={5030--5038},
  year={2025}
}
