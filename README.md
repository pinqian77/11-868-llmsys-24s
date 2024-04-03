# 11-868-llmsys-24s

This repo contains source code and my solution to CMU 11-868: Large Language Model Systems (Spring 2024).

## Assignment 1: MiniTorch Module

Implemented a basic deep learning framework named miniTorch that performs operations on tensors with automatic differentiation, using Python, C++, and CUDA.
- **Automatic Differentiation**: Implement `topological_sort` and `backpropagate` functions for automatic differentiation.
- **CUDA Backend Operators**: Implement CUDA kernels for matrix multiplication, and map, zip, and reduce functions.
- **Neural Network Architecture and Training Procedure**: Implement a simple feedforward neural network for sentiment classification, including linear layers and training loop.

## Assignment 2: Enhancing MiniTorch for GPT-2

Extended miniTorch to implement a decoder-only transformer architecture (GPT-2) for a machine translation task.
- **Scalar Power and Tanh**: Add missing operations to miniTorch.
- **Tensor Functions**: Implement GELU, logsumexp, one_hot, and softmax_loss functions.
- **Basic Modules**: Implement Linear, Dropout, LayerNorm1d, and Embedding modules.
- **Decoder-only Transformer Model**: Implement the GPT-2 architecture, including MultiHeadAttention, FeedForward, TransformerLayer, and DecoderLM.
- **Machine Translation Pipeline**: Implement training pipeline for machine translation on the IWSLT (De-En) dataset.

## Assignment 3: CUDA Optimizations for Transformers

Enhanced the efficiency of the transformer model, specifically targeting the Softmax and LayerNorm operations, by employing custom CUDA code optimizations derived from the lightseq2 approach.
- **Softmax Optimization**: Develop a fused CUDA kernel for the softmax operation, including handling of attention masks, which is crucial for the attention mechanism's performance. This involves implementing ker_attn_softmax and ker_attn_softmax_lt32 for different sequence lengths.
- **LayerNorm Optimization**: Implemented an fused CUDA kernel for the LayerNorm operation, focusing on improving batch reduction operations to enhance the efficiency of layer normalization within the transformer model.

## Assignment 4: Distributed and Pipeline Parallel Training

Implemented distributed training methods, including data parallelism and pipeline parallelism for GPT-2.
- **Data Parallel Training**: Implement partitioning datasets for training across GPUs, setup process groups, and aggregate gradients.
- **Pipeline Parallel Training**: Implement helper functions for layer-wise splitting and scheduling, understand worker code, and prepare GPT-2 for pipeline parallelism.
- **Performance Evaluation**: Compare performance metrics between single-device and multi-device training, including training time and tokens per second.