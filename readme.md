# PyTorch Transformer from Scratch

This repository contains an implementation of the Transformer model in PyTorch based on the seminal paper [Attention is All You Need](https://arxiv.org/abs/1706.03762). The code implements all the core components—from multi-head self-attention and positional encoding to encoder/decoder stacks and a final generator module.

## Overview

The Transformer architecture in this repository includes:

- **Multi-Head Attention:** Implements scaled dot-product attention with multiple heads.
- **Positional Encoding:** Adds positional information to token embeddings.
- **Encoder & Decoder Stacks:** Built with residual connections, layer normalization, and sub-layer cloning.
- **Position-wise Feed-Forward Networks:** Two-layer fully connected networks applied to each position.
- **Generator:** Produces log-probabilities over the target vocabulary.

This modular design makes it easy to customize and integrate the Transformer model into your own projects.

## Prerequisites

- Python 3.6+
- [PyTorch](https://pytorch.org/) (tested with version 1.7+)

Install PyTorch following the instructions on the [official website](https://pytorch.org/).
