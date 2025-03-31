# LLaVA: Large Language and Vision Assistant

## Project Overview

LLaVA (Large Language and Vision Assistant) is a multimodal AI model that combines large language models with vision capabilities. It enables the understanding and reasoning of visual information in a conversational manner. The project provides a framework for training, evaluation, and serving of vision-language models.

## Project Structure

```
llava/
├── __init__.py               # Package initialization
├── constants.py              # Constants and configurations
├── conversation.py           # Conversation handling and formatting
├── mm_utils.py               # Multimodal utilities
├── utils.py                  # General utilities
├── model/                    # Core model architecture
│   ├── __init__.py
│   ├── apply_delta.py        # Delta weights application
│   ├── builder.py            # Model building utilities
│   ├── consolidate.py        # Model consolidation
│   ├── llava_arch.py         # Main LLaVA architecture
│   ├── make_delta.py         # Delta weights creation
│   ├── utils.py              # Model-specific utilities
│   ├── language_model/       # LLM components
│   ├── multimodal_encoder/   # Vision encoders
│   └── multimodal_projector/ # Vision-language projectors
├── train/                    # Training code
│   ├── llama_flash_attn_monkey_patch.py
│   ├── llama_xformers_attn_monkey_patch.py
│   ├── llava_trainer.py      # Custom trainer
│   ├── train.py              # Main training script
│   ├── train_mem.py
│   └── train_xformers.py
├── serve/                    # Serving and inference
│   ├── __init__.py
│   ├── cli.py                # Command-line interface
│   ├── controller.py         # Controller for distributed serving
│   ├── examples/             # Example applications
│   ├── gradio_web_server.py  # Web UI with Gradio
│   ├── model_worker.py       # Worker for model inference
│   ├── register_worker.py    # Worker registration
│   ├── sglang_worker.py      # SGLang integration
│   └── test_message.py       # Testing utilities
└── eval/                     # Evaluation scripts
    ├── eval_gpt_review.py
    ├── eval_pope.py
    ├── eval_science_qa.py
    ├── model_qa.py
    ├── model_vqa.py
    ├── run_llava.py
    ├── table/                # Table-based evaluation
    └── webpage/              # Webpage-based evaluation
```

## Core Components

### 1. Conversation Management (`conversation.py`)

The conversation module manages the format and flow of conversations, supporting multiple conversation styles:

- `SeparatorStyle` enum for different separator styles (SINGLE, TWO, MPT, PLAIN, LLAMA_2)
- `Conversation` class that handles:
  - Message history management
  - Image processing and embedding
  - Prompt construction for different model types
  - Conversion to chatbot format

### 2. Model Architecture (`model/llava_arch.py`)

The core LLaVA architecture consists of:

- `LlavaMetaModel`: Base class for model definition
  - Vision tower integration
  - Vision-language projection
  - Initialization of vision modules

- `LlavaMetaForCausalLM`: Abstract base class for causal language modeling
  - Image encoding
  - Multimodal input preparation
  - Handling of image tokens and features

The model architecture follows a two-tower design:
1. Vision tower: Processes images and extracts visual features
2. Language model: Generates text based on the combined vision and language inputs

### 3. Model Builder (`model/builder.py`)

The builder module handles model loading and configuration:

- `load_pretrained_model`: Factory function for loading models
  - Support for different quantization levels (4-bit, 8-bit)
  - LoRA weights handling
  - Flash attention support
  - Model merging for LoRA fine-tuning

### 4. Training (`train/train.py`)

The training module provides:

- Data arguments, model arguments, and training arguments
- Dataset preparation with image and text handling
- Training loop with multimodal support
- LoRA adaptation for efficient fine-tuning
- Model saving utilities

Key classes:
- `ModelArguments`: Configuration for model architecture
- `DataArguments`: Configuration for training data
- `TrainingArguments`: Configuration for training process
- `LazySupervisedDataset`: Dataset implementation for supervised fine-tuning
- `DataCollatorForSupervisedDataset`: Batching and padding logic

### 5. Serving (`serve/`)

The serving infrastructure consists of:

- `controller.py`: Central controller for managing workers
- `model_worker.py`: Workers for model inference
  - Streaming text generation
  - Image processing
  - Multimodal input handling
- `gradio_web_server.py`: Web UI for user interaction
  - Conversation management
  - Model selection
  - Image upload and processing

### 6. Evaluation (`eval/`)

Evaluation scripts for different tasks:

- Visual question answering (VQA)
- Science QA
- POPE (Prompt Of Pointing Error)
- GPT review-based evaluation

## Dependencies and Code Flow

### Model Initialization and Loading

1. `load_pretrained_model` in `model/builder.py` creates the model instance based on the provided configuration
2. Vision tower is loaded from `vision_tower` parameter
3. MM projector (multimodal projector) is initialized to connect vision and language models
4. Tokenizer is prepared with special tokens for image handling

### Training Pipeline

1. Arguments are parsed and validated in `train.py`
2. Dataset is created with `make_supervised_data_module`
3. Model is initialized with `load_pretrained_model`
4. Training is performed using `LLaVATrainer` (extended from HuggingFace's `Trainer`)
5. Model checkpoints and adapters are saved during/after training

### Inference Pipeline

1. `model_worker.py` loads the model and registers with the controller
2. `controller.py` manages available workers and distributes requests
3. `gradio_web_server.py` provides the user interface and sends requests to the controller
4. When a request is received:
   - Images are processed and encoded
   - Text is tokenized and combined with image features
   - Model generates a response using the prepared inputs
   - Response is streamed back to the user

### Image Processing

1. Images are loaded and processed by the image processor
2. Vision tower extracts features from the processed images
3. MM projector projects visual features to the language model's embedding space
4. Image tokens in the text are replaced with the embedded visual features
5. Combined embeddings are passed to the language model for generation

## Key Features and Capabilities

- Multimodal understanding: Combines vision and language capabilities
- Conversational interface: Natural language interaction with images
- Efficient fine-tuning: Support for LoRA and QLoRA adaptation
- Distributed serving: Controller-worker architecture for scaling
- Web UI: User-friendly interface with Gradio
- Evaluation suite: Comprehensive evaluation on multiple tasks

## Important File Locations

- Main model architecture: `llava/model/llava_arch.py`
- Model building logic: `llava/model/builder.py`
- Training script: `llava/train/train.py`
- Inference worker: `llava/serve/model_worker.py`
- Web UI: `llava/serve/gradio_web_server.py`
- Conversation handling: `llava/conversation.py`

## System Requirements and Dependencies

Core dependencies include:
- PyTorch
- Transformers (HuggingFace)
- PEFT (Parameter-Efficient Fine-Tuning)
- Accelerate
- Gradio (for web UI)
- Flash Attention (optional, for performance)

## Deployment Architecture

For deployment, LLaVA uses a controller-worker architecture:
1. Controller manages available workers and routing
2. Workers handle model inference
3. Web server provides user interface
4. Communication between components via HTTP API

This architecture enables:
- Load balancing across multiple workers
- Scaling to handle multiple requests
- Fault tolerance with worker health checks 