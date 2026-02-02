# SAM 3: Segment Anything with Concepts - Project Documentation

## 1. Executive Summary
SAM 3 (Segment Anything Model 3) is a foundation model for promptable segmentation in images and videos. Developed by Meta Superintelligence Labs, it extends the capabilities of SAM 2 by introducing "Concept" understanding and a fully autonomous Agentic workflow.

**Core Problem**: Traditional segmentation models struggle with open-vocabulary concepts (e.g., distinguishing "player in white" from "player in red"), maintaining identity across video frames, and reasoning about complex queries that require multi-step verification.
**Solution**: SAM 3 creates a unified architecture that:
1.  **Understands Concepts**: Recognizes 270K+ unique concepts via a custom BPE-based text encoder ([VETextEncoder](file:///Users/mck/Desktop/sam3-orginal-repo-do-not-modify/sam3/model/text_encoder_ve.py#255-331)).
2.  **Autonomous Reasoning**: Features a built-in "Agent" that uses a Multimodal LLM to iteratively prompt the segmentation model, verify results, and self-correct.
3.  **Video Tracking**: Tracks objects across frames with temporal memory, using "Masklets" to persist identity.
4.  **Unified Architecture**: Uses a [TwoWayTransformer](file:///Users/mck/Desktop/sam3-orginal-repo-do-not-modify/sam3/sam/transformer.py#17-108) with Rotary Positional Embeddings (RoPE) and a [Sam3DualViTDetNeck](file:///Users/mck/Desktop/sam3-orginal-repo-do-not-modify/sam3/model/necks.py#14-127) for multi-scale feature fusion.

## 2. System Architecture Overview
The system follows a monolithic deep learning architecture pattern extended with an Agentic outer loop.

### 2.1 Major Subsystems
-   **Vision Backbone**: A Vision Transformer (ViT) processing images into high-dimensional embeddings.
-   **Neck**: [Sam3DualViTDetNeck](file:///Users/mck/Desktop/sam3-orginal-repo-do-not-modify/sam3/model/necks.py#14-127). Adapts SimpleFPN from ViTDet, processing features at scales (4.0, 2.0, 1.0, 0.5) to feed the decoder.
-   **Text Encoder** ([VETextEncoder](file:///Users/mck/Desktop/sam3-orginal-repo-do-not-modify/sam3/model/text_encoder_ve.py#255-331)): A lightweight transformer (12 layers, 512 width) using a custom BPE tokenizer. It employs `LayerScale` for training stability and projects features to the model's dimension.
-   **Two-Way Transformer**: The core interaction engine. It consists of:
    -   **Self-Attention**: Sparse inputs (points/boxes) attend to themselves.
    -   **Cross-Attention (Token-to-Image)**: Prompts query the image features.
    -   **Cross-Attention (Image-to-Token)**: Image features update based on prompts.
    -   **RoPE**: Rotary Positional Embeddings applied to queries and keys for precise relative positioning.
-   **Mask Decoder**: Predicts segmentation masks from transformer embeddings. It outputs:
    -   **Masks**: The pixel-wise segmentation.
    -   **IoU Scores**: Predicted quality of the mask.
    -   **Object Scores**: Probability that the object exists (presence detection).
-   **The Agent**: An outer loop utilizing an MLLM (e.g., Qwen) to "drive" SAM 3, verifying masks via visual inspection tools ([viz.py](file:///Users/mck/Desktop/sam3-orginal-repo-do-not-modify/sam3/agent/viz.py)).

## 3. Repository Structure
```
.
├── sam3/                   # Core Python Package
│   ├── agent/              # Autonomous Agent Logic (MLLM + SAM Loop)
│   ├── model/              # Neural Network Definitions
│   │   ├── sam3_image.py   # Grounding & Segmentation Logic
│   │   ├── sam3_tracker_base.py # Base Video Tracking Logic
│   │   ├── necks.py        # FPN / Feature Fusion
│   │   ├── geometry_encoders.py # Prompt encoding (box->point, RoPE)
│   │   ├── memory.py       # Video Memory Bank Implementation
│   │   ├── tokenizer_ve.py # Custom BPE Tokenizer
│   │   ├── sam/            # Core SAM Transformer & Decoder
│   │   │   ├── transformer.py  # TwoWayTransformer with RoPE
│   │   │   └── mask_decoder.py # Mask Prediction Heads
│   │   └── ...
│   ├── train/              # Training Infrastructure
│   │   ├── loss/           # Loss Functions (Focal, Dice, IoU)
│   │   ├── data/           # Datasets (Sam3ImageDataset)
│   │   └── trainer.py      # Main DDP Loop
│   ├── eval/               # Evaluation Tools (YTVIS, COCO)
│   └── perflib/            # Compilation Utilities
└── ...
```

## 4. Deep Dive: Core Concepts & Algorithms

### 4.1 Stability & Dynamic Multimask
SAM 3 introduces a stability mechanism in `MaskDecoder._dynamic_multimask_via_stability`. rather than blindly trusting the primary mask output, it calculates a **Stability Score**:
$$ \text{Stability} = \frac{\text{Area}(M > 0.05)}{\text{Area}(M > -0.05)} $$
If the primary mask is unstable (score < threshold), the model dynamically falls back to the "One-to-Many" (multimask) outputs, selecting the one with the highest predicted IoU.

### 4.2 The "FindQuery" Paradigm
Training is structured around [FindQuery](file:///Users/mck/Desktop/sam3-orginal-repo-do-not-modify/sam3/train/data/sam3_image_dataset.py#58-89) objects ([sam3_image_dataset.py](file:///Users/mck/Desktop/sam3-orginal-repo-do-not-modify/sam3/train/data/sam3_image_dataset.py)). Instead of standard annotations, each sample is a query:
-   **Query Text**: "Find the red cup."
-   **Exhaustivity**: Flags (`is_exhaustive`, `is_pixel_exhaustive`) tell the loss function whether to penalize the model for missing *other* similar objects.

### 4.3 Two-Way Transformer Details
Implemented in [sam3/sam/transformer.py](file:///Users/mck/Desktop/sam3-orginal-repo-do-not-modify/sam3/sam/transformer.py).
-   **Sparse-to-Dense**: Point embeddings (sparse) attend to Image embeddings (dense).
-   **Dense-to-Sparse**: Image embeddings attend back to Point embeddings.

## 5. Agentic Workflow
The agent ([agent_core.py](file:///Users/mck/Desktop/sam3-orginal-repo-do-not-modify/sam3/agent/agent_core.py)) implements a **Generate-Verify-Refine** loop:
1.  **Plan**: MLLM decides which tool to use (`segment_phrase`).
2.  **Act**: SAM 3 generates masks.
3.  **Verify**: The agent uses `visualize` to crop and zoom into each mask. It feeds these crops back to the MLLM.
4.  **Refine**: If the MLLM rejects a mask ("This is a shadow, not a car"), the mask is discarded.

## 6. Data & Control Flow

### 6.1 Training Data Pipeline
1.  [Sam3ImageDataset](file:///Users/mck/Desktop/sam3-orginal-repo-do-not-modify/sam3/train/data/sam3_image_dataset.py#437-529) loads an image and JSON annotations.
2.  It constructs `FindQueries` (e.g., positive queries for present objects, negative queries for absent ones).
3.  [Sam3LossWrapper](file:///Users/mck/Desktop/sam3-orginal-repo-do-not-modify/sam3/train/loss/sam3_loss.py#37-204) computes Focal, Dice, and IoU losses.

### 6.2 Inference Flow
1.  **User**: "Segment the runner."
2.  **Text Encoder**: Embeds "runner" -> $E_{text}$.
3.  **Vision Encoder**: Embeds Image -> $E_{img}$.
4.  **Neck**: Fuses features (4x, 2x, 1x) -> $F_{multi}$.
5.  **Prompt Encoder**: Fuses $E_{text}$ with $F_{multi}$.
6.  **Decoder**: Predicts 3 masks (Whole person, Upper body, Face).
7.  **Selection**: `Stability Score` heuristic picks the best mask.
8.  **Tracking (Video)**: Mask is encoded into Memory Bank; [Sam3TrackerBase](file:///Users/mck/Desktop/sam3-orginal-repo-do-not-modify/sam3/model/sam3_tracker_base.py#26-1175) propagates it to next frame.

## 7. Deep Implementation Notes

### 7.1 Text Tokenization ([tokenizer_ve.py](file:///Users/mck/Desktop/sam3-orginal-repo-do-not-modify/sam3/model/tokenizer_ve.py))
-   **Type**: Custom Byte-Pair Encoding (BPE), identical to CLIP.
-   **Vocabulary**: Loaded from [assets/bpe_simple_vocab_16e6.txt.gz](file:///Users/mck/Desktop/sam3-orginal-repo-do-not-modify/sam3/assets/bpe_simple_vocab_16e6.txt.gz).
-   **Context Length**: Hardcoded to `77` tokens.
-   **Special Tokens**: `<start_of_text>`, `<end_of_text>`.
-   **Cleaning**: Uses `ftfy` to fix unicode and `html.unescape`.

### 7.2 Memory Module ([memory.py](file:///Users/mck/Desktop/sam3-orginal-repo-do-not-modify/sam3/model/memory.py))
Used for video tracking.
-   **Downsampler**: [SimpleMaskDownSampler](file:///Users/mck/Desktop/sam3-orginal-repo-do-not-modify/sam3/model/memory.py#21-81). Progressively reduces mask resolution using strided convolutions (stride 4^levels).
-   **Fusion**: [SimpleFuser](file:///Users/mck/Desktop/sam3-orginal-repo-do-not-modify/sam3/model/memory.py#142-158) combines visual features with downsampled masks before storing them in the memory bank.

### 7.3 Geometry Encoders & Position Encoding
-   **Prompt Class**: Unified wrapping of boxes, points, and masks.
-   **Box Encoding**: Boxes are converted to Top-Left/Bottom-Right point pairs or encoded via RoI Align.
-   **Sine PosEnc**: [PositionEmbeddingSine](file:///Users/mck/Desktop/sam3-orginal-repo-do-not-modify/sam3/model/position_encoding.py#12-127) (temp=10000) creates fixed sinusoidal embeddings. It includes specialized [encode_boxes](file:///Users/mck/Desktop/sam3-orginal-repo-do-not-modify/sam3/model/position_encoding.py#73-78) and [encode_points](file:///Users/mck/Desktop/sam3-orginal-repo-do-not-modify/sam3/model/position_encoding.py#81-89) methods to generate high-freq spatial signatures for prompts.

### 7.4 Sam3TrackerBase ([sam3_tracker_base.py](file:///Users/mck/Desktop/sam3-orginal-repo-do-not-modify/sam3/model/sam3_tracker_base.py))
-   **Memory Management**: Manages a FIFO buffer of past frame features.
-   **Compilation**: Explicitly calls `sam3.perflib.compile.compile_wrapper` on backbone, memory encoder, and mask decoder to enable `torch.compile` optimizations (max-autotune).
-   **Scores**: Uses `NO_OBJ_SCORE = -1024.0` as a sentinel for missing objects.

### 7.5 Evaluation ([eval/ytvis_eval.py](file:///Users/mck/Desktop/sam3-orginal-repo-do-not-modify/sam3/eval/ytvis_eval.py))
-   **Metric**: 3D Space-Time IoU.
-   **Logic**: [iou_masklets](file:///Users/mck/Desktop/sam3-orginal-repo-do-not-modify/sam3/eval/ytvis_eval.py#117-142) computes $\sum(\text{Inter}) / \sum(\text{Union})$ across all frames.
-   **Sync**: [YTVISResultsWriter](file:///Users/mck/Desktop/sam3-orginal-repo-do-not-modify/sam3/eval/ytvis_eval.py#160-412) handles multi-GPU synchronization, deduplicating predictions based on [(video_id, category_id)](file:///Users/mck/Desktop/sam3-orginal-repo-do-not-modify/sam3/model/utils/misc.py#18-21) tuples to avoid double-counting in distributed dataloaders.

## 8. Configuration & Environment Variables
-   **Hydra**: Primary config system.
-   **Trainer**:
    -   `normalization`: Global vs Local loss normalization.
    -   `normalize_by_valid_object_num`: Critical for balancing loss batches with few vs many objects.

## 9. Debugging & Maintenance Guide
### Compilation Issues
Use `sam3.perflib.compile.shape_logging_wrapper` to detect if variable input shapes are causing `torch.compile` to re-trigger.

### Loss Spikes
Check `FindQuery.is_exhaustive`. If training on sparse datasets, ensure you aren't penalizing the model for "missing" unannotated objects (set `is_exhaustive=False`).

## 10. Glossary
-   **RoPE**: Rotary Positional Embeddings.
-   **O2O / O2M**: One-to-One vs One-to-Many predictions.
-   **FindQuery**: The fundamental data unit.
-   **Masklet**: Temporal mask sequence.
-   **Neck**: Feature Pyramid Network connecting backbone to decoder.
