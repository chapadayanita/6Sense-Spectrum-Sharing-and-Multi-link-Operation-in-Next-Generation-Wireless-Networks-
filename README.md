# 6Sense: Spectrum Sharing and Multi-link Operation in Next-Generation Wireless Networks

## Overview

6Sense is an AI-driven dynamic spectrum sharing framework designed for next-generation wireless communication systems. The framework intelligently predicts future network traffic and dynamically allocates spectrum resources among multiple Mobile Network Operators (MNOs) using hierarchical spectrum orchestration, fair resource sharing, and proportional fair scheduling.

The current implementation focuses on intelligent spectrum allocation and spectrum sharing using a hybrid deep learning traffic prediction model trained on multi-city real-world traffic datasets.

The framework combines:
- AI-driven traffic prediction
- Dynamic channel allocation
- Priority-aware spectrum orchestration
- Channel preemption
- CQI-aware RB demand estimation
- Fair RB spectrum sharing
- PF scheduling
- Real-time adaptive resource allocation

This project represents the initial implementation phase of the proposed 6Sense architecture for future 5G and 6G wireless networks.

---

# Key Features

- Hybrid deep learning-based traffic prediction
- Multi-MNO spectrum orchestration
- Dynamic channel allocation
- Priority-aware channel preemption
- CQI and SINR-aware RB demand estimation
- Fair shared RB pooling mechanism
- Proportional Fair (PF) scheduling
- Real-time dynamic resource adaptation
- Throughput visualization and logging
- Universal traffic prediction model

---

# Current Research Scope

The current implementation includes:

## Implemented Components

- AI traffic prediction framework
- Universal multi-city traffic model
- Dynamic spectrum allocation
- Channel-level spectrum sharing
- Priority-based preemption
- Fair RB sharing pool
- CQI-aware scheduling
- PF-based user allocation
- Throughput analysis and visualization

---

# Future Research Extensions

The next implementation phases will include:

- Multi-link operation
- Graph Neural Networks (GNN)
- PAL Tier-1 and Tier-2 orchestration
- Reinforcement learning-based optimization
- O-RAN integration
- Network slicing
- Distributed spectrum intelligence
- Multi-cell orchestration
- Topology-aware spectrum allocation

---

# System Workflow

```text
Traffic Dataset
↓
Hybrid AI Traffic Prediction
↓
Traffic Demand Estimation
↓
Priority-Based Channel Allocation
↓
Channel Preemption
↓
RB Allocation
↓
Fair RB Spectrum Sharing
↓
PF Scheduling
↓
Throughput Calculation
↓
Logging and Visualization
```

---

# AI Model Architecture

The framework uses a hybrid deep learning model consisting of:

| Component | Purpose |
|---|---|
| CNN | Spatial feature extraction |
| Temporal CNN | Time-series traffic learning |
| ConvLSTM | Spatio-temporal dependency learning |
| Attention Mechanism | Important timestamp focus |
| Fully Connected Layers | Future traffic prediction |

---

# Why Hybrid AI?

Wireless traffic contains:
- spatial dependencies
- temporal evolution
- hotspot propagation
- dynamic congestion patterns

A single deep learning model cannot efficiently capture all these characteristics.

Therefore:
- CNN extracts spatial hotspot features
- Temporal CNN learns temporal traffic evolution
- ConvLSTM captures spatio-temporal dependencies
- Attention identifies important traffic timestamps

This improves traffic prediction accuracy for intelligent spectrum orchestration.

---

# Datasets Used

The framework uses real-world datasets from:

- San Diego
- North Carolina
- New Jersey

Each dataset contains:
- Traffic density information
- User density information
- Point of Interest (POI) information

These datasets are used to train a universal traffic prediction model capable of adapting across multiple traffic environments.

---

# Spectrum Sharing Architecture

The framework implements:

## Channel-Level Spectrum Sharing
- Dynamic channel allocation among MNOs
- Priority-aware channel ownership
- Channel preemption support

## RB-Level Spectrum Sharing
- Demand-aware RB allocation
- Shared RB pooling mechanism
- Fair RB borrowing across MNOs

## PF Scheduling
- CQI-aware scheduling
- SINR-aware throughput optimization
- Fair user-level resource allocation

---

# Core Algorithms

## Algorithm 1: AI Traffic Prediction

1. Load traffic, user, and POI datasets
2. Create spatio-temporal tensors
3. Extract spatial features using CNN
4. Learn temporal evolution using Temporal CNN
5. Capture spatio-temporal dependencies using ConvLSTM
6. Apply Attention mechanism
7. Predict future traffic demand

---

## Algorithm 2: Dynamic Spectrum Allocation

1. Estimate future RB demand
2. Convert RB demand into channel demand
3. Sort MNOs based on priority
4. Allocate free channels dynamically
5. Apply channel preemption if needed
6. Update ownership and lease expiry

---

## Algorithm 3: Fair RB Sharing

1. Allocate RBs inside owned channels
2. Identify unused RBs
3. Create shared RB pool
4. Compute unmet RB demand
5. Assign weighted sharing quota
6. Borrow RBs fairly from shared pool
7. Apply PF scheduling on borrowed RBs

---

# Repository Structure

```text
6Sense-Spectrum-Sharing/
│
├── datasets/
│   ├── 24_TrafficSD_short64.json
│   ├── 24_TrafficNC_short64.json
│   ├── 24_TrafficNJ_short64.json
│   ├── TrafficSD_user64.json
│   ├── TrafficNC_user64.json
│   ├── TrafficNJ_user64.json
│   ├── TrafficSD_poi64.json
│   ├── TrafficNC_poi64.json
│   └── TrafficNJ_poi64.json
│
├── models/
│   └── best_universal_model.pt
│
├── src/
│   └── main.py
│
├── outputs/
│
├── README.md
├── requirements.txt
└── LICENSE
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/6Sense-Spectrum-Sharing.git

cd 6Sense-Spectrum-Sharing
```

---

# Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run the Framework

```bash
python src/main.py
```

---

# Example Inputs

```text
Number of MNOs: 3
Number of Channels: 6
RBs per Channel: 5
Time Slots: 5
```

---

# Example Outputs

The framework generates:
- Dynamic channel allocation logs
- Channel preemption events
- Shared RB borrowing summaries
- Throughput graphs
- Spectrum utilization plots
- CSV allocation logs

---

# Current Output Metrics

- RB utilization
- Shared RB borrowing
- User throughput
- Channel ownership
- Spectrum reuse efficiency
- Allocation fairness

---

# Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- Matplotlib
- SQLite

---

# Applications

- 5G Networks
- 6G Networks
- Dynamic Spectrum Access
- O-RAN
- AI-Native RAN
- Spectrum Pooling
- Intelligent Wireless Systems

---

# Authors

Dayanita Chapa

---

# License

MIT License
