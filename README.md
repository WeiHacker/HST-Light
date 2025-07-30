# HST-Light: Hierarchical Spatio-Temporal Decoupling for Adaptive Traffic Signal Control
## Overview

HST-Light is a novel multi-agent reinforcement learning framework for adaptive traffic signal control, featuring:

- 🚦 **Hierarchical Spatio-Temporal Decoupling**: Separates spatial and temporal learning processes
- 🧠 **Transformer-based Architecture**: Captures complex traffic patterns
- 🚗 **Multi-Agent Coordination**: Enables cooperative signal control
- 🏙️ **SUMO Integration**: Works with urban traffic simulation scenarios

## Installation

### Prerequisites
- Python 3.7+
- [SUMO](https://www.eclipse.org/sumo/) ≥1.10.0
- PyTorch ≥1.8.0

### Setup
```bash
git clone https://github.com/yourusername/HST-Light.git
cd HST-Light
pip install -r requirements.txt
```

## Project Structure

```
HST-Light/
├── onpolicy/                 # Core algorithm implementation
│   ├── algorithms/           # Network architectures
│   ├── envs/                 # SUMO environment wrapper
│   ├── runner/               # Training logic
│   └── scripts/              # Training scripts
├── scenarios/                # SUMO scenario files
├── requirements.txt          # Python dependencies
└── LICENSE
```

## Usage

### Training
```bash
python onpolicy/scripts/train/train_sumo.py
```

### Evaluation
```bash
python onpolicy/scripts/train/train_sumo.py
```

## Citation

If you use HST-Light in your research, please cite our paper:

```bibtex
@article{hstlight2025,
  title={HST-Light: Hierarchical Spatio-Temporal Decoupling for Adaptive Traffic Signal Control},
  author={Your Name, Co-authors},
  journal={Conference/Journal Name},
  year={2025}
}
```
