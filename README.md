# HST-Light: Hierarchical Spatio-Temporal Decoupling for Adaptive Traffic Signal Control

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Official implementation of the paper **"HST-Light: Hierarchical Spatio-Temporal Decoupling for Adaptive Traffic Signal Control"**.

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
python onpolicy/scripts/train/train_sumo.py \
    --env_name SUMO \
    --algorithm_name rmappo \
    --experiment_name hst_light \
    --scenario_name [your_scenario] \
    --num_agents [agent_count] \
    --n_rollout_threads [thread_count] \
    --lr 1e-4 \               # Learning rate
    --ppo_epoch 5 \           # PPO update epochs
    --clip_param 0.2          # PPO clip parameter
```

### Evaluation
```bash
python onpolicy/scripts/train/train_sumo.py \
    --env_name SUMO \
    --algorithm_name rmappo \
    --experiment_name hst_light_eval \
    --scenario_name [your_scenario] \
    --model_dir [path_to_models] \
    --not_update True
```

## Results

![Demo Visualization](docs/demo.gif)  <!-- Add your visualization file -->

| Scenario | Average Travel Time (s) | Throughput (veh/h) | Delay Reduction |
|----------|-------------------------|--------------------|-----------------|
| Grid 4x4 | 125.3 ± 2.1             | 980 ± 15           | 22.5%           |
| Arterial | 89.7 ± 1.8              | 1205 ± 22          | 18.3%           |

## FAQ

### Q: How to add custom scenarios?
A: Place your SUMO scenario files in `scenarios/` folder and update the config file.

### Q: Training is too slow?
A: Try reducing `n_rollout_threads` or using a smaller scenario first.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
