# Highly Rewarding

A PyTorch-based framework for training large language models with custom loss functions and model architectures. This project provides a flexible and extensible training pipeline that supports various model types, datasets, and training configurations.

## Features

- Support for multiple model architectures through a registry system
- Custom loss function implementations
- Distributed training capabilities with DeepSpeed
- Integration with Weights & Biases for experiment tracking
- Flash Attention 2.5.9 support for improved training efficiency
- Configurable training pipeline through YAML configuration files

## Requirements

- Python 3.10
- CUDA-compatible GPU (recommended)
- Linux/Unix environment (for setup.sh script)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/highly-rewarding.git
cd highly-rewarding
```

2. Run the setup script:
```bash
chmod +x setup.sh
./setup.sh
```

This will:
- Install system dependencies
- Set up a Python virtual environment
- Install required Python packages including:
  - PyTorch
  - DeepSpeed
  - Transformers
  - Weights & Biases
  - Flash Attention
  - Other dependencies listed in requirements.txt

## Project Structure

- `train.py`: Main training script
- `modeling.py`: Model architecture definitions and registry
- `losses.py`: Custom loss function implementations
- `dataset.py`: Dataset handling and data loading
- `utils.py`: Utility functions
- `model_type_registry.py`: Model type registration system

## Usage

1. Configure your training parameters in a YAML config file
2. Run the training script:
```bash
python train.py --config path/to/your/config.yaml
```

## Configuration

The training configuration should be specified in a YAML file with the following structure:
```yaml
training_type: "your_training_type"
learning_rate: 1e-5
batch_size: 4
train_data_path: "path/to/training/data"
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your license information here]

