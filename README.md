# Highly Rewarding

A framework for training large language models with custom loss functions and model architectures. This project provides a flexible and extensible training pipeline that supports various model types, datasets, and training configurations.

## Setup

1. Clone the repository:
  
```bash
git clone https://github.com/efrick2002/highly-rewarding.git
```

2. Run the setup script:

```bash
cd highly-rewarding
source setup.sh
```

> Note: the setup script installs `uv`. The environment in activated with `source .venv/bin/activate`.

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
deepspeed --num_gpus=8 train.py -c configs/bt_debug.yaml
```

## Configuration

The training configuration should be specified in a YAML file.

See [configs](./configs/) for examples.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


