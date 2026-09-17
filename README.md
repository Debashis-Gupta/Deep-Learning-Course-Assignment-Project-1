# MNIST Classification with a Neural Network from Scratch

This deep learning project implements a fully connected neural network for handwritten digit classification using NumPy. Forward propagation, cross-entropy loss, backpropagation, and mini-batch stochastic gradient descent are implemented manually. Jupyter notebooks investigate learning rate, initialization, activation functions, network depth, training-set size, robustness, and ensembles.

PyTorch and torchvision provide MNIST loading, checkpoint serialization, and an autograd reference for correctness checks. The core model trains on the CPU using NumPy.

## Model and data

- **Baseline architecture:** 784 input features → 128 hidden units → 10 output classes.
- **Activations:** ReLU, tanh, or sigmoid in the hidden layer; softmax for class probabilities.
- **Initialization:** zero, small Gaussian, Xavier, He, or large Gaussian weights; zero biases.
- **Preprocessing:** flatten each 28 × 28 image and scale pixels to `[0, 1]` using `float64` arrays.
- **Default split:** 50,000 training and 10,000 validation examples from the MNIST training set, plus the separate 10,000-example test set. The shuffled training/validation split uses seed 42 and is not stratified.
- **Repeated experiments:** three model seeds generated from base seed 42, with mean and sample standard deviation reported across runs.

## Repository guide

| File | Purpose |
| --- | --- |
| [Code/scratch.py](Code/scratch.py) | Shared data preparation, model, training, evaluation, checkpointing, and plotting functions. |
| [Code/correctness_check.ipynb](Code/correctness_check.ipynb) | Data and tensor-shape checks, numerical stability, finite-difference gradients, PyTorch autograd comparisons, and tiny-batch overfitting. |
| [Code/learning_rate.ipynb](Code/learning_rate.ipynb) | Compare learning rates `0.001`, `0.01`, `0.1`, `0.5`, and `0.9`. |
| [Code/learning_rate.py](Code/learning_rate.py) | Standalone learning-rate experiment using a smaller data split. |
| [Code/initialization_effect.ipynb](Code/initialization_effect.ipynb) | Compare initialization methods and inspect network diagnostics. |
| [Code/compare_relu.ipynb](Code/compare_relu.ipynb) | Compare ReLU, tanh, and sigmoid. |
| [Code/adding_depth.ipynb](Code/adding_depth.ipynb) | Compare one, two, and three hidden layers, each with 128 units. Defines its own deeper-network implementation. |
| [Code/testing_report.ipynb](Code/testing_report.ipynb) | Select hyperparameters using validation results, save checkpoints, and evaluate the selected configuration on the test set. |
| [Code/Hypothesis_one.ipynb](Code/Hypothesis_one.ipynb) | Compare nested training subsets of 5,000, 10,000, 25,000, and 50,000 examples. |
| [Code/Hypothesis_two.ipynb](Code/Hypothesis_two.ipynb) | Evaluate robustness to shifts of 1–3 pixels with zero padding. |
| [Code/Hypothesis_three.ipynb](Code/Hypothesis_three.ipynb) | Evaluate Gaussian noise with standard deviations `0`, `0.05`, `0.10`, `0.20`, and `0.30`. |
| [Code/Hypothesis_four.ipynb](Code/Hypothesis_four.ipynb) | Compare individual models with ensembles that average class probabilities. |

## Setup

From the repository root, create a Python 3 environment and install the packages imported by the code, together with JupyterLab:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy pandas matplotlib scikit-learn torch torchvision ipython jupyterlab
```

On Windows, activate the environment with `.venv\Scripts\activate` instead. The repository does not currently pin dependency versions.

Launch JupyterLab from `Code/`:

```bash
cd Code
jupyter lab
```

Use this environment as the notebook kernel and execute notebook cells from top to bottom. In an IDE, ensure the kernel's working directory is `Code/`: imports use the local `scratch.py`, while data and output paths are relative to the working directory.

`load_mnist_data()` downloads MNIST into `Code/data/` when needed, so the first download requires internet access.

## Running the experiments

1. Start with `correctness_check.ipynb` to inspect the implementation checks.
2. Run the learning-rate, initialization, activation, and depth notebooks for the controlled comparisons. These can be run independently.
3. Run `testing_report.ipynb` to regenerate the hyperparameter search and saved models. It evaluates four initialization methods and three learning rates across three seeds: 36 training runs of 100 epochs each.
4. Run `Hypothesis_one.ipynb` for the training-size experiment. It trains and saves its own models.
5. Run `Hypothesis_two.ipynb`, `Hypothesis_three.ipynb`, and `Hypothesis_four.ipynb` using the checkpoints from the testing report. Existing checkpoints are included in the repository.

The last three notebooks currently load filenames for `small` initialization and learning rate `0.1` from `output/test_reports/checkpoints/`. They do not automatically read the selected-configuration CSV. If you change the chosen configuration or seeds, update their checkpoint-loading cells accordingly.


This script uses 5,000 training and 1,000 validation examples, whereas the learning-rate notebook uses 50,000 and 10,000. Both use three seeds and 100 epochs per learning rate. Settings are edited directly in the script or notebook; there is no command-line argument interface. Full sweeps can take substantial CPU time.

## Using the shared model

Run this example in a notebook or Python session with `Code/` as the working directory:

```python
from scratch import load_mnist_data, preprocess_data, train_model, evaluate_model

train_data, test_data = load_mnist_data()
X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data(
    train_data, test_data
)

parameters, history = train_model(
    X_train, y_train, X_val, y_val,
    hidden_size=128,
    activation="relu",
    initialization="small",
    learning_rate=0.1,
    batch_size=128,
    epochs=10,
    seed=42,
)

val_loss, val_accuracy = evaluate_model(
    X_val, y_val, parameters, activation="relu"
)
print(f"Validation loss: {val_loss:.4f}; accuracy: {val_accuracy:.2%}")
```

`train_model()` returns the final epoch's parameters and a history dictionary. With `save_model=True`, it also saves the checkpoint with the highest validation accuracy, breaking ties with lower validation loss. The returned parameters still belong to the final epoch; reload the saved checkpoint to evaluate the best epoch. Checkpoint paths and best-epoch metrics are recorded in `history`.

## Outputs and recorded results

Experiments write CSV tables, PNG figures, some PDF figures, and PyTorch `.pt` checkpoint dictionaries under `Code/output/`:

| Directory | Contents |
| --- | --- |
| `correctness_checks/` | Implementation-check reports. |
| `h1/` | Learning-rate sweep. |
| `h2/` | Initialization comparisons and diagnostics. |
| `h3/` | Activation comparisons. |
| `depth_same_neurons/` | Depth comparisons. |
| `test_reports/` | Hyperparameter rankings, training histories, final evaluation, and checkpoints. |
| `hypotheses/` | Training-size, translation, Gaussian-noise, and ensemble studies. |

The older `h1`, `h2`, and `h3` output names refer to the parameter comparisons, not the numbered `Hypothesis_*.ipynb` studies. `Code/output_prev/` holds previous outputs and is ignored by Git. Rerunning experiments can overwrite existing output files.

The committed [selected configuration](Code/output/test_reports/selected_hyperparameter_configuration.csv) uses 128 hidden units, ReLU, small Gaussian initialization, learning rate `0.1`, batch size 128, and a maximum of 100 epochs. Configurations are ranked by mean best-checkpoint validation accuracy minus its standard deviation, with mean accuracy and then mean loss used as tie-breakers.

The committed [evaluation summary](Code/output/test_reports/final_validation_test_summary.csv) records the following results across three seeds:

| Dataset | Accuracy (mean ± sample SD) | Cross-entropy loss (mean ± sample SD) |
| --- | --- | --- |
| Validation | 97.95% ± 0.06 percentage points | 0.07426 ± 0.00168 |
| Test | 97.85% ± 0.04 percentage points | 0.07170 ± 0.00014 |

These are saved experiment results; the 10-epoch example above is not intended to reproduce them. Keep the test set separate from hyperparameter selection, as done in the testing-report workflow.
