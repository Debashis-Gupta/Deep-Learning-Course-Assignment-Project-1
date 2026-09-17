import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import torch
from torchvision.datasets import MNIST
from sklearn.model_selection import train_test_split
from pathlib import Path
try:
    from IPython.display import display
except ImportError:
    def display(obj):
        print(obj)


## Global Variable Declaration 
SPLIT_SEED = 42

def load_mnist_data():
    train_dataset = MNIST(root='./data', train=True, download=True)
    test_dataset = MNIST(root='./data', train=False, download=True)
    return train_dataset, test_dataset

def split_data(X,y, train_size, test_size, random_state, shuffle=True, stratify=None):
  return train_test_split(X,y, train_size=train_size, test_size=test_size, random_state=random_state, shuffle=shuffle, stratify=stratify)


def preprocess_data(train_dataset,test_dataset, train_size=50000, test_size=10000, SPLIT_SEED=SPLIT_SEED):
    X_train_full = train_dataset.data.numpy()
    y_train_full = train_dataset.targets.numpy()
    X_test = test_dataset.data.numpy()
    y_test = test_dataset.targets.numpy()
    print("Original training images:", X_train_full.shape)
    print("Original training labels:", y_train_full.shape)
    print("Original test images:", X_test.shape)
    print("Original test labels:", y_test.shape)

    print("Splitting the training data into training and validation sets...")
    X_train, X_val, y_train, y_val = split_data(X_train_full,y_train_full, train_size=train_size, test_size=test_size, random_state=SPLIT_SEED)
    print("Training images:", X_train.shape)
    print("Training labels:", y_train.shape)
    print("Validation images:", X_val.shape)
    print("Validation labels:", y_val.shape)
    print("Test images:", X_test.shape)
    print("Test labels:", y_test.shape)
    X_train = X_train.reshape(X_train.shape[0],-1).astype(np.float64)
    X_val = X_val.reshape(X_val.shape[0],-1).astype(np.float64)
    X_test = X_test.reshape(X_test.shape[0],-1).astype(np.float64)
    print("Before Normalization:")
    print(f"Training data range: {X_train.min()} to {X_train.max()}")
    print(f"Validation data range: {X_val.min()} to {X_val.max()}")
    print(f"Test data range: {X_test.min()} to {X_test.max()}")
    print(f"Performing normalization of the data...")
    X_train = X_train / 255.0
    X_val = X_val / 255.0
    X_test = X_test / 255.0
    print("After Normalization:")
    print(f"Training data range: {X_train.min()} to {X_train.max()}")
    print(f"Validation data range: {X_val.min()} to {X_val.max()}")
    print(f"Test data range: {X_test.min()} to {X_test.max()}")
    return X_train, y_train, X_val, y_val, X_test, y_test


    
# def initialize_parameters(
#     input_size=784,
#     hidden_size=128,
#     output_size=10,
#     initialization="he",
#     seed=42
# ):
#     rng = np.random.default_rng(seed)

#     if initialization == "small":
#         scale1 = 0.01
#         scale2 = 0.01

#     elif initialization == "xavier":
#         scale1 = np.sqrt(1.0 / input_size)
#         scale2 = np.sqrt(1.0 / hidden_size)

#     elif initialization == "he":
#         scale1 = np.sqrt(2.0 / input_size)
#         scale2 = np.sqrt(2.0 / hidden_size)

#     elif initialization == "large":
#         scale1 = 1.0
#         scale2 = 1.0

#     elif initialization == "zero":
#         scale1 = 0.0
#         scale2 = 0.0

#     else:
#         raise ValueError("Unknown initialization method")

#     W1 = rng.standard_normal(
#         size=(input_size, hidden_size)
#     ) * scale1

#     b1 = np.zeros((1, hidden_size))

#     W2 = rng.standard_normal(
#         size=(hidden_size, output_size)
#     ) * scale2

#     b2 = np.zeros((1, output_size))

#     parameters = {
#         "W1": W1.astype(np.float64),
#         "b1": b1.astype(np.float64),
#         "W2": W2.astype(np.float64),
#         "b2": b2.astype(np.float64)
#     }

#     return parameters
def initialize_parameters(
    input_size,
    hidden_size,
    output_size,
    initialization="he",
    seed=SPLIT_SEED
):
    rng = np.random.default_rng(seed)

    if initialization == "zero":
        W1 = np.zeros(
            (hidden_size, input_size)
        )
        W2 = np.zeros(
            (output_size, hidden_size)
        )

    elif initialization == "small":
        W1 = rng.normal(
            0.0,
            0.01,
            size=(hidden_size, input_size)
        )
        W2 = rng.normal(
            0.0,
            0.01,
            size=(output_size, hidden_size)
        )

    elif initialization == "xavier":
        W1 = rng.normal(
            0.0,
            np.sqrt(1.0 / input_size),
            size=(hidden_size, input_size)
        )
        W2 = rng.normal(
            0.0,
            np.sqrt(1.0 / hidden_size),
            size=(output_size, hidden_size)
        )

    elif initialization == "he":
        W1 = rng.normal(
            0.0,
            np.sqrt(2.0 / input_size),
            size=(hidden_size, input_size)
        )
        W2 = rng.normal(
            0.0,
            np.sqrt(2.0 / hidden_size),
            size=(output_size, hidden_size)
        )

    elif initialization == "large":
        W1 = rng.normal(
            0.0,
            1.0,
            size=(hidden_size, input_size)
        )
        W2 = rng.normal(
            0.0,
            1.0,
            size=(output_size, hidden_size)
        )

    else:
        raise ValueError(
            "Unknown initialization method: "
            f"{initialization}"
        )

    b1 = np.zeros(hidden_size)
    b2 = np.zeros(output_size)

    return {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }

def activation_forward(Z, activation):
    if activation == "relu":
        A = np.maximum(0.0, Z)

    elif activation == "tanh":
        A = np.tanh(Z)

    elif activation == "sigmoid":
        A = np.empty_like(Z)

        positive = Z >= 0
        negative = ~positive

        A[positive] = 1.0 / (
            1.0 + np.exp(-Z[positive])
        )

        exp_z = np.exp(Z[negative])
        A[negative] = exp_z / (1.0 + exp_z)

    else:
        raise ValueError(
            "activation must be 'relu', 'tanh', or 'sigmoid'"
        )

    return A

def activation_derivative(Z,activation):
  if activation=="relu":
    derivative = np.where(Z > 0, 1.0, 0.0).astype(Z.dtype)
  elif activation=="tanh":
    A = np.tanh(Z)
    derivative = 1.0 - np.square(A)
  elif activation=="sigmoid":
    A = activation_forward(Z,activation)
    derivative = A * (1.0 - A)
  else:
    raise ValueError("activation must be 'relu', 'tanh', or 'sigmoid'")
  return derivative

def softmax(logits):
  shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
  exp_logits = np.exp(shifted_logits)
  probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
  return probabilities

def cross_entropy_loss(logits,y):
  batch_size = logits.shape[0]
  shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
  #calculate the log of the softmax denominator
  log_sum_exp = np.log(np.sum(np.exp(shifted_logits),axis=1,keepdims=True))
  log_prob_abilities = shifted_logits - log_sum_exp
  correct_log_probabilities = log_prob_abilities[
        np.arange(batch_size),
        y
    ]
  loss = -np.mean(correct_log_probabilities)
  return loss
  
  
# def forward_pass(X, parameters, activation):
#   W1 = parameters["W1"]
#   b1 = parameters["b1"]
#   W2 = parameters["W2"]
#   b2 = parameters["b2"]

#   ## linear transformation into the hid layer
#   Z1 = np.dot(X,W1) + b1
#   ## activation function
#   A1 = activation_forward(Z1,activation) 
#   ## Linear trans into hid2 later
#   logits = np.dot(A1,W2) + b2
#   probabilities = softmax(logits)

#   cache = {
#       "X":X,
#       "Z1": Z1,
#       "A1": A1,
#       "logits": logits,
#       "probabilities": probabilities
#   }

#   return probabilities, cache

def forward_pass(
    X,
    parameters,
    activation="relu"
):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = X @ W1.T + b1

    A1 = activation_forward(
        Z1,
        activation
    )

    logits = A1 @ W2.T + b2

    probabilities = softmax(
        logits
    )

    cache = {
        "X": X,
        "Z1": Z1,
        "A1": A1,
        "logits": logits,
        "probabilities": probabilities
    }

    return probabilities, cache

# def backward_pass(y,parameters,cache,activation):
#   W2 = parameters["W2"]
#   A1 = cache["A1"]
#   Z1 = cache["Z1"]
#   X = cache["X"]
#   probabilities = cache["probabilities"]
#   batch_size = X.shape[0]
#   dlogits = probabilities.copy()
#   dlogits[np.arange(batch_size),y] -= 1.0
#   dlogits /= batch_size
#   dW2 = np.dot(A1.T,dlogits)
#   db2 = np.sum(dlogits,axis=0,keepdims=True)
#   dA1 = np.dot(dlogits,W2.T)
#   dZ1 = dA1 * activation_derivative(Z1,activation)
#   dW1 = np.dot(X.T,dZ1)
#   db1 = np.sum(dZ1,axis=0,keepdims=True)
#   grads = {
#       "dW1": dW1,
#       "db1": db1,
#       "dW2": dW2,
#       "db2": db2
#   }
#   return grads
def backward_pass(
    y,
    parameters,
    cache,
    activation="relu"
):
    W2 = parameters["W2"]

    X = cache["X"]
    Z1 = cache["Z1"]
    A1 = cache["A1"]
    probabilities = cache[
        "probabilities"
    ]

    batch_size = X.shape[0]

    delta2 = probabilities.copy()

    delta2[
        np.arange(batch_size),
        y
    ] -= 1.0

    delta2 /= batch_size

    dW2 = delta2.T @ A1

    db2 = np.sum(
        delta2,
        axis=0
    )

    delta1 = (
        delta2 @ W2
    ) * activation_derivative(
        Z1,
        activation
    )

    dW1 = delta1.T @ X

    db1 = np.sum(
        delta1,
        axis=0
    )

    gradients = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2
    }

    return gradients

def update_parameters(
    parameters,
    gradients,
    learning_rate
):
    for parameter_name in ["W1", "b1", "W2", "b2"]:
        gradient_name = "d" + parameter_name

        parameters[parameter_name] = (
            parameters[parameter_name]
            - learning_rate * gradients[gradient_name]
        )

    return parameters

def evaluate_model(X,y,parameters,activation,evaluation_batch_size=1000):
  total_loss =0.0
  total_correct =0
  total_examples = X.shape[0]
  for start in range(0,total_examples,evaluation_batch_size):
    end = min(start+evaluation_batch_size,total_examples)
    X_batch = X[start:end]
    y_batch = y[start:end]
    probabilities,cache = forward_pass(X_batch,parameters,activation)
    batch_loss = cross_entropy_loss(cache["logits"],y_batch)
    predictions = np.argmax(probabilities,axis=1)
    total_loss += (batch_loss * X_batch.shape[0])
    total_correct += np.sum(predictions == y_batch)
  
  average_loss = total_loss /total_examples
  accuracy = total_correct / total_examples 
  return average_loss,accuracy


# def train_model(X_train, y_train, X_val, y_val, 
#                 hidden_size=128,activation='relu',initialization='he',learning_rate=0.1,
#                 batch_size =128, epochs=10, seed=SPLIT_SEED, verbose=True):
#   rng = np.random.default_rng(seed)
#   parameters = initialize_parameters(input_size=X_train.shape[1],hidden_size=hidden_size,output_size=10,
#                                      initialization=initialization,seed=seed)
#   history = {
#       "train_loss" : [],
#       "train_accuracy" : [],
#       "val_loss" : [],
#       "val_accuracy" : []
#   }
#   number_of_examples = X_train.shape[0]
#   for epoch in range(epochs):
#     shuffled_indices = rng.permutation(number_of_examples)
#     epoch_loss_sum = 0.0
#     epoch_correct =0
#     for start in range(0, number_of_examples, batch_size):
#       end = min(start+batch_size,number_of_examples)
#       batch_indices = shuffled_indices[start:end]
#       X_batch = X_train[batch_indices]
#       y_batch = y_train[batch_indices]
#       probabilities, cache = forward_pass(X_batch,parameters,activation)
#       batch_loss = cross_entropy_loss(
#           cache['logits'],y_batch
#       )
#       ## stops at unsatable state
#       if not np.isfinite(batch_loss):
#         raise FloatingPointError(f"Non-finite loss at epoch {epoch+1}")
      
#       epoch_loss_sum += (batch_loss * X_batch.shape[0])
#       predictions = np.argmax(probabilities,axis=1)
#       epoch_correct += np.sum(predictions == y_batch)

#       #back progate the gradients
#       gradients = backward_pass(y_batch,parameters, cache, activation)
#       ## SGD update
#       parameters = update_parameters(parameters,gradients,learning_rate)
    
#     train_loss = epoch_loss_sum / number_of_examples
#     train_accuracy = epoch_correct / number_of_examples
#     val_loss, val_accuracy = evaluate_model(X_val,y_val,parameters,activation)
#     history['train_loss'].append(train_loss)
#     history['train_accuracy'].append(train_accuracy)
#     history['val_loss'].append(val_loss)
#     history['val_accuracy'].append(val_accuracy)
#     if verbose and epoch%10==0:
#       print(
#           f"Epoch {epoch+1}: "
#           f"Train loss={train_loss:.4f}, "
#           f"Train accuracy={train_accuracy:.4f}, "
#           f"Val loss={val_loss:.4f}, "
#           f"Val accuracy={val_accuracy:.4f}"
#       )
#     if epoch == epochs - 1:
#       print(
#           f"Epoch {epoch+1}: "
#           f"Train loss={train_loss:.4f}, "
#           f"Train accuracy={train_accuracy:.4f}, "
#           f"Val loss={val_loss:.4f}, "
#           f"Val accuracy={val_accuracy:.4f}"
#       )
#   return parameters, history
def train_model(
    X_train,
    y_train,
    X_val,
    y_val,
    hidden_size=128,
    activation="relu",
    initialization="small",
    learning_rate=0.1,
    batch_size=128,
    epochs=10,
    seed=SPLIT_SEED,
    verbose=True,
    save_model=False,
    output_dir="output/final_models",
    model_filename=None
):
    rng = np.random.default_rng(seed)

    parameters = initialize_parameters(
        input_size=X_train.shape[1],
        hidden_size=hidden_size,
        output_size=10,
        initialization=initialization,
        seed=seed
    )

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],

        # These fields are used only when save_model=True.
        "best_epoch": None,
        "best_val_loss": None,
        "best_val_accuracy": None,
        "checkpoint_path": None
    }

    number_of_examples = X_train.shape[0]

    # --------------------------------------------------------
    # Prepare checkpoint information
    # --------------------------------------------------------

    best_val_accuracy = -np.inf
    best_val_loss = np.inf
    best_epoch = None
    checkpoint_path = None

    if save_model:
        output_dir = Path(output_dir)

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        if model_filename is None:
            model_filename = (
                f"best_model_seed_{seed}.pt"
            )

        checkpoint_path = (
            output_dir / model_filename
        )

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    for epoch in range(epochs):
        shuffled_indices = rng.permutation(
            number_of_examples
        )

        epoch_loss_sum = 0.0
        epoch_correct = 0

        for start in range(
            0,
            number_of_examples,
            batch_size
        ):
            end = min(
                start + batch_size,
                number_of_examples
            )

            batch_indices = shuffled_indices[
                start:end
            ]

            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]

            probabilities, cache = forward_pass(
                X_batch,
                parameters,
                activation
            )

            batch_loss = cross_entropy_loss(
                cache["logits"],
                y_batch
            )

            if not np.isfinite(batch_loss):
                raise FloatingPointError(
                    f"Non-finite loss at "
                    f"epoch {epoch + 1}"
                )

            epoch_loss_sum += (
                batch_loss * X_batch.shape[0]
            )

            predictions = np.argmax(
                probabilities,
                axis=1
            )

            epoch_correct += np.sum(
                predictions == y_batch
            )

            gradients = backward_pass(
                y_batch,
                parameters,
                cache,
                activation
            )

            gradients_are_finite = all(
                np.all(np.isfinite(gradient))
                for gradient in gradients.values()
            )

            if not gradients_are_finite:
                raise FloatingPointError(
                    f"Non-finite gradient at "
                    f"epoch {epoch + 1}"
                )

            parameters = update_parameters(
                parameters,
                gradients,
                learning_rate
            )

        train_loss = (
            epoch_loss_sum / number_of_examples
        )

        train_accuracy = (
            epoch_correct / number_of_examples
        )

        val_loss, val_accuracy = evaluate_model(
            X_val,
            y_val,
            parameters,
            activation
        )

        history["train_loss"].append(
            train_loss
        )

        history["train_accuracy"].append(
            train_accuracy
        )

        history["val_loss"].append(
            val_loss
        )

        history["val_accuracy"].append(
            val_accuracy
        )

        # ----------------------------------------------------
        # Save the best validation checkpoint
        # ----------------------------------------------------

        if save_model:
            accuracy_improved = (
                val_accuracy > best_val_accuracy
            )

            accuracy_tied = np.isclose(
                val_accuracy,
                best_val_accuracy,
                rtol=0.0,
                atol=1e-12
            )

            loss_improved_during_tie = (
                accuracy_tied
                and val_loss < best_val_loss
            )

            if (
                accuracy_improved
                or loss_improved_during_tie
            ):
                best_val_accuracy = val_accuracy
                best_val_loss = val_loss
                best_epoch = epoch + 1

                # The network is trained with NumPy.
                # Conversion to tensors is only for saving.
                saved_parameters = {
                    parameter_name: torch.from_numpy(
                        parameter_value.copy()
                    )
                    for (
                        parameter_name,
                        parameter_value
                    ) in parameters.items()
                }

                checkpoint = {
                    "parameters": saved_parameters,
                    "seed": int(seed),
                    "best_epoch": int(best_epoch),
                    "best_val_accuracy": float(
                        best_val_accuracy
                    ),
                    "best_val_loss": float(
                        best_val_loss
                    ),
                    "hidden_size": int(hidden_size),
                    "activation": activation,
                    "initialization": initialization,
                    "learning_rate": float(
                        learning_rate
                    ),
                    "batch_size": int(batch_size),
                    "maximum_epochs": int(epochs)
                }

                torch.save(
                    checkpoint,
                    checkpoint_path
                )

        # ----------------------------------------------------
        # Print progress
        # ----------------------------------------------------

        if (
            verbose
            and (
                epoch == 0
                or (epoch + 1) % 10 == 0
                or epoch == epochs - 1
            )
        ):
            message = (
                f"Epoch {epoch + 1}: "
                f"Train loss={train_loss:.4f}, "
                f"Train accuracy={train_accuracy:.4f}, "
                f"Val loss={val_loss:.4f}, "
                f"Val accuracy={val_accuracy:.4f}"
            )

            if save_model:
                message += (
                    f", Best epoch={best_epoch}"
                )

            print(message)

    # --------------------------------------------------------
    # If saving was enabled, reload and return the best model
    # --------------------------------------------------------

    if save_model:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True
        )

        parameters = {
            parameter_name: (
                parameter_tensor.numpy().copy()
            )
            for (
                parameter_name,
                parameter_tensor
            ) in checkpoint[
                "parameters"
            ].items()
        }

        history["best_epoch"] = checkpoint[
            "best_epoch"
        ]

        history["best_val_accuracy"] = checkpoint[
            "best_val_accuracy"
        ]

        history["best_val_loss"] = checkpoint[
            "best_val_loss"
        ]

        history["checkpoint_path"] = str(
            checkpoint_path
        )

        print(
            f"Best model saved: {checkpoint_path}"
        )

        print(
            f"Selected epoch: "
            f"{history['best_epoch']}"
        )

        print(
            f"Best validation accuracy: "
            f"{100 * history['best_val_accuracy']:.2f}%"
        )

        print(
            f"Validation loss at selected epoch: "
            f"{history['best_val_loss']:.4f}"
        )

    return parameters, history

  
def run_three_seed_configuration(
    configuration_name,
    X_train,
    y_train,
    X_val,
    y_val,
    hidden_size=128,
    activation="relu",
    initialization="small",
    learning_rate=0.1,
    batch_size=128,
    epochs=10,
    seeds=None,
    verbose=False
):
    models = {}
    histories = []

    print("=" * 70)
    print("Configuration:", configuration_name)
    print("Seeds:", seeds)
    print("=" * 70)

    for run_number, seed in enumerate(
        seeds,
        start=1
    ):
        print(
            f"\nRun {run_number}/{len(seeds)} "
            f"using seed {seed}"
        )

        parameters, history = train_model(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            hidden_size=hidden_size,
            activation=activation,
            initialization=initialization,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            seed=seed,
            verbose=verbose
        )

        models[seed] = parameters
        histories.append(history)

        print(
            f"Final train accuracy: "
            f"{history['train_accuracy'][-1] * 100:.2f}%"
        )

        print(
            f"Final validation accuracy: "
            f"{history['val_accuracy'][-1] * 100:.2f}%"
        )

    return models, histories


import pandas as pd
def summarize_three_seed_results(
    histories,
    seeds
):
    individual_rows = []

    for seed, history in zip(seeds, histories):
        individual_rows.append({
            "Seed": seed,
            "Train Accuracy": (
                f"{history['train_accuracy'][-1] * 100:.2f}%"
            ),
            "Validation Accuracy": (
                f"{history['val_accuracy'][-1] * 100:.2f}%"
            ),
            "Train Loss": (
                f"{history['train_loss'][-1]:.4f}"
            ),
            "Validation Loss": (
                f"{history['val_loss'][-1]:.4f}"
            )
        })

    individual_table = pd.DataFrame(
        individual_rows
    )

    print("\nFinal-epoch results for each seed")
    display(individual_table)

    summary_rows = []

    metric_information = [
        ("Train Accuracy", "train_accuracy", True),
        ("Validation Accuracy", "val_accuracy", True),
        ("Train Loss", "train_loss", False),
        ("Validation Loss", "val_loss", False)
    ]

    for display_name, history_key, percentage in metric_information:
        final_values = np.array([
            history[history_key][-1]
            for history in histories
        ])

        mean_value = np.mean(final_values)

        # Sample standard deviation across three seeds
        standard_deviation = np.std(
            final_values,
            ddof=1
        )

        if percentage:
            mean_text = f"{mean_value * 100:.2f}%"
            std_text = (
                f"{standard_deviation * 100:.2f}%"
            )
            combined_text = (
                f"{mean_value * 100:.2f}% ± "
                f"{standard_deviation * 100:.2f}%"
            )
        else:
            mean_text = f"{mean_value:.4f}"
            std_text = f"{standard_deviation:.4f}"
            combined_text = (
                f"{mean_value:.4f} ± "
                f"{standard_deviation:.4f}"
            )

        summary_rows.append({
            "Metric": display_name,
            "Mean": mean_text,
            "Standard Deviation": std_text,
            "Mean ± SD": combined_text
        })

    summary_table = pd.DataFrame(
        summary_rows
    )

    print("\nFinal-epoch mean ± standard deviation")
    display(summary_table)

    return individual_table, summary_table

def generate_experiment_seeds(
    base_seed=SPLIT_SEED,
    number_of_seeds=3
):
    seed_generator = np.random.default_rng(
        base_seed
    )

    seeds = seed_generator.choice(
        1_000_000,
        size=number_of_seeds,
        replace=False
    )

    return [int(seed) for seed in seeds]

def plot_and_save_three_seed_results(
    histories,
    configuration_name,
    save_path
):
    train_accuracy = np.array([
        history["train_accuracy"]
        for history in histories
    ])

    val_accuracy = np.array([
        history["val_accuracy"]
        for history in histories
    ])

    train_loss = np.array([
        history["train_loss"]
        for history in histories
    ])

    val_loss = np.array([
        history["val_loss"]
        for history in histories
    ])

    epochs = np.arange(
        1,
        train_accuracy.shape[1] + 1
    )

    # Means across seeds at every epoch
    mean_train_accuracy = np.mean(
        train_accuracy,
        axis=0
    )

    mean_val_accuracy = np.mean(
        val_accuracy,
        axis=0
    )

    mean_train_loss = np.mean(
        train_loss,
        axis=0
    )

    mean_val_loss = np.mean(
        val_loss,
        axis=0
    )

    # Sample standard deviations across seeds
    std_train_accuracy = np.std(
        train_accuracy,
        axis=0,
        ddof=1
    )

    std_val_accuracy = np.std(
        val_accuracy,
        axis=0,
        ddof=1
    )

    std_train_loss = np.std(
        train_loss,
        axis=0,
        ddof=1
    )

    std_val_loss = np.std(
        val_loss,
        axis=0,
        ddof=1
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5)
    )

    # --------------------------------------------------------
    # Accuracy subplot
    # --------------------------------------------------------

    axes[0].plot(
        epochs,
        mean_train_accuracy,
        color="blue",
        label="Training mean"
    )

    axes[0].fill_between(
        epochs,
        np.clip(
            mean_train_accuracy
            - std_train_accuracy,
            0.0,
            1.0
        ),
        np.clip(
            mean_train_accuracy
            + std_train_accuracy,
            0.0,
            1.0
        ),
        color="blue",
        alpha=0.2,
        label="Training ±1 SD"
    )

    axes[0].plot(
        epochs,
        mean_val_accuracy,
        color="orange",
        label="Validation mean"
    )

    axes[0].fill_between(
        epochs,
        np.clip(
            mean_val_accuracy
            - std_val_accuracy,
            0.0,
            1.0
        ),
        np.clip(
            mean_val_accuracy
            + std_val_accuracy,
            0.0,
            1.0
        ),
        color="orange",
        alpha=0.2,
        label="Validation ±1 SD"
    )

    axes[0].set_title(
        "Accuracy Across Three Seeds"
    )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")

    axes[0].yaxis.set_major_formatter(
        PercentFormatter(1.0)
    )

    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # --------------------------------------------------------
    # Loss subplot
    # --------------------------------------------------------

    axes[1].plot(
        epochs,
        mean_train_loss,
        color="blue",
        label="Training mean"
    )

    axes[1].fill_between(
        epochs,
        np.maximum(
            mean_train_loss - std_train_loss,
            0.0
        ),
        mean_train_loss + std_train_loss,
        color="blue",
        alpha=0.2,
        label="Training ±1 SD"
    )

    axes[1].plot(
        epochs,
        mean_val_loss,
        color="orange",
        label="Validation mean"
    )

    axes[1].fill_between(
        epochs,
        np.maximum(
            mean_val_loss - std_val_loss,
            0.0
        ),
        mean_val_loss + std_val_loss,
        color="orange",
        alpha=0.2,
        label="Validation ±1 SD"
    )

    axes[1].set_title(
        "Loss Across Three Seeds"
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Cross-Entropy Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(
        configuration_name,
        fontsize=14
    )

    plt.tight_layout()

    # Save before displaying
    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    print("Saved:", save_path)

    plt.show()
    plt.close(fig)
