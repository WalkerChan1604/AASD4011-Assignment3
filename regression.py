import torch
from torch import nn


def create_linear_regression_model(input_size, output_size):
    """
    Create a linear regression model with the given input and output sizes.
    Hint: use nn.Linear
    """
    model = nn.Linear(input_size, output_size)
    return model


def train_iteration(X, y, model, loss_fn, optimizer):
    # Compute prediction and loss
    pred = model(X)
    loss = loss_fn(pred, y)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def fit_regression_model(X, y):
    """
    Train the model for the given number of epochs.
    Hint: use the train_iteration function.
    Hint 2: while woring you can use the print function to print the loss every 1000 epochs.
    Hint 3: you can use the previos_loss variable to stop the training when the loss is not changing much.
    """
    # Check for NaN or infinite values in input data
    if torch.isnan(X).any() or torch.isinf(X).any():
        raise ValueError("Input data contains NaN or infinite values.")
    # Check for NaN or infinite values in target data
    if torch.isnan(y).any() or torch.isinf(y).any():
        raise ValueError("Target data contains NaN or infinite values.")

    learning_rate = 0.001  # Reduced learning rate
    num_epochs = 5000  # Increased number of epochs
    input_features = X.shape[1]
    output_features = y.shape[1]
    model = create_linear_regression_model(input_features, output_features)
    
    loss_fn = nn.MSELoss()  # Using Mean Squared Error Loss

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    previous_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        loss = train_iteration(X, y, model, loss_fn, optimizer)
        if epoch % 1000 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}")
        if abs(previous_loss - loss.item()) < 1e-5:
            print("Loss convergence reached. Stopping training.")
            break
        previous_loss = loss.item()

    return model, loss