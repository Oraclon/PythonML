import math

# L1 Loss and its derivative
def l1_loss(y_true, y_pred):
    """
    L1 Loss and its derivative.
    L1 Loss = |y_true - y_pred|
    Derivative of L1 Loss = sign(y_pred - y_true)
    """
    loss = math.fabs(y_true - y_pred)
    grad = 1 if y_pred > y_true else (-1 if y_pred < y_true else 0)  # sign function
    return loss, grad

# L2 Loss and its derivative
def l2_loss(y_true, y_pred):
    """
    L2 Loss and its derivative.
    L2 Loss = (y_true - y_pred)^2
    Derivative of L2 Loss = 2 * (y_pred - y_true)
    """
    loss = (y_true - y_pred) ** 2
    grad = 2 * (y_pred - y_true)
    return loss, grad

# Smooth L1 Loss and its derivative
def smooth_l1_loss(y_true, y_pred, beta=1.0):
    """
    Smooth L1 Loss and its derivative.
    Smooth L1 Loss is a combination of L1 and L2 Loss
    with a smooth transition between them at |y_true - y_pred| = beta.
    """
    diff = y_pred - y_true
    abs_diff = math.fabs(diff)
    
    # Calculate loss
    if abs_diff < beta:
        loss = 0.5 * (diff ** 2) / beta
    else:
        loss = abs_diff - 0.5 * beta
    
    # Calculate gradient
    if abs_diff < beta:
        grad = diff / beta
    else:
        grad = 1 if diff > 0 else -1  # sign function
    
    return loss, grad

# Example usage
if __name__ == "__main__":
    y_true = 2.0  # Ground truth
    y_pred = 3.5  # Predicted value

    # L1 Loss
    l1_loss_value, l1_grad = l1_loss(y_true, y_pred)
    print(f"L1 Loss: {l1_loss_value}, L1 Gradient: {l1_grad}")

    # L2 Loss
    l2_loss_value, l2_grad = l2_loss(y_true, y_pred)
    print(f"L2 Loss: {l2_loss_value}, L2 Gradient: {l2_grad}")

    # Smooth L1 Loss
    smooth_l1_loss_value, smooth_l1_grad = smooth_l1_loss(y_true, y_pred)
    print(f"Smooth L1 Loss: {smooth_l1_loss_value}, Smooth L1 Gradient: {smooth_l1_grad}")
