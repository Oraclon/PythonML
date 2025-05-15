# Initial parameters
w = 0.1  # weight
b = 0.1  # bias

# Adam hyperparameters
alpha = 0.1       # learning rate
beta1 = 0.9       # for first moment (mean)
beta2 = 0.999     # for second moment (variance)
epsilon = 1e-8

# Initialize first (m) and second (v) moments
m_w = 0
v_w = 0
m_b = 0
v_b = 0

# Input batches (2 rows per batch)
X1 = [[1], [2]]  # batch 1
y1 = [5, 8]

X2 = [[3], [4]]  # batch 2
y2 = [11, 14]

batches = [(X1, y1), (X2, y2)]
epochs = 2
t = 0  # timestep for bias correction

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}")
    
    for batch_num, (X, y) in enumerate(batches, 1):
        dw = 0
        db = 0

        # Compute gradients over the batch
        for i in range(2):  # each batch has 2 samples
            x_i = X[i][0]
            y_i = y[i]
            pred = w * x_i + b
            error = pred - y_i
            dw += 2 * x_i * error
            db += 2 * error

        # Average gradients
        dw /= 2
        db /= 2

        # Increment timestep
        t += 1

        # Update biased first moment estimates
        m_w = beta1 * m_w + (1 - beta1) * dw
        m_b = beta1 * m_b + (1 - beta1) * db

        # Update biased second raw moment estimates
        v_w = beta2 * v_w + (1 - beta2) * (dw ** 2)
        v_b = beta2 * v_b + (1 - beta2) * (db ** 2)

        # Compute bias-corrected moments
        m_w_hat = m_w / (1 - beta1 ** t)
        v_w_hat = v_w / (1 - beta2 ** t)
        m_b_hat = m_b / (1 - beta1 ** t)
        v_b_hat = v_b / (1 - beta2 ** t)

        # Update parameters
        w -= alpha * m_w_hat / ((v_w_hat ** 0.5) + epsilon)
        b -= alpha * m_b_hat / ((v_b_hat ** 0.5) + epsilon)

        print(f"  Batch {batch_num}: w = {w:.4f}, b = {b:.4f}")

# Final parameters
print(f"\nFinal weight (w): {w:.4f}")
print(f"Final bias (b): {b:.4f}")
