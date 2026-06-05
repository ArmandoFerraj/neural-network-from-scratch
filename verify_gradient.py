def gradient_check(nn, x, y, num_samples=10, epsilon=1e-5):
    # analytical gradients from your backprop
    a2, a1, z1 = nn.forward(x)
    dW2, db2, dW1, db1 = nn.backprop(x, y, a2, a1, z1)

    # pair each parameter with its analytical gradient
    params_and_grads = [
        ("W1", nn.w1, dW1),
        ("b1", nn.b1, db1),
        ("W2", nn.w2, dW2),
        ("b2", nn.b2, db2),
    ]

    for name, param, analytic_grad in params_and_grads:
        max_diff = 0.0
        flat = param.ravel()              # view the param as 1D for easy indexing
        analytic_flat = analytic_grad.ravel()

        # check num_samples random entries
        indices = np.random.choice(flat.size, size=min(num_samples, flat.size), replace=False)
        for idx in indices:
            original = flat[idx]

            flat[idx] = original + epsilon
            a2_plus, _, _ = nn.forward(x)
            loss_plus = nn.error_func(a2_plus, y)

            flat[idx] = original - epsilon
            a2_minus, _, _ = nn.forward(x)
            loss_minus = nn.error_func(a2_minus, y)

            flat[idx] = original          # restore

            numerical = (loss_plus - loss_minus) / (2 * epsilon)
            diff = abs(numerical - analytic_flat[idx])
            max_diff = max(max_diff, diff)

        print(f"{name}: max difference = {max_diff:.2e}")