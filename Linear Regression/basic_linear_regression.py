import matplotlib.pyplot as plt
from func import predict, compute_gradient, compute_loss, update_weights

# =====================================================================
# DATA & HYPERPARAMETERS
# =====================================================================
x = [3, 5, 7, 9, 10, 12, 14, 16, 18, 20]
y = [52, 58, 67, 60, 78, 75, 95, 90, 112, 118]

w = 10.0
b = 5.0
epochs = 5000
lr = 0.0002 

w_init, b_init = w, b
y_hat_initial = [predict(xi, w_init, b_init) for xi in x]

# =====================================================================
# TRAINING LOOP
# =====================================================================
losses = []
for epoch in range(epochs):
    total_loss = 0
    for xi, yi in zip(x, y):
        y_hat = predict(xi, w, b)
        loss = compute_loss(y_hat, yi)
        dw, db = compute_gradient(xi, y_hat, yi)
        w, b = update_weights(w, b, dw, db, lr)
        total_loss += loss
        
    epoch_loss = total_loss / len(x)
    losses.append(epoch_loss)
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch}/{epochs} - Average Loss: {epoch_loss:.4f}")

print(f"Training completed. Final parameters: w = {w:.4f}, b = {b:.4f}")

# =====================================================================
# VISUALIZATION
# =====================================================================
y_hat_final = [predict(xi, w, b) for xi in x]

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Biểu đồ 1: Dữ liệu thực tế và các đường hồi quy trước/sau khi train
ax[0].scatter(x, y, color='blue', marker='o', label='Data Points')
ax[0].plot(x, y_hat_initial, color='gray', linestyle='--', label=f'Initial Line (w={w_init:.1f}, b={b_init:.1f})')
ax[0].plot(x, y_hat_final, color='red', linestyle='-', label=f'Fitted Line (w={w:.2f}, b={b:.2f})')
ax[0].set_xlabel('X')
ax[0].set_ylabel('Y')
ax[0].set_title('Linear Regression Fit')
ax[0].grid(True)
ax[0].legend()

# Biểu đồ 2: Hàm mất mát (Loss) giảm dần qua các Epoch
ax[1].plot(losses, color='orange')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Loss')
ax[1].set_title('Loss vs Epochs')
ax[1].grid(True)

plt.tight_layout()
plt.show()