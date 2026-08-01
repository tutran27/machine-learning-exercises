def predict(x, w, b):
    return x * w + b

def compute_loss(y_hat, y):
    return (y_hat - y) ** 2

def compute_gradient(x, y_hat, y):
    dw = 2 * x * (y_hat - y)
    db = 2 * (y_hat - y)
    return dw, db

def update_weights(w, b, dw, db, lr):
    w = w - lr * dw
    b = b - lr * db
    return w, b