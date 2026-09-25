"""
Polynomial Regression from scratch (NumPy only)
------------------------------------------------
1. Tạo dữ liệu giả: y = sin(3x) + noise
2. Biến x -> design matrix [1, x, x^2, ..., x^d]
3. Giải bằng 2 cách: Normal Equation (closed form) và Gradient Descent
4. Chọn degree bằng train/test split, thêm Ridge (L2) regularization
5. So sánh kết quả với scikit-learn
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)


# ---------- 1. Data ----------
def make_data(n=40, noise=0.25):
    x = rng.uniform(-1, 1, n)
    y = np.sin(3 * x) + rng.normal(0, noise, n)
    return x, y


def train_test_split(x, y, test_ratio=0.5):
    idx = rng.permutation(len(x))
    n_test = int(len(x) * test_ratio)
    test, train = idx[:n_test], idx[n_test:]
    return x[train], x[test], y[train], y[test]


# ---------- 2. Feature transform ----------
def poly_features(x, degree):
    """x shape (n,) -> X shape (n, degree+1) với các cột x^0, x^1, ..., x^d"""
    return np.column_stack([x ** p for p in range(degree + 1)])


# ---------- 3a. Closed form ----------
def fit_closed_form(X, y, lam=0.0):
    if lam == 0:
        # lstsq ổn định hơn inv(X.T @ X) khi degree cao
        return np.linalg.lstsq(X, y, rcond=None)[0]
    I = np.eye(X.shape[1])
    I[0, 0] = 0  # không regularize bias w0
    return np.linalg.solve(X.T @ X + lam * I, X.T @ y)


# ---------- 3b. Gradient descent ----------
def fit_gradient_descent(X, y, lr=0.05, epochs=20000, lam=0.0):
    n, d = X.shape
    w = np.zeros(d)
    history = []
    for _ in range(epochs):
        error = X @ w - y
        grad = (2 / n) * (X.T @ error)
        grad[1:] += (2 / n) * lam * w[1:]          # gradient của Ridge penalty
        w -= lr * grad
        history.append(np.mean(error ** 2))
    return w, history


# ---------- Metrics ----------
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot


# ---------- Model class (giống API của sklearn) ----------
class PolynomialRegression:
    def __init__(self, degree=2, lam=0.0, method="closed", lr=0.05, epochs=20000):
        self.degree, self.lam, self.method = degree, lam, method
        self.lr, self.epochs = lr, epochs

    def _transform(self, x):
        z = (x - self.x_mean) / self.x_std   # standardize TRƯỚC khi lấy lũy thừa
        return poly_features(z, self.degree)

    def fit(self, x, y):
        self.x_mean, self.x_std = x.mean(), x.std()
        X = self._transform(x)
        if self.method == "closed":
            self.w = fit_closed_form(X, y, self.lam)
            self.loss_history = None
        else:
            self.w, self.loss_history = fit_gradient_descent(
                X, y, self.lr, self.epochs, self.lam)
        return self

    def predict(self, x):
        return self._transform(x) @ self.w


# ---------- Main ----------
if __name__ == "__main__":
    x, y = make_data()
    x_tr, x_te, y_tr, y_te = train_test_split(x, y)

    # (1) Chọn degree
    degrees = range(1, 16)
    train_err, test_err = [], []
    print(f"{'degree':>6} | {'train MSE':>10} | {'test MSE':>10}")
    for d in degrees:
        m = PolynomialRegression(degree=d).fit(x_tr, y_tr)
        train_err.append(mse(y_tr, m.predict(x_tr)))
        test_err.append(mse(y_te, m.predict(x_te)))
        print(f"{d:>6} | {train_err[-1]:>10.4f} | {test_err[-1]:>10.4f}")
    best_d = list(degrees)[int(np.argmin(test_err))]
    print(f"\nBest degree theo test MSE: {best_d}")
    # Lưu ý: trong project thật, chọn degree bằng validation set / cross-validation,
    # để test set dành riêng cho đánh giá cuối cùng.

    # (2) Closed form vs Gradient Descent (degree 3)
    m_cf = PolynomialRegression(degree=3, method="closed").fit(x_tr, y_tr)
    m_gd = PolynomialRegression(degree=3, method="gd").fit(x_tr, y_tr)
    print("\nWeights closed form :", np.round(m_cf.w, 4))
    print("Weights grad descent:", np.round(m_gd.w, 4))
    print(f"Test R^2 (closed form): {r2_score(y_te, m_cf.predict(x_te)):.4f}")

    # (3) Ridge cứu model degree 15
    m15 = PolynomialRegression(degree=15).fit(x_tr, y_tr)
    m15r = PolynomialRegression(degree=15, lam=0.1).fit(x_tr, y_tr)
    print(f"\nDegree 15, no ridge : test MSE = {mse(y_te, m15.predict(x_te)):.4f}")
    print(f"Degree 15, lam = 0.1: test MSE = {mse(y_te, m15r.predict(x_te)):.4f}")

    # (4) Kiểm tra với scikit-learn
    try:
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        sk = make_pipeline(PolynomialFeatures(3), LinearRegression())
        sk.fit(x_tr.reshape(-1, 1), y_tr)
        sk_mse = mse(y_te, sk.predict(x_te.reshape(-1, 1)))
        print(f"\nsklearn test MSE (degree 3): {sk_mse:.4f}  "
              f"| của mình: {mse(y_te, m_cf.predict(x_te)):.4f}")
    except ImportError:
        print("\n(sklearn chưa cài, bỏ qua bước so sánh)")

    # (5) Vẽ hình
    grid = np.linspace(-1, 1, 300)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))

    ax[0].scatter(x_tr, y_tr, s=18, label="train")
    ax[0].scatter(x_te, y_te, s=18, label="test")
    for d in (1, best_d, 15):
        m = PolynomialRegression(degree=d).fit(x_tr, y_tr)
        ax[0].plot(grid, m.predict(grid), label=f"degree {d}")
    ax[0].set_ylim(-2.5, 2.5)
    ax[0].set_title("Underfit vs good fit vs overfit")
    ax[0].legend()

    ax[1].plot(degrees, train_err, "o-", label="train MSE")
    ax[1].plot(degrees, test_err, "o-", label="test MSE")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("degree")
    ax[1].set_title("Bias-variance tradeoff")
    ax[1].legend()

    ax[2].plot(m_gd.loss_history)
    ax[2].set_yscale("log")
    ax[2].set_xlabel("epoch")
    ax[2].set_title("Gradient descent loss (degree 3)")

    plt.tight_layout()
    plt.show()
