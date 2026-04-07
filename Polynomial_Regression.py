import numpy as np
import matplotlib.pyplot as plt

class PolynomialRegression:
    def __init__(self, degree=2, learning_rate=0.01, epochs=1000):
        self.degree = degree
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        
    def _create_polynomial_features(self, X):
        X_poly = np.ones((len(X), 1))
        for d in range(1, self.degree + 1):
            X_poly = np.hstack((X_poly, X**d))
        return X_poly
    
    def fit(self, X, Y):
        X = np.array(X).reshape(-1, 1)
        Y = np.array(Y)
        
        X_poly = self._create_polynomial_features(X)
        n_samples, n_features = X_poly.shape
        
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for epoch in range(self.epochs):
            y_pred = np.dot(X_poly, self.weights) + self.bias
            
            dw = (-2/n_samples) * np.dot(X_poly.T, (Y - y_pred))
            db = (-2/n_samples) * np.sum(Y - y_pred)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            if (epoch + 1) % 100 == 0:
                loss = np.mean((Y - y_pred)**2)
                print(f"Epoch {epoch+1}, Loss: {loss:.6f}")
    
    def predict(self, X):
        X = np.array(X).reshape(-1, 1)
        X_poly = self._create_polynomial_features(X)
        return np.dot(X_poly, self.weights) + self.bias
    
    def r2_score(self, Y_true, Y_pred):
        ss_res = np.sum((Y_true - Y_pred)**2)
        ss_tot = np.sum((Y_true - np.mean(Y_true))**2)
        return 1 - (ss_res / ss_tot)


if __name__ == "__main__":
    X = np.array([1, 2, 3, 4, 5])
    Y = np.array([2, 6, 12, 24, 40])
    
    model = PolynomialRegression(degree=3, learning_rate=0.01, epochs=1000)
    model.fit(X, Y)
    
    Y_pred = model.predict(X)
    print(f"\nR² Score: {model.r2_score(Y, Y_pred):.4f}")
    print(f"Weights: {model.weights}")
    print(f"Bias: {model.bias}")
    
    plt.scatter(X, Y, color='blue', label='Actual')
    plt.plot(X, Y_pred, color='red', label='Predicted')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title('Polynomial Regression from Scratch')
    plt.show()
