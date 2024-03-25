import numpy as np
from LinearRegression import LinearRegression
from LogisticRegression import LogisticRegression
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Breast Cancer dataset
breast_cancer = load_breast_cancer()
X_linear, _, Y_linear, _ = train_test_split(breast_cancer.data[:, [2]], breast_cancer.target, test_size=0.2,
                                            random_state=42)

# Diabetes dataset
diabetes = load_diabetes()
X_logistic, _, Y_logistic, _ = train_test_split(diabetes.data, diabetes.target, test_size=0.2, random_state=42)

X_linear = MinMaxScaler().fit_transform(X_linear)
X_logistic = StandardScaler().fit_transform(X_logistic)

print("Breast Cancer Dataset:")
print("X_linear shape:", X_linear.shape)
print("Y_linear shape:", Y_linear.shape)

print("\nDiabetes Dataset:")
print("X_logistic shape:", X_logistic.shape)
print("Y_logistic shape:", Y_logistic.shape)

linear_model = LinearRegression()

learning_rate = 0.0011
epochs = 1000
for epoch in range(epochs):
    predictions = linear_model.forward(X_linear)
    loss = np.mean((predictions - Y_linear) ** 2)
    dL_dw, dL_db, _ = linear_model.backward(2 * (predictions - Y_linear), X_linear)
    linear_model.update_parameters(dL_dw, dL_db, learning_rate)
    if epoch % 100 == 0:
        print(f"Linear Regression - Epoch {epoch}: Loss {loss}")

        logistic_model = LogisticRegression(input_size=X_logistic.shape[1])

        learning_rate = 0.0001
        epochs = 1000

        for epoch in range(epochs):

            predictions = logistic_model.forward(X_logistic)
            epsilon = 1e-10
            predictions = np.clip(predictions, epsilon, 1 - epsilon)
            loss = -np.mean(Y_logistic * np.log(predictions) + (1 - Y_logistic) * np.log(1 - predictions))

            dL_dw, dL_db = logistic_model.backward(predictions, Y_logistic, X_logistic)
            logistic_model.update_parameters(dL_dw, dL_db, learning_rate)

            if epoch % 100 == 0:
                print(f"Logistic Regression - Epoch {epoch}: Loss {loss}")

logistic_model = LogisticRegression(input_size=X_logistic.shape[1])

learning_rate = 0.0001
epochs = 1000

for epoch in range(epochs):

    predictions = logistic_model.forward(X_logistic)
    epsilon = 1e-10
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    loss = -np.mean(Y_logistic * np.log(predictions) + (1 - Y_logistic) * np.log(1 - predictions))

    dL_dw, dL_db = logistic_model.backward(predictions, Y_logistic, X_logistic)
    logistic_model.update_parameters(dL_dw, dL_db, learning_rate)

    if epoch % 100 == 0:
        print(f"Logistic Regression - Epoch {epoch}: Loss {loss}")
