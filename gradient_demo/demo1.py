import numpy as np
import matplotlib.pyplot as plt

plot_x = np.linspace(-1, 6, 141)
print(plot_x)

plot_y = (plot_x - 2.5)**2 -1
plt.plot(plot_x, plot_y)
plt.show()


def dJ(theta):
    return 2*(theta-2.5)


def J(theta):
    return (theta-2.5)**2 -1


eta = 0.1
epsilon = 1e-8

theta = 0.0
while True:
    gradient = dJ(theta)
    last_theta = theta
    theta = theta - eta * gradient

    if(abs(J(theta) - J(last_theta)) < epsilon):
        break

print(theta)
print(J(theta))


theta = 0.0
theta_history = [theta]
while True:
    gradient = dJ(theta)
    last_theta = theta
    theta = theta - eta * gradient
    theta_history.append(theta)

    if(abs(J(theta) - J(last_theta)) < epsilon):
        break

plt.plot(plot_x, J(plot_x))
plt.plot(np.array(theta_history), J(np.array(theta_history)), color='r', marker='+')
plt.show()

print(len(theta_history))


def gradient_descent(initial_theta, eta, n_iters=1e4, epsilon=1e-8):
    theta = initial_theta
    theta_history.append(initial_theta)
    i_iter = 0
    while i_iter < n_iters:
        gradient = dJ(theta)
        last_theta = theta
        theta = theta - eta * gradient
        theta_history.append(theta)

        if abs(J(theta) - J(last_theta)) < epsilon:
            break
        i_iter += 1


def plot_theta_history():
    plt.plot(plot_x, J(plot_x))
    plt.plot(np.array(theta_history), J(np.array(theta_history)), color='r', marker='+')
    plt.show()


eta = 0.01
theta_history = []
gradient_descent(0., eta, n_iters=10)
plot_theta_history()

print(len(theta_history))
