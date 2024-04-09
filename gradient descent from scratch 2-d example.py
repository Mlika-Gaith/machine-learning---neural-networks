import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# Show Plots in matplotlib window
matplotlib.use('TkAgg')  # Use the TkAgg backend
# define plot styles
sns.set_style("darkgrid")
# Set the figure size
sns.set(rc={'figure.figsize': (10,8)})
# Color Palette
custom_palette = ["#a1ff0a", "#ff0000", "#deff0a", "#0aefff", "#be0aff"]
sns.set_palette(custom_palette)
# Set the plots context
sns.set_context("talk", font_scale=0.8, rc={"grid.linewidth":0.6, "lines.linewidth":0.5})

"""

# first function
def y_function(x):
    return x ** 2


# first derivative of function
def y_derivative(x):
    return 2 * x


x = np.arange(-100,100,0.01)
y = y_function(x)

"""


# Second function
def y_function(x):
    return np.sin(x)


def y_derivative(x):
    return np.cos(x)


x = np.arange(-5, 5, 0.1)
y = y_function(x)


# Starting position
current_position = (1.5, y_function(1.5))
plt.xlabel('Value of weight', fontsize=14, color='black')
plt.ylabel('Value of Loss', fontsize=14, color='black')
plt.title('Initial Starting Position Picked at Random', fontsize=14, color='black')
plt.xticks(fontsize=12, color='black')
plt.yticks(fontsize=12, color='black')
# Customize the grid color
plt.grid(color='darkgrey', linestyle='--', linewidth=0.5)
sns.lineplot(x=x, y=y, color="#2329a6")
# Assuming current_position is a list or array with two elements
sns.scatterplot(x=[current_position[0]], y=[current_position[1]], color="#e64552")
plt.show()


learning_rate = 0.1

for i in range(100):
    plt.xlabel('Value of weight', fontsize=14, color='black')
    plt.ylabel('Value of Loss', fontsize=14, color='black')
    plt.title('Gradient Descent Algorithm', fontsize=14, color='black')
    plt.xticks(fontsize=12, color='black')
    plt.yticks(fontsize=12, color='black')
    # Customize the grid color
    plt.grid(color='darkgrey', linestyle='--', linewidth=0.5)
    new_x = current_position[0] - learning_rate * y_derivative(current_position[0])
    new_y = y_function(new_x)
    current_position = (new_x, new_y)
    sns.lineplot(x=x, y=y, color="#003049")
    # Assuming current_position is a list or array with two elements
    sns.scatterplot(x=[current_position[0]], y=[current_position[1]], color="#e64552")
    plt.pause(0.01)
    plt.clf()



