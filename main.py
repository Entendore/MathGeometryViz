import sys
import numpy as np
from itertools import product, combinations
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget,
                             QSlider, QPushButton, QLabel, QScrollArea, QGridLayout)
from PyQt6.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

# -------------------------------
# ND Geometry Utilities
# -------------------------------

def generate_nd_cube(n_dim, points_per_axis=2):
    axes = [np.linspace(-1, 1, points_per_axis) for _ in range(n_dim)]
    return np.array(list(product(*axes)))

def generate_edges(points):
    edges = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if np.sum(np.abs(points[i]-points[j]) != 0) == 1:
                edges.append((i, j))
    return edges

def rotation_matrix_nd(n_dim, axis1, axis2, theta):
    R = np.eye(n_dim)
    c, s = np.cos(theta), np.sin(theta)
    R[axis1, axis1] = c
    R[axis2, axis2] = c
    R[axis1, axis2] = -s
    R[axis2, axis1] = s
    return R

def rotate_nd(points, n_dim, angles_dict):
    rotated = points.copy()
    for key in angles_dict:
        if isinstance(key, tuple) and key[1] == "angle":
            pair = key[0]
            angle = angles_dict[key]
            R = rotation_matrix_nd(n_dim, pair[0], pair[1], angle)
            rotated = rotated @ R.T
    return rotated

def project_nd(points, mode='2D', random_seed=42):
    np.random.seed(random_seed)
    n_dim = points.shape[1]
    if mode == '2D':
        proj = np.random.randn(n_dim, 2)
    elif mode == '3D':
        proj = np.random.randn(n_dim, 3)
    else:
        raise ValueError("Projection mode must be '2D' or '3D'")
    return points @ proj

# -------------------------------
# Matplotlib Widget
# -------------------------------

class GeometryCanvas(FigureCanvas):
    def __init__(self):
        self.fig = plt.figure()
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.axis('off')
        self.points = None
        self.edges = None
        self.angles_dict = {}
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_animation)
        self.animating = False
        self.projection_mode = '2D'
        self.scale = 1.0

    def plot_hypercube(self, points, mode='2D', angles_dict=None):
        self.projection_mode = mode
        self.points = points
        self.edges = generate_edges(points)
        self.angles_dict = angles_dict if angles_dict else {}
        self.draw_frame(points)

    def draw_frame(self, points_nd):
        self.ax.clear()
        self.ax.axis('off')
        self.ax.set_aspect('equal')

        projected = project_nd(points_nd, mode=self.projection_mode)
        projected *= self.scale

        if self.projection_mode == '2D':
            for i, j in self.edges:
                self.ax.plot([projected[i, 0], projected[j, 0]],
                             [projected[i, 1], projected[j, 1]], color='black')
            self.ax.scatter(projected[:, 0], projected[:, 1], c='black')
        elif self.projection_mode == '3D':
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.ax.axis('off')
            for i, j in self.edges:
                self.ax.plot([projected[i, 0], projected[j, 0]],
                             [projected[i, 1], projected[j, 1]],
                             [projected[i, 2], projected[j, 2]], color='black')
            self.ax.scatter(projected[:, 0], projected[:, 1], projected[:, 2], c='black')
            self.ax.set_box_aspect([1,1,1])

        self.draw()

    def start_animation(self):
        if not self.animating:
            self.animating = True
            self.timer.start(30)

    def stop_animation(self):
        if self.animating:
            self.animating = False
            self.timer.stop()

    def update_animation(self):
        # Increment each axis pair angle by its speed
        for key in self.angles_dict:
            if isinstance(key, tuple) and key[1] == "speed":
                pair = key[0]
                self.angles_dict[(pair, "angle")] += self.angles_dict[key]
        rotated = rotate_nd(self.points, self.points.shape[1], self.angles_dict)
        self.draw_frame(rotated)

# -------------------------------
# Main Qt Window
# -------------------------------

class GeometryVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ND Hypercube Rotation per Axis Pair")

        self.main_widget = QWidget()
        self.layout = QVBoxLayout(self.main_widget)

        self.canvas = GeometryCanvas()
        self.layout.addWidget(self.canvas)

        # Dimension slider
        self.dim_slider = QSlider(Qt.Orientation.Horizontal)
        self.dim_slider.setMinimum(2)
        self.dim_slider.setMaximum(6)
        self.dim_slider.setValue(4)
        self.dim_slider.valueChanged.connect(self.update_axis_sliders)
        self.layout.addWidget(QLabel("Dimension"))
        self.layout.addWidget(self.dim_slider)

        # Projection toggle button
        self.proj_button = QPushButton("Toggle Projection 2D/3D")
        self.proj_button.clicked.connect(self.toggle_projection)
        self.layout.addWidget(self.proj_button)

        # Scroll area for axis-pair sliders
        self.scroll_area = QScrollArea()
        self.scroll_widget = QWidget()
        self.scroll_layout = QGridLayout(self.scroll_widget)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.scroll_widget)
        self.layout.addWidget(QLabel("Per-Axis-Pair Rotation Speeds"))
        self.layout.addWidget(self.scroll_area)

        # Store sliders and labels
        self.axis_sliders = {}
        self.axis_labels = {}

        # Animation button
        self.anim_button = QPushButton("Start/Stop Animation")
        self.anim_button.clicked.connect(self.toggle_animation)
        self.layout.addWidget(self.anim_button)

        self.setCentralWidget(self.main_widget)

        self.projection_mode = '2D'
        self.angles_dict = {}
        self.update_axis_sliders()
        self.update_plot()

    def update_axis_sliders(self):
        # Clear previous sliders
        for label in self.axis_labels.values():
            label.deleteLater()
        for slider in self.axis_sliders.values():
            slider.deleteLater()
        self.axis_sliders.clear()
        self.axis_labels.clear()

        n_dim = self.dim_slider.value()
        axes_pairs = list(combinations(range(n_dim), 2))
        self.angles_dict = {}

        row = 0
        for pair in axes_pairs:
            label = QLabel(f"Axis {pair}")
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setMinimum(-50)
            slider.setMaximum(50)
            slider.setValue(10)
            slider.valueChanged.connect(self.update_speeds)
            self.scroll_layout.addWidget(label, row, 0)
            self.scroll_layout.addWidget(slider, row, 1)
            self.axis_labels[pair] = label
            self.axis_sliders[pair] = slider
            # Initialize angles and speeds
            self.angles_dict[(pair, "angle")] = 0.0
            self.angles_dict[(pair, "speed")] = slider.value()/100.0
            row += 1
        self.update_plot()

    def update_speeds(self):
        for pair, slider in self.axis_sliders.items():
            self.angles_dict[(pair, "speed")] = slider.value()/100.0

    def update_plot(self):
        n_dim = self.dim_slider.value()
        points = generate_nd_cube(n_dim)
        self.canvas.plot_hypercube(points, mode=self.projection_mode, angles_dict=self.angles_dict)

    def toggle_animation(self):
        if self.canvas.animating:
            self.canvas.stop_animation()
        else:
            self.canvas.start_animation()

    def toggle_projection(self):
        self.projection_mode = '3D' if self.projection_mode=='2D' else '2D'
        self.update_plot()

# -------------------------------
# Run Application
# -------------------------------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GeometryVisualizer()
    window.resize(900, 900)
    window.show()
    sys.exit(app.exec())
