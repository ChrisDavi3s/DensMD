"""
A PyQt5-based application for visualising atomic trajectory data in 3D.

This module provides a graphical interface to load molecular dynamics trajectory data,
compute density histograms and averaged positions, and interactively visualise the results
with various rendering modes and slicing options. Key features include:

- Multiple visualisation modes (histogram density, averaged positions)
- Interactive region-of-interest selection
- Miller plane slicing for advanced analysis
- Real-time parameter adjustments
- Camera controls and automated rotation

The main class DensityVisualiser handles data loading, processing, and GUI management.

# TODO SEPERATE INTO MULTIPLE FILES - This single file is disgusting (sorry)

Copyright (c) 2025 Chris Davies
"""

import numpy as np
from scipy.ndimage import gaussian_filter
import pyvista as pv
import matplotlib.pyplot as plt
from pyvistaqt import QtInteractor
from typing import List, Dict, Tuple, Optional
from typing import Any, Dict, Tuple, Optional
from vtk import VTK_CUBIC_INTERPOLATION
from PyQt5 import QtWidgets, QtCore, QtGui
import random
from ase.io import read
from tqdm import tqdm
import sys
import pickle

###############################################################
# APPLICATION CONFIGURATION
###############################################################

# Input/Output Configuration
INPUT_FILE_CONFIG = {
    'path': '/Users/chrisdavies/Desktop/long_traj_Li6PS5Br_87_13_1000K_1000k_1.000000ns.dat',
    'format': 'ase',  # 'pickle' or 'ase'
    'slice': ":",  # ASE index string (e.g. ':' for all frames, '::2' for every 2nd frame) or a slice object
}

# Atom type mapping for visualization (None to use original types)
# THIS IS AN EXAMPLE SETUP
ATOM_TYPE_MAP = {
    'H': 'Li',
    'He': 'P',
    'Li': 'S',
    'Be': 'Br'
}

# Where to get the average positions from
AVERAGE_POSITIONS_FRAME_SLICE = slice(10000, 10001) # DEFAULT: slice(None)
# THIS IS A MESSY WORKAROUND TO GET AROUND WRAPPING ISSUES

# Application Constants and Configuration
GRID_RESOLUTION = 200  # Fixed grid resolution for histogram binning
GAUSSIAN_SIGMA = 1.5   # Sigma value for Gaussian smoothing
UPDATE_DELAY_MS = 200  # Delay for visualization updates
ROTATION_FPS = 20      # Frames per second for rotation animation
ROTATION_AZIMUTH = 0.5   # Azimuth angle for each rotation frame

# Window Configuration
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 900
MAIN_PANEL_RATIO = 3   # Ratio of plotter to control panel width

# Visualisation Parameters
SPHERE_SIZE_RANGE = (1, 50)
SPHERE_SIZE_DEFAULT = 5
DENSITY_RANGE = (0, 255)
DENSITY_LOWER_DEFAULT = 77
DENSITY_UPPER_DEFAULT = 178
OPACITY_DEFAULT = 100
GAMMA_RANGE    = (0.0, 2.0)
GAMMA_DEFAULT  = 0.0

# Miller Indices Configuration
MILLER_INDEX_RANGE = (-10, 10)
MILLER_INDEX_DEFAULT = 1
MILLER_THICKNESS_RANGE = (0.1, 20.0)
MILLER_THICKNESS_DEFAULT = 2.0
MILLER_OFFSET_RANGE = (-10.0, 10.0)
MILLER_OFFSET_DEFAULT = 0.0

# Available Colormaps - these are the colormaps available in matplotlib
# See:https://matplotlib.org/stable/users/explain/colors/colormaps.html
AVAILABLE_COLORMAPS = [
    "coolwarm",
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",
    "Reds",
    "Blues",
    "Greens",
    "Purples"
]

def _calculate_roi_bounds(roi_def: Optional[Dict], cell: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Calculates the absolute min/max bounds based on ROI definition."""
    if roi_def is None:
        # Use full cell if no ROI is defined
        corners = np.array([np.dot(np.array([i, j, k]), cell)
                            for i in (0, 1) for j in (0, 1) for k in (0, 1)])
        roi_min = corners.min(axis=0)
        roi_max = corners.max(axis=0)
        print("No initial ROI defined, using full cell.")
        return roi_min, roi_max

    roi_type = roi_def.get('type', 'absolute')
    bounds = roi_def.get('bounds')

    if bounds is None:
        raise ValueError("ROI definition must include 'bounds'")

    if roi_type == 'fractional':
        fbounds = np.array(bounds)
        # Generate 8 corners of the fractional box
        f_corners = np.array([
            (fbounds[0, i], fbounds[1, j], fbounds[2, k])
            for i in (0, 1) for j in (0, 1) for k in (0, 1)
        ])
        # Convert fractional corners to absolute Cartesian coordinates
        abs_corners = np.dot(f_corners, cell)
        roi_min = abs_corners.min(axis=0)
        roi_max = abs_corners.max(axis=0)
        print(f"Calculated initial ROI bounds (fractional): {roi_min} to {roi_max}")
    elif roi_type == 'absolute':
        roi_min = np.array([b[0] for b in bounds])
        roi_max = np.array([b[1] for b in bounds])
        print(f"Using initial ROI bounds (absolute): {roi_min} to {roi_max}")
    else:
        raise ValueError(f"Unknown ROI type: {roi_type}")

    return roi_min, roi_max

###############################################################
# MAIN APPLICATION CLASS
###############################################################

class DensityVisualiser(QtWidgets.QMainWindow):
    """Main application window for atomic data visualisation.
    
    Handles GUI setup, data loading/preprocessing, and interactive visualisation controls.

    Attributes:
        atom_type_map (Dict[str, str]): Mapping of atomic symbols for visualisation
        grid_resolution (int): Resolution for 3D histogram grid
        plotter (QtInteractor): PyVista Qt interactor for 3D rendering
        atom_data (Dict[str, Any]): Precomputed atomic trajectory data
        rendered_actors (Dict[str, Any]): Currently active VTK actors
    """
    def __init__(self):
        """Initialize the application window and visualization components.
        
        Initializes the main window, sets up the visualization layout, control panels,
        and configures all interactive elements for atomic visualization. The initialization
        process follows these main steps:
        1. Basic window setup
        2. Main layout configuration
        3. Global slicing controls
        4. View and rotation controls
        5. Miller indices slicing
        6. Atom-specific visualization settings
        7. Timer and visualization initialization
        """
        super().__init__()

         # Initialize cache for region data
        self._region_data_cache = None
        self._last_roi_hash = None
        self._last_miller_hash = None

        # Initialize basic window properties and core parameters
        self._init_window_properties()
        
        # Set up main layout and visualization components
        self._init_main_layout()
        
        # Initialize control panel components
        self._init_control_panel()
        
        # Set up global slicing controls
        self._init_global_slicing()

        # Initialize Gaussian smoothing controls
        self._init_smoothing_controls()
        
        # Configure view and rotation controls
        self._init_view_controls()
        
        # Set up Miller indices slicing controls
        self._init_miller_controls()
        
        # Load data and initialize atom visualization settings
        self._init_atom_settings()
        
        # Configure update timers and initialize visualization
        self._init_timers()

        # Now finally we can update the visualisation
        self.update_visualisation()

    ###############################################################
    # INITIALISATION METHODS
    ###############################################################

    def _init_window_properties(self) -> None:
        """Initialize basic window properties and core parameters."""
        self.setWindowTitle("DensMD")
        self.atom_type_map = ATOM_TYPE_MAP
        self.grid_resolution = GRID_RESOLUTION
        self.rotation_azimuth = ROTATION_AZIMUTH
        self.destroyed.connect(self.cleanup)

    def _init_main_layout(self) -> None:
        """Set up main window layout with visualization and control panels."""
        # Main container setup
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QHBoxLayout(central_widget)
        
        # VTK visualization panel
        self.plotter = QtInteractor()
        self.plotter.camera.parallel_projection = False
        main_layout.addWidget(self.plotter, stretch=MAIN_PANEL_RATIO)
        
        # Control panel scroll area
        self.control_scroll = QtWidgets.QScrollArea()
        self.control_scroll.setWidgetResizable(True)
        main_layout.addWidget(self.control_scroll, stretch=1)
        
        # Control panel widget and layout
        self.control_widget = QtWidgets.QWidget()
        self.control_layout = QtWidgets.QVBoxLayout(self.control_widget)
        self.control_scroll.setWidget(self.control_widget)

    def _init_global_slicing(self) -> None:
        """Initialize global ROI slicing controls."""
        global_group = QtWidgets.QGroupBox("Global Slicing Settings")
        global_layout = QtWidgets.QVBoxLayout(global_group)
        
        # Create sliders for each dimension
        slider_configs = [
            ("X Min", 0, 0),
            ("X Max", 0, self.grid_resolution - 1),
            ("Y Min", 0, 0),
            ("Y Max", 0, self.grid_resolution - 1),
            ("Z Min", 0, 0),
            ("Z Max", 0, self.grid_resolution - 1)
        ]
        
        for label, min_val, init_val in slider_configs:
            slider_obj = self.create_labelled_slider(
                label, min_val, self.grid_resolution - 1, init_val, global_layout
            )
            setattr(self, f"{label.lower().replace(' ', '')}_slider_obj", slider_obj)
        
        # Connect slider signals
        self._connect_roi_sliders()
        
        self.control_layout.addWidget(global_group)

    def _init_view_controls(self) -> None:
        """Initialize camera view and rotation controls."""
        view_controls = QtWidgets.QWidget()
        view_layout = QtWidgets.QGridLayout(view_controls)
        
        # View alignment buttons
        self._setup_view_buttons(view_layout)
        
        # Rotation controls
        rotation_controls = self._setup_rotation_controls()
        
        # Add controls to global layout
        global_layout = self.control_layout.itemAt(0).widget().layout()
        global_layout.addWidget(view_controls)
        global_layout.addWidget(rotation_controls)
        global_layout.addWidget(self.reset_view_button)

    def _init_smoothing_controls(self) -> None:
        """Initialize Gaussian smoothing controls."""
        smoothing_group = QtWidgets.QGroupBox("Smoothing Settings")
        smoothing_layout = QtWidgets.QVBoxLayout(smoothing_group)
        self.sigma_slider_obj = self.create_labelled_slider(
            "Gaussian Sigma", 0, 20, GAUSSIAN_SIGMA, smoothing_layout
        )
        # Add the smoothing group to the main control layout:
        self.control_layout.addWidget(smoothing_group)

    def _init_miller_controls(self) -> None:
        """Initialize Miller indices slicing controls."""
        miller_group = QtWidgets.QGroupBox("Miller Slicing Settings")
        miller_layout = QtWidgets.QVBoxLayout(miller_group)
        
        # Miller controls container
        self.miller_controls = QtWidgets.QWidget()
        miller_controls_layout = QtWidgets.QVBoxLayout(self.miller_controls)
        
        # Miller indices spinboxes
        self._setup_miller_spinboxes(miller_controls_layout)
        
        # Miller control visibility toggle
        self._setup_miller_visibility(miller_layout)
        
        self.control_layout.addWidget(miller_group)
        self.miller_controls.setVisible(False)

        self.align_miller_button = QtWidgets.QPushButton("View Miller Plane")
        self.align_miller_button.clicked.connect(self.align_miller_view)
        miller_controls_layout.addWidget(self.align_miller_button)

    def _init_atom_settings(self) -> None:
        """Initialize atom-specific visualization settings."""
        self.load_and_precompute(INPUT_FILE_CONFIG)
        self.atom_settings_widgets = {}
        
        for atype in sorted(self.atom_data.keys()):
            settings_group = self._create_atom_settings_group(atype)
            self.control_layout.addWidget(settings_group)
        
        self.control_layout.addStretch()

    def _init_timers(self) -> None:
        """Initialize update timers and visualization state."""
        # Update timer for visualization changes
        self.update_timer = QtCore.QTimer()
        self.update_timer.setInterval(UPDATE_DELAY_MS)
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.update_visualisation)
        
        # Rotation animation timer
        self.rotation_timer = QtCore.QTimer()
        self.rotation_timer.setInterval(int(1/ROTATION_FPS*1000))
        self.rotation_timer.timeout.connect(self.rotate_view)
        
        # Initialize visualization state
        self._rendered_actors = {}

    def _init_control_panel(self) -> None:
        """Initialize the control panel base structure."""
        self.control_widget = QtWidgets.QWidget()
        self.control_layout = QtWidgets.QVBoxLayout(self.control_widget)
        self.control_scroll.setWidget(self.control_widget)

    def _connect_roi_sliders(self) -> None:
        """Connect ROI slider signals to their respective update methods."""
        self.xmin_slider_obj["slider"].valueChanged.connect(self.update_x_min)
        self.xmax_slider_obj["slider"].valueChanged.connect(self.update_x_max)
        self.ymin_slider_obj["slider"].valueChanged.connect(self.update_y_min)
        self.ymax_slider_obj["slider"].valueChanged.connect(self.update_y_max)
        self.zmin_slider_obj["slider"].valueChanged.connect(self.update_z_min)
        self.zmax_slider_obj["slider"].valueChanged.connect(self.update_z_max)

    def _setup_view_buttons(self, layout: QtWidgets.QGridLayout) -> None:
        """Set up view alignment buttons.
        
        Args:
            layout: Parent layout for the buttons
        """
        self.view_x_button = QtWidgets.QPushButton("View X")
        self.view_y_button = QtWidgets.QPushButton("View Y")
        self.view_z_button = QtWidgets.QPushButton("View Z")
        self.reset_view_button = QtWidgets.QPushButton("Reset View")
        
        layout.addWidget(self.view_x_button, 0, 0)
        layout.addWidget(self.view_y_button, 0, 1)
        layout.addWidget(self.view_z_button, 0, 2)
        
        self.view_x_button.clicked.connect(lambda: self.align_view('x'))
        self.view_y_button.clicked.connect(lambda: self.align_view('y'))
        self.view_z_button.clicked.connect(lambda: self.align_view('z'))
        self.reset_view_button.clicked.connect(self.reset_view)

    def _setup_rotation_controls(self) -> QtWidgets.QWidget:
        """Set up rotation control widgets.
        
        Returns:
            Container widget with rotation controls
        """
        rotation_controls = QtWidgets.QWidget()
        rotation_layout = QtWidgets.QGridLayout(rotation_controls)
        rotation_layout.setContentsMargins(0, 0, 0, 0)
        rotation_layout.setSpacing(4)
        
        self.rotate_button = QtWidgets.QPushButton("Rotate")
        self.rotation_speed = QtWidgets.QDoubleSpinBox()
        self.rotation_speed.setRange(0.1, 10.0)
        self.rotation_speed.setValue(0.5)
        self.rotation_speed.setSingleStep(0.1)
        self.rotation_speed.setDecimals(1)
        self.rotation_speed.setFixedWidth(70)
        
        speed_label = QtWidgets.QLabel("°/frame")
        speed_label.setFixedWidth(45)
        
        rotation_layout.addWidget(self.rotate_button, 0, 0)
        rotation_layout.addWidget(self.rotation_speed, 0, 1)
        rotation_layout.addWidget(speed_label, 0, 2)
        
        self.rotate_button.clicked.connect(self.toggle_rotation)
        self.rotation_speed.valueChanged.connect(self.update_rotation_speed)
        
        return rotation_controls

    def _setup_miller_spinboxes(self, layout: QtWidgets.QVBoxLayout) -> None:
        """Set up Miller indices control spinboxes.
        
        Args:
            layout: Parent layout for the spinboxes
        """
        self.miller_h = self.create_labelled_spinbox(
            "Miller h", 
            MILLER_INDEX_RANGE[0], 
            MILLER_INDEX_RANGE[1], 
            MILLER_INDEX_DEFAULT, 
            layout
        )
        self.miller_k = self.create_labelled_spinbox(
            "Miller k",
            MILLER_INDEX_RANGE[0],
            MILLER_INDEX_RANGE[1],
            MILLER_INDEX_DEFAULT,
            layout
        )
        self.miller_l = self.create_labelled_spinbox(
            "Miller l",
            MILLER_INDEX_RANGE[0],
            MILLER_INDEX_RANGE[1],
            MILLER_INDEX_DEFAULT,
            layout
        )
        self.miller_thickness = self.create_labelled_double_spinbox(
            "Slice Thickness (Å)",
            MILLER_THICKNESS_RANGE[0],
            MILLER_THICKNESS_RANGE[1],
            MILLER_THICKNESS_DEFAULT,
            0.1,
            layout
        )
        self.miller_offset = self.create_labelled_double_spinbox(
            "Slice Offset (Å)",
            MILLER_OFFSET_RANGE[0],
            MILLER_OFFSET_RANGE[1],
            MILLER_OFFSET_DEFAULT,
            0.1,
            layout
        )

    def _setup_miller_visibility(self, layout: QtWidgets.QVBoxLayout) -> None:
        """Set up Miller controls visibility toggle.
        
        Args:
            layout: Parent layout for the controls
        """
        self.miller_checkbox = QtWidgets.QCheckBox("Enable Miller Slicing")
        self.miller_checkbox.setChecked(False)
        self.miller_checkbox.stateChanged.connect(self.toggle_miller_controls)
        layout.addWidget(self.miller_checkbox)
        layout.addWidget(self.miller_controls)

    def _create_atom_settings_group(self, atype: str) -> QtWidgets.QGroupBox:
        """Create settings group for an atom type.
        
        Args:
            atype: Atom type identifier
            
        Returns:
            QGroupBox containing atom-specific controls
        """
        group_box = QtWidgets.QGroupBox(f"Atom: {atype}")
        group_layout = QtWidgets.QVBoxLayout(group_box)
        
        # Visualization mode selector
        mode_label = QtWidgets.QLabel("Visualization Mode:")
        group_layout.addWidget(mode_label)
        mode_combo = QtWidgets.QComboBox()
        modes = ["Hidden", "Histogram", "Averaged Positions"]
        mode_combo.addItems(modes)
        mode_combo.currentIndexChanged.connect(
            lambda _, a=atype: self.update_atom_ui_visibility(a)
        )
        mode_combo.currentIndexChanged.connect(self.schedule_update)
        group_layout.addWidget(mode_combo)
        
        # Colormap selector for histogram mode
        cmap_label = QtWidgets.QLabel("Colormap:")
        group_layout.addWidget(cmap_label)
        cmap_combo = QtWidgets.QComboBox()
        cmap_combo.addItems(AVAILABLE_COLORMAPS)
        cmap_combo.setCurrentText("coolwarm")
        cmap_combo.currentIndexChanged.connect(self.schedule_update)
        group_layout.addWidget(cmap_combo)
        
        # Density and opacity controls
        density_lower = self.create_labelled_slider(
            "Density Lower", 
            DENSITY_RANGE[0], 
            DENSITY_RANGE[1], 
            DENSITY_LOWER_DEFAULT, 
            group_layout
        )
        density_upper = self.create_labelled_slider(
            "Density Upper",
            DENSITY_RANGE[0],
            DENSITY_RANGE[1],
            DENSITY_UPPER_DEFAULT,
            group_layout
        )
        opacity_slider = self.create_labelled_slider(
            "Opacity (%)",
            0,
            100,
            OPACITY_DEFAULT,
            group_layout
        )
        sphere_slider = self.create_labelled_slider(
            "Sphere Size",
            SPHERE_SIZE_RANGE[0],
            SPHERE_SIZE_RANGE[1],
            SPHERE_SIZE_DEFAULT,
            group_layout
        )
        # Gamma correction control for histogram
        gamma_spinbox = self.create_labelled_double_spinbox(
            "Opacity Gamma",
            GAMMA_RANGE[0],
            GAMMA_RANGE[1],
            GAMMA_DEFAULT,
            0.1,
            group_layout
        )
        # Color selector
        random_color = "#%06x" % random.randint(0, 0xFFFFFF)
        color_button = QtWidgets.QPushButton("Select Color")
        color_button.setStyleSheet(f"background-color: {random_color}")
        color_button.setProperty("selected_color", random_color)
        color_button.clicked.connect(
            lambda _, a=atype, btn=color_button: self.choose_color(a, btn)
        )
        group_layout.addWidget(color_button)
        
        # Store widget references
        self.atom_settings_widgets[atype] = {
            "mode_combo": mode_combo,
            "density_lower": density_lower,
            "density_upper": density_upper,
            "opacity_slider": opacity_slider,
            "sphere_slider": sphere_slider,
            "opacity_gamma_spinbox": gamma_spinbox,
            "color_button": color_button,
            "cmap_combo": cmap_combo,
            "cmap_label": cmap_label,
        }
        
        # Initialize visibility
        self.update_atom_ui_visibility(atype)
        
        return group_box
            
    ###############################################################
    # UI Component Creation Helpers
    ###############################################################

    def create_labelled_slider(self, label_text: str, min_val: int, max_val: int, 
                            init_val: int, layout: QtWidgets.QLayout) -> Dict[str, QtWidgets.QWidget]:
        """Create a slider control with associated label.
        
        Args:
            label_text: Display text for the slider label
            min_val: Minimum slider value
            max_val: Maximum slider value
            init_val: Initial slider value
            layout: Parent layout for the new controls
            
        Returns:
            Dictionary containing slider components:
            {
                'container': QWidget containing all elements,
                'label': QLabel showing current value,
                'slider': QSlider instance
            }
        """
        container = QtWidgets.QWidget()
        container_layout = QtWidgets.QVBoxLayout(container)
        label = QtWidgets.QLabel(f"{label_text}: {init_val}")
        container_layout.addWidget(label)
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(init_val)
        # Update label when slider moves.
        slider.valueChanged.connect(lambda val, l=label, t=label_text: l.setText(f"{t}: {val}"))
        # *** New: trigger an update when the slider value changes ***
        slider.valueChanged.connect(self.schedule_update)
        container_layout.addWidget(slider)
        layout.addWidget(container)
        return {"container": container, "label": label, "slider": slider}

    def create_labelled_spinbox(self, label_text: str, min_val: int, max_val: int,
                              init_val: int, layout: QtWidgets.QLayout) -> QtWidgets.QSpinBox:
        """Create a labelled integer spinbox control.
        
        Args:
            label_text: Display text for the label
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            init_val: Initial value
            layout: Parent layout for the new controls
            
        Returns:
            Configured QSpinBox instance
        """
        container = QtWidgets.QWidget()
        container_layout = QtWidgets.QHBoxLayout(container)
        label = QtWidgets.QLabel(f"{label_text}:")
        spinbox = QtWidgets.QSpinBox()
        spinbox.setMinimum(min_val)
        spinbox.setMaximum(max_val)
        spinbox.setValue(init_val)
        spinbox.setFixedWidth(60)
        container_layout.addWidget(label)
        container_layout.addWidget(spinbox)
        layout.addWidget(container)
        spinbox.valueChanged.connect(self.schedule_update)
        return spinbox

    def create_labelled_double_spinbox(self, label_text: str, min_val: float, 
                                     max_val: float, init_val: float, step: float,
                                     layout: QtWidgets.QLayout) -> QtWidgets.QDoubleSpinBox:
        """Create a labelled floating-point spinbox control.
        
        Args:
            label_text: Display text for the label
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            init_val: Initial value
            step: Increment step size
            layout: Parent layout for the new controls
            
        Returns:
            Configured QDoubleSpinBox instance
        """
        container = QtWidgets.QWidget()
        container_layout = QtWidgets.QHBoxLayout(container)
        label = QtWidgets.QLabel(f"{label_text}:")
        spinbox = QtWidgets.QDoubleSpinBox()
        spinbox.setMinimum(min_val)
        spinbox.setMaximum(max_val)
        spinbox.setValue(init_val)
        spinbox.setSingleStep(step)
        spinbox.setFixedWidth(80)
        container_layout.addWidget(label)
        container_layout.addWidget(spinbox)
        layout.addWidget(container)
        spinbox.valueChanged.connect(self.schedule_update)
        return spinbox

    ###############################################################
    # UI EVENT HANDLERS
    ###############################################################
    
    def update_x_min(self, val: int) -> None:
        """Handle X minimum slider value changes.
        
        Ensures xmin <= xmax and triggers visual update.
        
        Args:
            val: New X minimum value
        """
        xmax_slider = self.xmax_slider_obj["slider"]
        if val > xmax_slider.value():
            xmax_slider.blockSignals(True)
            xmax_slider.setValue(val)
            xmax_slider.blockSignals(False)
            self.xmax_slider_obj["label"].setText(f"X Max: {val}")
        self.schedule_update(val)

    def update_x_max(self, val):
        """Handle X maximum slider value changes.

        Ensures xmax >= xmin and triggers visual update.

        Args:
            val: New X maximum value
        """
        xmin_slider = self.xmin_slider_obj["slider"]
        if val < xmin_slider.value():
            xmin_slider.blockSignals(True)
            xmin_slider.setValue(val)
            xmin_slider.blockSignals(False)
            self.xmin_slider_obj["label"].setText(f"X Min: {val}")
        self.schedule_update(val)

    def update_y_min(self, val):
        """Handle Y minimum slider value changes.
        
        Ensures ymin <= ymax and triggers visual update.
        
        Args:
            val: New Y minimum value
        """
        ymax_slider = self.ymax_slider_obj["slider"]
        if val > ymax_slider.value():
            ymax_slider.blockSignals(True)
            ymax_slider.setValue(val)
            ymax_slider.blockSignals(False)
            self.ymax_slider_obj["label"].setText(f"Y Max: {val}")
        self.schedule_update(val)

    def update_y_max(self, val):
        """Handle Y maximum slider value changes.

        Ensures ymax >= ymin and triggers visual update.

        Args:
            val: New Y maximum value
        """
        ymin_slider = self.ymin_slider_obj["slider"]
        if val < ymin_slider.value():
            ymin_slider.blockSignals(True)
            ymin_slider.setValue(val)
            ymin_slider.blockSignals(False)
            self.ymin_slider_obj["label"].setText(f"Y Min: {val}")
        self.schedule_update(val)

    def update_z_min(self, val):
        """Handle Z minimum slider value changes.

        Ensures zmin <= zmax and triggers visual update.

        Args:
            val: New Z minimum value
        """
        zmax_slider = self.zmax_slider_obj["slider"]
        if val > zmax_slider.value():
            zmax_slider.blockSignals(True)
            zmax_slider.setValue(val)
            zmax_slider.blockSignals(False)
            self.zmax_slider_obj["label"].setText(f"Z Max: {val}")
        self.schedule_update(val)

    def update_z_max(self, val):
        """Handle Z maximum slider value changes.

        Ensures zmax >= zmin and triggers visual update.

        Args:
            val: New Z maximum value
        """
        zmin_slider = self.zmin_slider_obj["slider"]
        if val < zmin_slider.value():
            zmin_slider.blockSignals(True)
            zmin_slider.setValue(val)
            zmin_slider.blockSignals(False)
            self.zmin_slider_obj["label"].setText(f"Z Min: {val}")
        self.schedule_update(val)

    def update_atom_ui_visibility(self, atype: str) -> None:
        """Update UI component visibility for atom type settings.
        
        Args:
            atype: Atom type identifier
        """
        ui = self.atom_settings_widgets[atype]
        mode = ui["mode_combo"].currentText()
        is_histogram = (mode == "Histogram")
        is_average = (mode == "Averaged Positions")

        # Make histogram-specific controls visible/invisible
        ui["density_lower"]["container"].setVisible(is_histogram)
        ui["density_upper"]["container"].setVisible(is_histogram)
        ui["opacity_slider"]["container"].setVisible(is_histogram)
        ui["opacity_gamma_spinbox"].parentWidget().setVisible(is_histogram) 
        ui["cmap_combo"].setVisible(is_histogram)
        ui["cmap_label"].setVisible(is_histogram)

        # Make average-specific controls visible/invisible
        ui["sphere_slider"]["container"].setVisible(is_average)
        ui["color_button"].setVisible(is_average)

    def schedule_update(self, value: Optional[Any] = None) -> None:
        """Schedule a visualisation update after UI parameter changes."""
        # Consider adding a small delay to batch updates during slider movements
        if self.update_timer.isActive():
            self.update_timer.stop()
        self.update_timer.start()

    def choose_color(self, atype: str, btn: QtWidgets.QPushButton) -> None:
        """Launch colour picker for atom type colour.
        
        Args:
            atype: Atom type identifier
            btn: Colour button to update
        """
        color = QtWidgets.QColorDialog.getColor(initial=QtGui.QColor(btn.property("selected_color")), parent=self)
        if color.isValid():
            new_color = color.name()
            btn.setStyleSheet(f"background-color: {new_color}")
            btn.setProperty("selected_color", new_color)
            self.schedule_update()

    def update_rotation_speed(self, value: float) -> None:
        """Update rotation speed in degrees per frame.
        
        Args:
            value: New rotation speed in degrees
        """
        self.rotation_azimuth = value
        print(f"Updated rotation speed to {value}°/frame")  

    def toggle_miller_controls(self, state: int) -> None:
        """Toggle visibility of Miller slicing controls.
        
        Args:
            state: Qt check state (Checked/Unchecked)
        """
        self.miller_controls.setVisible(state == QtCore.Qt.Checked)
        self.schedule_update()

    ###############################################################
    # DATA LOADING AND PREPROCESSING
    ###############################################################

    def _get_chemical_symbols_mapped(self, atoms: Any) -> List[str]:
        """Map atomic symbols using configured atom_type_map.
        
        Args:
            atoms: ASE Atoms object
            
        Returns:
            List of mapped atomic symbols similar to ASE Atoms.get_chemical_symbols()
        """
        if self.atom_type_map is None:
            return atoms.get_chemical_symbols()
        return [self.atom_type_map.get(sym, sym) for sym in atoms.get_chemical_symbols()]

    def load_and_precompute(self, input_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Load atomic trajectory data, apply initial ROI filtering, and compute per-species statistics.

        Algorithm:
          1. Loads frames from the file.
          2. Determines initial ROI absolute bounds (roi_min, roi_max).
          3. Uses the first frame to determine unique species and indices.
          4. Stacks per-frame positions.
          5. Filters positions based on the initial ROI.
          6. Computes filtered global positions and filtered averaged positions per species.
          7. Defines the grid based on the initial ROI bounds.
          8. Computes histograms using filtered data and the ROI-based grid.

        Args:
            input_config: Optional configuration override dict.

        Raises:
            ValueError: For unsupported file formats or invalid ROI.
        """
        if input_config is None:
            raise ValueError("No input configuration provided")

        file_path = input_config.get('path')
        file_format = input_config.get('format')
        frames_slice = input_config.get('slice')

        if isinstance(frames_slice, str):
            parts = frames_slice.split(':')
            nums = [int(p) if p else None for p in parts]
            frames_slice = slice(*nums)
        elif not isinstance(frames_slice, slice):
            raise ValueError("Invalid slice object")

        print("Loading trajectory data...")
        if file_format == 'pickle':
            import pickle
            with open(file_path, 'rb') as f:
                self.frames = pickle.load(f)
                if isinstance(frames_slice, slice): # Ensure slice is applied if provided
                     self.frames = self.frames[frames_slice]
        elif file_format == 'ase':
            from ase.io import read
            self.frames = read(file_path, index=frames_slice)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        if not self.frames:
             raise ValueError("No frames loaded. Check file path and slice.")

        print(f"Loaded {len(self.frames)} frames")

        first_frame = self.frames[0]
        cell = first_frame.get_cell() # Get cell for coordinate transformations

        self.initial_roi_min, self.initial_roi_max = _calculate_roi_bounds(ROI_DEFINITION, cell)

        symbols_first: List[str] = self._get_chemical_symbols_mapped(first_frame)
        unique_types: List[str] = sorted(set(symbols_first))

        indices_map: Dict[str, np.ndarray] = {
            atype: np.array([i for i, sym in enumerate(symbols_first) if sym == atype])
            for atype in unique_types
        }

        print("Processing frames (vectorised stacking)...")
        positions_array = np.stack([frame.get_positions() for frame in self.frames], axis=0)

        species_data = self._process_frames(positions_array, indices_map,
                                             self.initial_roi_min, self.initial_roi_max)
        self.atom_data = species_data

        if self.initial_roi_min is None or self.initial_roi_max is None:
             # Fallback needed if _calculate_roi_bounds returned None (shouldn't happen with current logic)
             raise RuntimeError("Initial ROI bounds not set correctly.")

        global_min = self.initial_roi_min
        global_max = self.initial_roi_max
        roi_dimensions = global_max - global_min

        # Ensure dimensions are non-zero to avoid division by zero
        if np.any(roi_dimensions <= 1e-9):
             print(f"Warning: Initial ROI dimension is zero or negative: {roi_dimensions}. Check ROI_DEFINITION.")
             # Handle degenerate case - perhaps default to a small size or raise error
             roi_dimensions = np.maximum(roi_dimensions, 1e-6) # Prevent zero division

        self.global_cell_origin = global_min
        nx = ny = nz = GRID_RESOLUTION
        self.global_cell_dims = np.array([nx, ny, nz])
        # Adjust spacing calculation for potentially non-cubic ROI
        self.global_spacing = roi_dimensions / (np.array([nx, ny, nz]) -1) # Use N-1 for spacing between N points
        self.global_spacing = np.where(self.global_spacing == 0, 1e-6, self.global_spacing) # Avoid zero spacing

        self.global_cell_center = global_min + roi_dimensions / 2

        print(f"Grid defined over ROI: Origin={self.global_cell_origin}, Spacing={self.global_spacing}")

        for atype in tqdm(unique_types, desc="Computing histograms"):
            positions = self.atom_data[atype]['global_positions'] # These are already filtered by _process_frames
            if positions.size == 0:
                 print(f"Warning: No positions found for species {atype} within the initial ROI.")
                 # Store empty/default data to prevent errors later
                 self.atom_data[atype]['histogram_data'] = {
                    'data': np.zeros((nx, ny, nz)),
                    'sorted_data': np.array([0.0]),
                    'origin': tuple(global_min),
                    'spacing': tuple(self.global_spacing),
                    'global_min': 0.0,
                    'global_max': 0.0,
                 }
                 continue

            # Ensure edges cover the full range of the ROI grid
            xedges = global_min[0] + np.arange(nx + 1) * self.global_spacing[0]
            yedges = global_min[1] + np.arange(ny + 1) * self.global_spacing[1]
            zedges = global_min[2] + np.arange(nz + 1) * self.global_spacing[2]
            # Adjust the last edge to exactly match roi_max to handle potential float precision issues
            xedges[-1] = global_max[0]
            yedges[-1] = global_max[1]
            zedges[-1] = global_max[2]

            # Clip positions to be strictly within bins defined by edges to avoid histogram errors
            # This is important if ROI bounds were slightly outside actual particle positions
            positions_clipped = np.clip(positions, global_min, global_max - 1e-9) # Subtract small epsilon

            hist, _ = np.histogramdd(positions_clipped, bins=(xedges, yedges, zedges))

            raw_hist = hist
            global_min_hist = float(raw_hist.min())
            global_max_hist = float(raw_hist.max())
            sorted_data = np.sort(raw_hist.ravel())
            
            self.atom_data[atype]['histogram_data'] = {
                'raw_data': raw_hist,
                'sorted_data': sorted_data,
                'origin': tuple(global_min),
                'spacing': tuple(self.global_spacing),
                'global_min': global_min_hist,
                'global_max': global_max_hist,
            }

        del positions_array # Free memory
        if hasattr(self, 'frames'):
             del self.frames # Free memory

    # MODIFY this method
    def _process_frames(self, positions_array: np.ndarray,
                       indices_map: Dict[str, np.ndarray],
                       roi_min: np.ndarray, roi_max: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Process trajectory frames to compute per-species statistics, filtered by initial ROI.

        Args:
            positions_array: Array of shape (n_frames, n_atoms, 3) containing all positions
            indices_map: Dictionary mapping species to atom indices
            roi_min: Minimum absolute coordinates of the initial ROI.
            roi_max: Maximum absolute coordinates of the initial ROI.

        Returns:
            Dictionary mapping species to ROI-filtered processed data:
            {
                'global_positions': All positions within ROI across frames (n_filtered, 3),
                'individual_averages': Per-atom averaged positions within ROI (m_filtered, 3)
            }
        """
        results: Dict[str, Dict[str, np.ndarray]] = {}
        for atype, indices in indices_map.items():
            # trajectories shape: (n_frames, n_atoms_of_type, 3)
            trajectories = positions_array[:, indices, :]
            # Flatten to get all positions for this type: shape (n_frames * n_atoms_of_type, 3)
            global_positions_unfiltered = trajectories.reshape(-1, 3)

            roi_mask_global = np.all((global_positions_unfiltered >= roi_min) &
                                     (global_positions_unfiltered <= roi_max), axis=1)
            global_positions_filtered = global_positions_unfiltered[roi_mask_global]

            # Compute the per-atom average over the frame axis.
            # shape: (n_atoms_of_type, 3)
            individual_averages_unfiltered = np.mean(trajectories[AVERAGE_POSITIONS_FRAME_SLICE], axis=0)

            roi_mask_avg = np.all((individual_averages_unfiltered >= roi_min) &
                                  (individual_averages_unfiltered <= roi_max), axis=1)
            individual_averages_filtered = individual_averages_unfiltered[roi_mask_avg]

            results[atype] = {
                'global_positions': global_positions_filtered,
                'individual_averages': individual_averages_filtered,
            }
        return results

    def _histogram_normalise_gamma(self, image: np.ndarray, gamma: float = 0.5) -> np.ndarray:
        """Apply gamma correction to normalised data.

        A gamma correction means that the intensity values are raised to the power of gamma.
        This looks like an s-shaped curve when plotted.
        
        Args:
            image: Input data array
            gamma: Gamma correction factor
            
        Returns:
            Gamma-corrected data array
        """
        data_min, data_max = image.min(), image.max()
        if np.isclose(data_max, data_min):
            return np.zeros_like(image)
        normalised = (image - data_min) / (data_max - data_min)
        gamma_corrected = normalised ** gamma
        return gamma_corrected

    ###############################################################
    # VISUALISATION METHODS
    ###############################################################

    def update_visualisation(self) -> None:
        """Main visualisation update method.
        
        Coordinates:
        - Clearing previous actors
        - ROI calculation
        - Data processing
        - VTK actor creation
        - View updates
        """
        # Clear previous visualization
        self._clear_actors()
        
        # Get current ROI settings
        roi_indices = self._get_roi_indices()
        grid_params = self._get_grid_parameters()
        
        # Precompute region data once
        region_data = self._precompute_region_data(roi_indices, grid_params)
        
        # Process each atom type using precomputed region
        for atype, ui in self.atom_settings_widgets.items():
            mode = ui["mode_combo"].currentText()
            if mode == "Hidden":
                continue
            if mode == "Histogram":
                self._visualise_histogram(atype, ui, region_data)
            elif mode == "Averaged Positions":
                self._visualise_averages(atype, ui, region_data)
        
        # Update camera view
        self._update_camera_view(region_data['focal_point'])

    def _clear_actors(self) -> None:
        """Remove all current VTK actors from the plotter."""
        for actor in list(self.rendered_actors.values()):
            self.plotter.remove_actor(actor)
        self.rendered_actors.clear()

    def _get_roi_indices(self) -> Dict[str, int]:
        """Get current region-of-interest indices from sliders.
        
        Returns:
            Dictionary with keys 'xmin', 'xmax', etc. containing grid indices
        """
        return {
            'xmin': self.xmin_slider_obj["slider"].value(),
            'xmax': self.xmax_slider_obj["slider"].value(),
            'ymin': self.ymin_slider_obj["slider"].value(),
            'ymax': self.ymax_slider_obj["slider"].value(),
            'zmin': self.zmin_slider_obj["slider"].value(),
            'zmax': self.zmax_slider_obj["slider"].value()
        }

    def _get_grid_parameters(self) -> Dict[str, Any]:
        """Get precomputed grid parameters. These are constant for all regions.
        
        Returns:
            Dictionary with grid metadata:
            {
                'origin': Grid origin in 3D space,
                'spacing': Voxel spacing,
                'cell_center': Simulation cell center
            }
        """
        return {
            'origin': np.array(self.global_cell_origin),
            'spacing': np.array(self.global_spacing),
            'cell_center': np.array(self.global_cell_center)
        }

    def _precompute_region_data(self, roi_indices: Dict[str, int],
                              grid_params: Dict[str, Any]) -> Dict[str, Any]:
        """Precompute region-related data for visualisation.
        
        Args:
            roi_indices: Current region-of-interest grid indices
            grid_params: Global grid parameters
            
        Returns:
            Dictionary containing precomputed region data:
            {
                'phys_bounds': (min, max) physical coordinates,
                'grid_coords': Meshgrid of voxel coordinates,
                'miller_params': Miller slicing parameters,
                'focal_point': Calculated view focal point,
                'miller_mask': Boolean mask for Miller slicing
            }
        """
        # Generate hash values for current settings
        roi_hash = hash(tuple(sorted(roi_indices.items())))
        miller_params = self._get_miller_parameters()
        miller_hash = hash((
            miller_params['use_miller'],
            miller_params['h'],
            miller_params['k'],
            miller_params['l'],
            miller_params['thickness'],
            miller_params['offset']
        ))
        
        # Check cache for reuse
        if (self._region_data_cache is not None and
            roi_hash == self._last_roi_hash and
            miller_hash == self._last_miller_hash):
            # Cache hit - return the cached data
            print("Using cached region data")
            return self._region_data_cache
        
        # Cache miss - compute new region data
        print("Computing new region data")
        
        # 1. Physical bounds
        phys_bounds = self._compute_physical_bounds(roi_indices, grid_params)
        
        if miller_params['use_miller'] and miller_params['n'] is not None:
            # Compute voxel centers and mask in one pass.
            roi_voxel_centers, miller_mask = self.compute_roi_miller_mask(roi_indices, grid_params, miller_params)
        else:
            miller_mask = None
            roi_voxel_centers = None  # if needed later

        # Optionally, compute the focal point as the mean of the masked voxel centers.
        if miller_mask is not None and np.any(miller_mask):
            # Take a strided sample of both arrays together
            sample_voxel_centers = roi_voxel_centers[::5, ::5, ::5]
            sample_mask = miller_mask[::5, ::5, ::5]
            if np.any(sample_mask):
                # Only use the points that are in the mask
                focal_point = sample_voxel_centers[sample_mask].mean(axis=0)
            else:
                focal_point = 0.5 * (phys_bounds[0] + phys_bounds[1])
        else:
            focal_point = 0.5 * (phys_bounds[0] + phys_bounds[1])

        # Store the computed data
        region_data = {
            'roi_indices': roi_indices,
            'phys_bounds': phys_bounds,
            'roi_voxel_centers': roi_voxel_centers,
            'miller_mask': miller_mask,
            'focal_point': focal_point,
            'grid_params': grid_params,
            'miller_params': miller_params
        }
        
        # Update cache
        self._region_data_cache = region_data
        self._last_roi_hash = roi_hash
        self._last_miller_hash = miller_hash
        
        return region_data

    def _compute_physical_bounds(self, roi_indices: Dict[str, int],
                               grid_params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Convert grid indices to physical coordinates.
        
        Args:
            roi_indices: Region-of-interest grid indices
            grid_params: Grid metadata
            
        Returns:
            Tuple of (min, max) physical coordinates for ROI
        """
        origin = grid_params['origin']
        spacing = grid_params['spacing']
        
        roi_min = origin + np.array([
            roi_indices['xmin'],
            roi_indices['ymin'],
            roi_indices['zmin']
        ]) * spacing
        
        roi_max = origin + np.array([
            roi_indices['xmax'],
            roi_indices['ymax'],
            roi_indices['zmax']
        ]) * spacing
        
        return roi_min, roi_max

    def compute_roi_miller_mask(self, roi_indices: Dict[str, int],
                            grid_params: Dict[str, Any],
                            miller_params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute voxel centers for the ROI and return a Boolean mask according to the Miller slicing.

        Args:
            roi_indices: Dictionary with keys 'xmin', 'xmax', etc.
            grid_params: Dictionary containing 'origin', 'spacing', and 'cell_center'.
            miller_params: Dictionary containing 'use_miller', 'n', 'thickness', and 'offset'.

        Returns:
            A tuple (voxel_centers, mask) where:
            - voxel_centers is an array of shape (Nx, Ny, Nz, 3) holding the physical coordinates.
            - mask is a Boolean array of shape (Nx, Ny, Nz) indicating voxels within the slice.
        """
        spacing = grid_params['spacing']
        origin = grid_params['origin']
        # Compute ROI physical origin
        roi_origin = origin + np.array([roi_indices['xmin'],
                                        roi_indices['ymin'],
                                        roi_indices['zmin']]) * spacing
        dims = np.array([roi_indices['xmax'] - roi_indices['xmin'] + 1,
                        roi_indices['ymax'] - roi_indices['ymin'] + 1,
                        roi_indices['zmax'] - roi_indices['zmin'] + 1])

        # Create coordinate arrays for each axis, placing voxel centers at mid-voxel.
        x = roi_origin[0] + (np.arange(dims[0]) + 0.5) * spacing[0]
        y = roi_origin[1] + (np.arange(dims[1]) + 0.5) * spacing[1]
        z = roi_origin[2] + (np.arange(dims[2]) + 0.5) * spacing[2]
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        voxel_centers = np.stack([X, Y, Z], axis=-1)

        # Compute distances from each voxel center to the Miller plane.
        cell_center = grid_params['cell_center']
        n = miller_params['n']  # already normalized if valid
        distances = np.abs(np.sum((voxel_centers - cell_center) * n, axis=-1) -
                        miller_params['offset'])
        mask = distances < (miller_params['thickness'] / 2)

        return voxel_centers, mask

    def filter_points_by_miller(self, points: np.ndarray,
                            cell_center: np.ndarray,
                            miller_params: Dict[str, Any]) -> np.ndarray:
        """
        Filters a set of points according to the Miller slicing parameters.

        Args:
            points: Array of shape (N, 3) containing 3D points.
            cell_center: The central coordinate of the simulation cell.
            miller_params: Dictionary with keys 'use_miller', 'n', 'thickness', and 'offset'.

        Returns:
            Filtered array of points within the Miller slice.
        """
        if miller_params['use_miller'] and miller_params['n'] is not None:
            distances = np.abs(np.dot(points - cell_center, miller_params['n']) -
                            miller_params['offset'])
            return points[distances < (miller_params['thickness'] / 2)]
        return points

    def _get_miller_parameters(self) -> Dict[str, Any]:
        """Get current Miller slicing parameters.
        
        Returns:
            Dictionary with keys:
            {
                'use_miller': True if slicing enabled,
                'h', 'k', 'l': Miller indices,
                'thickness': Slice thickness,
                'offset': Slice offset,
                'n': Normalised normal vector
            }
        """
        params = {
            'use_miller': self.miller_checkbox.isChecked(),
            'h': self.miller_h.value(),
            'k': self.miller_k.value(),
            'l': self.miller_l.value(),
            'thickness': self.miller_thickness.value(),
            'offset': self.miller_offset.value(),
            'n': None
        }
        
        if params['use_miller']:
            norm = np.sqrt(params['h']**2 + params['k']**2 + params['l']**2)
            params['n'] = np.array([params['h'], params['k'], params['l']]) / norm if norm > 0 else None
        
        return params

    def _visualise_histogram(self, atype: str, ui: Dict[str, Any], region_data: Dict[str, Any]) -> None:
        """Visualise histogram data with dynamic min/max based on current ROI and Miller slice."""
        try:
            hist_data = self.atom_data[atype]['histogram_data']
            sigma = self.sigma_slider_obj["slider"].value()
            raw_data = hist_data['raw_data']

            # Initialise the cache dict if needed.
            if not hasattr(self, "_smoothed_hist_cache"):
                self._smoothed_hist_cache = {}

            # Compute a key from the sigma and raw_data content
            cache_key = (sigma, hash(raw_data.tobytes()))
            if cache_key in self._smoothed_hist_cache:
                processed_data = self._smoothed_hist_cache[cache_key]
            else:
                if sigma > 0:
                    processed_data = gaussian_filter(raw_data, sigma=sigma)
                else:
                    processed_data = raw_data
                self._smoothed_hist_cache[cache_key] = processed_data

            sub_data = processed_data[
                region_data['roi_indices']['xmin']:region_data['roi_indices']['xmax']+1,
                region_data['roi_indices']['ymin']:region_data['roi_indices']['ymax']+1,
                region_data['roi_indices']['zmin']:region_data['roi_indices']['zmax']+1
            ]

            # Apply Miller plane masking
            miller_mask = region_data['miller_mask']
            filtered_data = np.where(miller_mask, sub_data, 0) if miller_mask is not None else sub_data


            # Calculate local min/max based on current view
            if miller_mask is not None:
                visible_data = sub_data[miller_mask]
            else:
                visible_data = sub_data.ravel()

            if visible_data.size == 0:
                return  # No data to visualize

            data_min = visible_data.min()
            data_max = visible_data.max()

            # Normalize data using local range
            if np.isclose(data_max, data_min):
                normalized = np.zeros_like(filtered_data)
            else:
                normalized = (filtered_data - data_min) / (data_max - data_min)
            normalized = np.clip(normalized, 0, 1)  # Ensure within [0,1]

            # Get visualisation parameters
            lower = ui["density_lower"]["slider"].value() / 255.0
            upper = ui["density_upper"]["slider"].value() / 255.0
            max_alpha = ui["opacity_slider"]["slider"].value() / 100.0
            gamma = ui["opacity_gamma_spinbox"].value()
            cmap = plt.get_cmap(ui["cmap_combo"].currentText())

            # Create RGBA array
            rgba = np.zeros((*normalized.shape, 4), dtype=np.uint8)
            colors = (cmap(normalized)[..., :3] * 255).astype(np.uint8)

            # Calculate alpha channel
                        # Calculate alpha channel
            if gamma == 0:
                in_range = (normalized > lower) & (normalized <= upper)
            else:
                in_range = (normalized >= lower) & (normalized <= upper)
            scaled = np.clip((normalized - lower) / (upper - lower + 1e-9), 0, 1)
            # Apply gamma correction; note that gamma=0 will now keep zeros unshaded because np.power(0, 0) remains 0 via this mask.
            corrected = scaled ** gamma
            alpha = (corrected * max_alpha * 255).astype(np.uint8)
            alpha[~in_range] = 0

            # Outside of the ROI make completely transparent
            rgba[..., :3] = colors
            rgba[..., 3] = alpha

            # Create ImageData structure
            vol = pv.ImageData()
            vol.dimensions = np.array(filtered_data.shape) + 1
            vol.origin = region_data['phys_bounds'][0]
            vol.spacing = region_data['grid_params']['spacing']
            vol.cell_data["rgba"] = rgba.reshape(-1, 4, order='F')

            # Add to plotter
            actor = self.plotter.add_volume(
                vol,
                scalars="rgba",
                clim=[0, 255],
                scalar_bar_args={"title": f"{atype} Density"},
                blending='composite',
                shade=False,
                opacity='linear',
            )
            actor.GetProperty().SetInterpolationType(VTK_CUBIC_INTERPOLATION)
            self.rendered_actors[atype] = actor

        except Exception as e:
            print(f"Error visualizing {atype}: {str(e)}")

            
    def _visualise_averages(self, atype: str, ui: Dict[str, Any],
                          region_data: Dict[str, Any]) -> None:
        """Visualise averaged positions for given atom type.
        
        Args:
            atype: Atom type identifier
            ui: UI component dictionary
            region_data: Precomputed region data
        """
        points = self.atom_data[atype]['individual_averages']
        roi_min, roi_max = region_data['phys_bounds']
        miller_params = region_data['miller_params']
        cell_center = region_data['grid_params']['cell_center']
        
        # Spatial filtering
        in_roi = (
            (points[:, 0] >= roi_min[0]) & (points[:, 0] <= roi_max[0]) &
            (points[:, 1] >= roi_min[1]) & (points[:, 1] <= roi_max[1]) &
            (points[:, 2] >= roi_min[2]) & (points[:, 2] <= roi_max[2])
        )
        filtered_points = points[in_roi]
        
        # Miller plane filtering
        filtered_points = self.filter_points_by_miller(filtered_points, cell_center, miller_params)
        
        if filtered_points.size == 0:
            return
        
        # Create glyphs
        sphere_size = ui["sphere_slider"]["slider"].value() * 0.1
        color = ui["color_button"].property("selected_color")
        cloud = pv.PolyData(filtered_points)
        glyphs = cloud.glyph(geom=pv.Sphere(radius=sphere_size))
        actor = self.plotter.add_mesh(glyphs, color=color, name=f"{atype}_avg")
        self.rendered_actors[atype] = actor

    def _update_camera_view(self, focal_point: np.ndarray) -> None:
        """Update camera to focus on specified point.
        
        Args:
            focal_point: 3D coordinates to focus on
        """
        self.plotter.reset_camera()
        self.plotter.camera.SetFocalPoint(*focal_point)
        self.plotter.render()

    ###############################################################
    # VIEW CONTROL METHODS
    ###############################################################
    
    def reset_view(self) -> None:
        """Reset camera to default isometric view."""
        self.plotter.camera.up = (0, 0, 1)
        self.plotter.camera.position = (1, 1, 1)
        self.plotter.camera.focal_point = (0, 0, 0)
        self.plotter.reset_camera()
        self.plotter.render()

    def align_miller_view(self) -> None:
        """Align view perpendicular to current Miller plane."""
        if not hasattr(self, 'miller_h'):
            return

        # Get Miller indices
        h = self.miller_h.value()
        k = self.miller_k.value()
        l = self.miller_l.value()

        # Calculate normal vector
        normal = np.array([h, k, l])
        norm = np.linalg.norm(normal)
        
        if norm == 0:
            return
        
        # Normalise the vector
        normal = normal / norm
        
        # Set camera position along normal vector
        self.plotter.camera.up = (0, 0, 1)  # Reset up vector
        cell_center = self.global_cell_center
        
        # Position camera along normal vector, looking at cell center
        camera_distance = np.linalg.norm(self.plotter.camera.position - cell_center)
        camera_pos = cell_center + normal * camera_distance
        
        self.plotter.camera.position = camera_pos
        self.plotter.camera.focal_point = cell_center
        self.plotter.reset_camera()
        self.plotter.render()

    def toggle_rotation(self) -> None:
        """Toggle automated rotation animation on/off."""
        if not self.rotation_timer.isActive():
            self.rotation_timer.start()
            self.rotate_button.setText("Stop")
        else:
            self.rotation_timer.stop()
            self.rotate_button.setText("Rotate")

    def rotate_view(self) -> None:
        """Rotate view by configured azimuth increment."""
        self.plotter.camera.Azimuth(self.rotation_azimuth)
        self.plotter.render()

    def align_view(self, axis: str) -> None:
        """Align camera view along specified axis.
        
        Args:
            axis: 'x', 'y', or 'z' for desired view direction
        """
        self.plotter.camera.up = (0, 0, 1)  # Reset up vector
        if axis == 'x':
            self.plotter.camera.position = (1, 0, 0)
            self.plotter.camera.focal_point = (0, 0, 0)
        elif axis == 'y':
            self.plotter.camera.position = (0, 1, 0)
            self.plotter.camera.focal_point = (0, 0, 0)
        elif axis == 'z':
            self.plotter.camera.position = (0, 0, 1)
            self.plotter.camera.focal_point = (0, 0, 0)
        self.plotter.reset_camera()
        self.plotter.render()
    
    ###############################################################
    # Cleanup and Utilities
    ###############################################################

    def cleanup(self) -> None:
        """Perform cleanup before application exit.
        
        Order:
        1. Stop timers
        2. Disconnect signals 
        3. Clear VTK actors and plotter
        4. Clear data structures
        5. Force garbage collection
        """
        # Stop all timers first
        if hasattr(self, 'update_timer'):
            self.update_timer.stop()
        if hasattr(self, 'rotation_timer'):
            self.rotation_timer.stop()
            
        # Disconnect signals that could trigger during cleanup
        self._disconnect_signals()
        
        # Clear VTK resources
        self._clear_actors()
        if hasattr(self, 'plotter'):
            self.plotter.close()
            self.plotter.deep_clean()
            
        # Clear data structures
        if hasattr(self, 'atom_data'):
            del self.atom_data
        if hasattr(self, '_rendered_actors'):
            del self._rendered_actors
        if hasattr(self, 'plotter'):
            del self.plotter
            
        # Force garbage collection
        import gc
        gc.collect()

    def _disconnect_signals(self) -> None:
        """Safely disconnect all Qt signals."""
        try:
            # Core UI signals
            if hasattr(self, 'miller_checkbox'):
                self.miller_checkbox.stateChanged.disconnect()
            if hasattr(self, 'rotate_button'):
                self.rotate_button.clicked.disconnect()
            if hasattr(self, 'view_x_button'):
                self.view_x_button.clicked.disconnect()
            if hasattr(self, 'view_y_button'):
                self.view_y_button.clicked.disconnect()
            if hasattr(self, 'view_z_button'):
                self.view_z_button.clicked.disconnect()
            if hasattr(self, 'reset_view_button'):
                self.reset_view_button.clicked.disconnect()
            if hasattr(self, 'rotation_speed'):
                self.rotation_speed.valueChanged.disconnect()
            
            # Timer signals
            if hasattr(self, 'update_timer'):
                self.update_timer.timeout.disconnect()
            if hasattr(self, 'rotation_timer'):
                self.rotation_timer.timeout.disconnect()

            # ROI slider signals
            for name in ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax']:
                slider_obj = getattr(self, f'{name}_slider_obj', None)
                if slider_obj and 'slider' in slider_obj:
                    slider_obj['slider'].valueChanged.disconnect()
                    
            # Atom settings signals
            if hasattr(self, 'atom_settings_widgets'):
                for widgets in self.atom_settings_widgets.values():
                    if 'mode_combo' in widgets:
                        widgets['mode_combo'].currentIndexChanged.disconnect()
                    if 'sphere_slider' in widgets and 'slider' in widgets['sphere_slider']:
                        widgets['sphere_slider']['slider'].valueChanged.disconnect()
                    if 'color_button' in widgets:
                        widgets['color_button'].clicked.disconnect()
        except Exception as e:
            print(f"Warning during signal disconnect: {e}")

    ###############################################################
    # Class Properties
    ###############################################################
    
    @property
    def rendered_actors(self) -> Dict[str, Any]:
        """dict: Currently active VTK actors."""
        if not hasattr(self, "_rendered_actors"):
            self._rendered_actors = {}
        return self._rendered_actors

    @rendered_actors.setter
    def rendered_actors(self, value: Dict[str, Any]) -> None:
        """Update the currently active VTK actors."""
        self._rendered_actors = value

    ###############################################################
    # Run Application
    ###############################################################

if __name__ == '__main__':
    # LETS GO
    app = QtWidgets.QApplication(sys.argv)
    window = DensityVisualiser()
    window.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
    window.show()
    
    sys.exit(app.exec_())
    # Goodbye :(
