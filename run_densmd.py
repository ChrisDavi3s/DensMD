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
from pyvistaqt import QtInteractor
from typing import List, Dict, Tuple, Optional
from typing import Any, Dict, Tuple, Optional
from vtk import vtkPiecewiseFunction
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
    'path': '/Users/chrisdavies/Downloads/MD.log1-10.pickle',
    'format': 'pickle',  # 'pickle' or 'ase'
    'slice': "::10",  # ASE index string (e.g. ':' for all frames, '::2' for every 2nd frame) or a slice object
}

# Atom type mapping for visualization (None to use original types)
# THIS IS AN EXAMPLE SETUP
ATOM_TYPE_MAP = {
    'H': 'Cs',
    'He': 'Pb',
    'Li': 'I',
    'Be': 'Sn'
}

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

# Visualization Parameters
SPHERE_SIZE_RANGE = (1, 50)
SPHERE_SIZE_DEFAULT = 5
DENSITY_RANGE = (0, 255)
DENSITY_LOWER_DEFAULT = 77
DENSITY_UPPER_DEFAULT = 178
OPACITY_DEFAULT = 100

# Miller Indices Configuration
MILLER_INDEX_RANGE = (-10, 10)
MILLER_INDEX_DEFAULT = 1
MILLER_THICKNESS_RANGE = (0.1, 20.0)
MILLER_THICKNESS_DEFAULT = 2.0
MILLER_OFFSET_RANGE = (-10.0, 10.0)
MILLER_OFFSET_DEFAULT = 0.0

# Atom Type Mapping
# Maps input atom types to desired visualization types
# Make this None to use the original atom types
ATOM_TYPE_MAP = {
    'H': 'Cs',
    'He': 'Pb',
    'Li': 'I',
    'Be': 'Sn'
}

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
        
        # Initialize basic window properties and core parameters
        self._init_window_properties()
        
        # Set up main layout and visualization components
        self._init_main_layout()
        
        # Initialize control panel components
        self._init_control_panel()
        
        # Set up global slicing controls
        self._init_global_slicing()
        
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
        self.load_and_precompute()
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
        if mode == "Histogram":
            ui["density_lower"]["container"].setVisible(True)
            ui["density_upper"]["container"].setVisible(True)
            ui["opacity_slider"]["container"].setVisible(True)
            ui["sphere_slider"]["container"].setVisible(False)
            ui["color_button"].setVisible(False)
            ui["cmap_combo"].setVisible(True)
            ui["cmap_label"].setVisible(True)
        elif mode == "Averaged Positions":
            ui["density_lower"]["container"].setVisible(False)
            ui["density_upper"]["container"].setVisible(False)
            ui["opacity_slider"]["container"].setVisible(False)
            ui["sphere_slider"]["container"].setVisible(True)
            ui["color_button"].setVisible(True)
            ui["cmap_combo"].setVisible(False)
            ui["cmap_label"].setVisible(False)
        else:  # Hidden
            ui["density_lower"]["container"].setVisible(False)
            ui["density_upper"]["container"].setVisible(False)
            ui["opacity_slider"]["container"].setVisible(False)
            ui["sphere_slider"]["container"].setVisible(False)
            ui["color_button"].setVisible(False)
            ui["cmap_combo"].setVisible(False)
            ui["cmap_label"].setVisible(False)

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
        Load atomic trajectory data from file and compute per-species statistics.

        Algorithm:
          1. Loads frames from the file (using pickle or ase formats).
          2. Uses the first frame to determine the unique species and their indices.
          3. Stacks the per-frame positions into a 3D NumPy array for vectorised indexing.
          4. Computes, for each species, (a) a flattened array of global positions and
             (b) the per-atom averaged positions (by stacking the trajectory over frames).
          5. Computes the cell boundaries from the first frame’s cell (and its 8 corners)
             so the global histogram grid can be determined.
          6. For each species, generates a 3D histogram from the global positions, applies
             Gaussian smoothing, and stores density information.

                
        Args:
            input_config: Optional configuration override dict with keys:
                - path: Input file path
                - format: File format ('pickle' or 'ase')
                - slice: Frame slicing specification
                
        Raises:
            ValueError: For unsupported file formats

        """
        if input_config is None:
            input_config = {
                'path': '/Users/chrisdavies/Downloads/MD.log1-10.pickle',
                'format': 'pickle',
                'slice': "::10",
            }

        file_path = input_config.get('path')
        file_format = input_config.get('format')
        frames_slice = input_config.get('slice')

        # Convert slice string (e.g. "::10") to a proper slice object.
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
                self.frames = self.frames[frames_slice]
        elif file_format == 'ase':
            from ase.io import read
            ase_index = frames_slice  # Here you might want to customize for ASE
            self.frames = read(file_path, index=frames_slice)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        print(f"Loaded {len(self.frames)} frames")

        # Assume the ordering of atoms is consistent; use the first frame to get species.
        first_frame = self.frames[0]
        symbols_first: List[str] = self._get_chemical_symbols_mapped(first_frame)
        unique_types: List[str] = sorted(set(symbols_first))

        # Precompute index mapping: for each species, record a NumPy array of indices.
        indices_map: Dict[str, np.ndarray] = {
            atype: np.array([i for i, sym in enumerate(symbols_first) if sym == atype])
            for atype in unique_types
        }

        # Vectorised processing: stack all frame positions into one 3D array.
        # all_positions has shape (n_frames, natoms, 3)
        print("Processing frames (vectorised stacking)...")
        positions_array = np.stack([frame.get_positions() for frame in self.frames], axis=0)

        # Use the helper to compute per-species trajectories
        species_data = self._process_frames(positions_array, indices_map)

        # Store results in self.atom_data: for each species, store the global positions
        # and the individual per-atom averages.
        self.atom_data = species_data

        # --- Compute cell info for global histogram grid ---
        cell = first_frame.get_cell()
        # Compute the 8 corners of the cell.
        corners = np.array([np.dot(np.array([i, j, k]), cell)
                            for i in (0, 1) for j in (0, 1) for k in (0, 1)])
        global_min = corners.min(axis=0)
        global_max = corners.max(axis=0)
        cell_lengths = global_max - global_min
        self.global_cell_origin = global_min
        nx = ny = nz = GRID_RESOLUTION
        self.global_cell_dims = np.array([nx, ny, nz])
        self.global_spacing = cell_lengths / GRID_RESOLUTION
        self.global_cell_center = global_min + cell_lengths / 2

        # --- Generate histogram data for each species using the calculated grid bounds ---
        for atype in tqdm(unique_types, desc="Computing histograms"):
            positions = self.atom_data[atype]['global_positions']
            xedges = np.linspace(global_min[0], global_max[0], nx + 1)
            yedges = np.linspace(global_min[1], global_max[1], ny + 1)
            zedges = np.linspace(global_min[2], global_max[2], nz + 1)
            hist, _ = np.histogramdd(positions, bins=(xedges, yedges, zedges))

            # Apply Gaussian smoothing.
            smoothed_hist = gaussian_filter(hist, sigma=GAUSSIAN_SIGMA)
            # Instead of sorting, obtain min and max directly.
            global_min_hist = float(smoothed_hist.min())
            global_max_hist = float(smoothed_hist.max())
            sorted_data = np.sort(smoothed_hist.ravel())

            self.atom_data[atype]['histogram_data'] = {
                'data': smoothed_hist,
                'sorted_data': sorted_data,
                'origin': tuple(global_min),
                'spacing': tuple(self.global_spacing),
                'global_min': global_min_hist,
                'global_max': global_max_hist,
            }

        # Free temporary array if desired.
        del positions_array
        del self.frames

    def _process_frames(self, positions_array: np.ndarray,
                       indices_map: Dict[str, np.ndarray]) -> Dict[str, Dict[str, np.ndarray]]:
        """Process trajectory frames to compute per-species statistics.
        
        Args:
            positions_array: Array of shape (n_frames, n_atoms, 3) containing all positions
            indices_map: Dictionary mapping species to atom indices
            
        Returns:
            Dictionary mapping species to processed data:
            {
                'global_positions': All positions across frames (n, 3),
                'individual_averages': Per-atom averaged positions (m, 3)
            }
        """
        results: Dict[str, Dict[str, np.ndarray]] = {}
        for atype, indices in indices_map.items():
            # Use NumPy’s advanced indexing to extract the positions corresponding to this species.
            # trajectories has shape (n_frames, n_atoms, 3).
            trajectories = positions_array[:, indices, :]
            # Flatten the first two dimensions to obtain all positions for the histogram.
            global_positions = trajectories.reshape(-1, 3)
            # Compute the per-atom average over the frame axis.
            individual_averages = np.mean(trajectories, axis=0)
            results[atype] = {
                'global_positions': global_positions,
                'individual_averages': individual_averages,
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
        # 1. Physical bounds
        phys_bounds = self._compute_physical_bounds(roi_indices, grid_params)
        
        # 2. Grid coordinates
        grid_coords = self._compute_grid_coordinates(phys_bounds, grid_params['spacing'])
        
        # 3. Miller parameters
        miller_params = self._get_miller_parameters()
        
        # 4. Focal point
        focal_point = self._compute_focal_point(
            roi_indices, grid_params, phys_bounds, miller_params
        )

        # 5. Miller mask - Calculate only when needed
        miller_params = self._get_miller_parameters()
        miller_mask = None
        need_recalc = False

        # Check if we need to recalculate the mask
        if miller_params['use_miller']:
            need_recalc = (not hasattr(self, '_cached_miller_mask') or 
                        not hasattr(self, '_cached_miller_params') or
                        self._cached_miller_params != miller_params or
                        not hasattr(self, '_cached_roi_indices') or
                        self._cached_roi_indices != roi_indices)
            
        if miller_params['use_miller'] and need_recalc:
            miller_mask = self._compute_miller_mask(grid_coords, grid_params['cell_center'], miller_params)
            self._cached_miller_mask = miller_mask
            self._cached_miller_params = miller_params.copy()
            self._cached_roi_indices = roi_indices.copy()
        elif miller_params['use_miller']:
            miller_mask = self._cached_miller_mask

        return {
            'roi_indices': roi_indices,
            'phys_bounds': phys_bounds,
            'grid_coords': grid_coords,
            'miller_params': miller_params,
            'focal_point': focal_point,
            'miller_mask': miller_mask,
            'grid_params': grid_params
        }

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

    def _compute_grid_coordinates(self, phys_bounds: Tuple[np.ndarray, np.ndarray],
                                spacing: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate physical coordinates for ROI voxel centers.
        
        Args:
            phys_bounds: (min, max) physical coordinates
            spacing: Voxel spacing
            
        Returns:
            Tuple of (X, Y, Z) meshgrid arrays for voxel centers
        """
        roi_min, roi_max = phys_bounds
        dims = ((roi_max - roi_min) / spacing + 1).astype(int)
        
        return np.meshgrid(
            np.linspace(roi_min[0] + spacing[0]/2, roi_max[0] - spacing[0]/2, dims[0]),
            np.linspace(roi_min[1] + spacing[1]/2, roi_max[1] - spacing[1]/2, dims[1]),
            np.linspace(roi_min[2] + spacing[2]/2, roi_max[2] - spacing[2]/2, dims[2]),
            indexing='ij'
        )

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

    def _compute_focal_point(self, roi_indices: Dict[str, int],
                           grid_params: Dict[str, Any], 
                           phys_bounds: Tuple[np.ndarray, np.ndarray],
                           miller_params: Dict[str, Any]) -> np.ndarray:
        """Calculate optimal focal point for current view.
        
        Args:
            roi_indices: Region-of-interest grid indices
            grid_params: Grid metadata
            phys_bounds: Physical ROI bounds
            miller_params: Miller slicing parameters
            
        Returns:
            3D focal point coordinates
        """

        if miller_params['use_miller'] and miller_params['n'] is not None:
            
            sample_count = self.global_cell_dims.prod() // 5
            
            # Generate random samples within ROI
            random_samples = np.random.random((sample_count, 3))
            voxel_centers = np.array([
                grid_params['origin'][0] + (roi_indices['xmin'] + random_samples[:, 0] * (roi_indices['xmax'] - roi_indices['xmin'])) * grid_params['spacing'][0],
                grid_params['origin'][1] + (roi_indices['ymin'] + random_samples[:, 1] * (roi_indices['ymax'] - roi_indices['ymin'])) * grid_params['spacing'][1],
                grid_params['origin'][2] + (roi_indices['zmin'] + random_samples[:, 2] * (roi_indices['zmax'] - roi_indices['zmin'])) * grid_params['spacing'][2]
            ]).T

            
            # Compute distances to Miller plane
            distances = np.abs(
                np.dot(voxel_centers - grid_params['cell_center'], miller_params['n']) -
                miller_params['offset']
            )
            
            # Get voxels within slice thickness
            mask = distances < (miller_params['thickness'] / 2)
            valid_voxels = voxel_centers[mask]
            
            if valid_voxels.size > 0:
                return valid_voxels.mean(axis=0)
        
        # Fallback to geometric center
        return 0.5 * (phys_bounds[0] + phys_bounds[1])

    def _compute_miller_mask(self, grid_coords: Tuple[np.ndarray, np.ndarray, np.ndarray],
                           cell_center: np.ndarray,
                           miller_params: Dict[str, Any]) -> Optional[np.ndarray]:
        """Compute boolean mask for Miller plane slicing.
        
        Args:
            grid_coords: Voxel coordinate meshgrid
            cell_center: Simulation cell center
            miller_params: Miller slicing parameters
            
        Returns:
            3D boolean array indicating voxels within slice, or None
        """
        if not miller_params['use_miller'] or miller_params['n'] is None:
            return None
            
        X, Y, Z = grid_coords
        n = miller_params['n']
        
        coords = np.stack([X, Y, Z], axis=-1)
        distances = np.abs(np.sum((coords - cell_center) * n, axis=-1) - miller_params['offset'])
        
        return distances < (miller_params['thickness'] / 2)

    def _visualise_histogram(self, atype: str, ui: Dict[str, Any],
                           region_data: Dict[str, Any]) -> None:
        """Visualise density histogram for given atom type.
        
        Args:
            atype: Atom type identifier
            ui: UI component dictionary
            region_data: Precomputed region data
        """
        # Extract region data
        grid_coords = region_data['grid_coords']
        miller_mask = region_data['miller_mask']
        spacing = region_data['grid_params']['spacing']
        
        # Get histogram data subset
        hist_data = self.atom_data[atype]['histogram_data']['data']
        sub_data = hist_data[
            region_data['roi_indices']['xmin']:region_data['roi_indices']['xmax']+1,
            region_data['roi_indices']['ymin']:region_data['roi_indices']['ymax']+1,
            region_data['roi_indices']['zmin']:region_data['roi_indices']['zmax']+1
        ]
        
        # Apply Miller mask if available
        filtered_data = np.where(miller_mask, sub_data, 0) if miller_mask is not None else sub_data
        
        # Normalisation and thresholding
        norm_data = self._histogram_normalise_gamma(filtered_data)
        lower = ui["density_lower"]["slider"].value() / 255.0
        upper = ui["density_upper"]["slider"].value() / 255.0
        thresh_data = np.where(lower <= upper, np.clip((norm_data - lower) / (upper - lower), 0, 1), 0)

        # Create volume
        vol = pv.ImageData()
        vol.dimensions = np.array(sub_data.shape) + 1
        vol.origin = (grid_coords[0].min(), grid_coords[1].min(), grid_coords[2].min())
        vol.spacing = spacing
        vol.cell_data["density"] = thresh_data.ravel(order="F")
        
        # Configure visualization
        opacity = ui["opacity_slider"]["slider"].value() / 100.0
        cmap = ui["cmap_combo"].currentText()
        
        actor = self.plotter.add_volume(
            vol, cmap=cmap, clim=[0, 1],
            scalar_bar_args={"title": f"Density: {atype}"}
        )
        
        # Configure opacity transfer function
        opacity_func = vtkPiecewiseFunction()
        opacity_func.AddPoint(0.0, 0.0)
        opacity_func.AddPoint(lower, 0.0)
        opacity_func.AddPoint(upper, opacity)
        actor.prop.SetScalarOpacity(0, opacity_func)
        
        self.rendered_actors[atype] = actor

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
        if miller_params['use_miller'] and miller_params['n'] is not None:
            distances = np.abs(
                np.dot(filtered_points - cell_center, miller_params['n']) - 
                miller_params['offset']
            )
            filtered_points = filtered_points[distances < (miller_params['thickness'] / 2)]
        
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
