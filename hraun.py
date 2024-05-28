import sys
import numpy as np
import pyvista as pv
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QSplitter, QLabel, QCheckBox, QLineEdit, QPushButton, QComboBox
from PyQt5.QtCore import Qt
from skimage import measure, filters
from skimage.filters import gaussian
from skimage.measure import block_reduce
import matplotlib.pyplot as plt
from volman import VolMan
from pyvistaqt import QtInteractor
from skimage.exposure import equalize_adapthist
from skimage.measure import label
import vtk

def rescale_array(arr):
    min_val = arr.min()
    max_val = arr.max()
    rescaled_arr = (arr - min_val) /  (max_val - min_val)
    return rescaled_arr

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        self.setWindowTitle("Voxel Data Visualization")

        self.splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(self.splitter)

        self.control_panel = QWidget()
        self.control_layout = QVBoxLayout()
        self.control_panel.setLayout(self.control_layout)
        self.splitter.addWidget(self.control_panel)

        self.plotter = QtInteractor(self.splitter)
        self.splitter.addWidget(self.plotter.interactor)

        self.splitter.setStretchFactor(1, 1)

        self.init_params_ui()
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()
        self.voxel_data = None
        self.mesh = None
        self.orig_colors = None
        self.delete_mask = None
        self.selected_faces = []

    def keyPressEvent(self, event):
        pass

    def init_params_ui(self):
        # Parameters for voxel data
        self.label_vol = QLabel("Volume ID:")
        self.control_layout.addWidget(self.label_vol)

        self.vol_id = QLineEdit("PHerc1667")
        self.vol_id.setFixedHeight(30)
        self.control_layout.addWidget(self.vol_id)

        self.label_type = QLabel("Type:")
        self.control_layout.addWidget(self.label_type)

        self.vol_type = QLineEdit("volumes")
        self.vol_type.setFixedHeight(30)
        self.control_layout.addWidget(self.vol_type)

        self.label_timestamp = QLabel("Timestamp:")
        self.control_layout.addWidget(self.label_timestamp)

        self.vol_timestamp = QLineEdit("20231107190228")
        self.vol_timestamp.setFixedHeight(30)
        self.control_layout.addWidget(self.vol_timestamp)

        self.label_dim = QLabel("Dimensions (z,y,c):")
        self.control_layout.addWidget(self.label_dim)

        self.dim_x = QLineEdit("4000")
        self.dim_x.setFixedHeight(30)
        self.control_layout.addWidget(self.dim_x)

        self.dim_y = QLineEdit("4000")
        self.dim_y.setFixedHeight(30)
        self.control_layout.addWidget(self.dim_y)

        self.dim_z = QLineEdit("4000")
        self.dim_z.setFixedHeight(30)
        self.control_layout.addWidget(self.dim_z)

        self.label_chunk = QLabel("Chunk size (x,y,z):")
        self.control_layout.addWidget(self.label_chunk)

        self.chunk_x = QLineEdit("128")
        self.chunk_x.setFixedHeight(30)
        self.control_layout.addWidget(self.chunk_x)

        self.chunk_y = QLineEdit("128")
        self.chunk_y.setFixedHeight(30)
        self.control_layout.addWidget(self.chunk_y)

        self.chunk_z = QLineEdit("128")
        self.chunk_z.setFixedHeight(30)
        self.control_layout.addWidget(self.chunk_z)

        self.label_block_size = QLabel("Reduce Block size multiplier:")
        self.control_layout.addWidget(self.label_block_size)

        self.block_reduce = QLineEdit("1")
        self.block_reduce.setFixedHeight(30)
        self.control_layout.addWidget(self.block_reduce)

        self.load_button = QPushButton("Load Data")
        self.load_button.setFixedHeight(30)
        self.load_button.clicked.connect(self.load_voxel_data)
        self.control_layout.addWidget(self.load_button)

        self.label_isolevel = QLabel("Isolevel:")
        self.control_layout.addWidget(self.label_isolevel)

        self.isolevel_input = QLineEdit("100")
        self.isolevel_input.setFixedHeight(30)
        self.control_layout.addWidget(self.isolevel_input)

        self.label_side = QLabel("Select Side:")
        self.control_layout.addWidget(self.label_side)

        self.side_selector = QComboBox()
        self.side_selector.addItems(["Verso", "Recto"])
        self.side_selector.setFixedHeight(30)
        self.control_layout.addWidget(self.side_selector)

        self.label_segment = QLabel("Segment Number:")
        self.control_layout.addWidget(self.label_segment)

        self.segment_selector = QComboBox()
        self.segment_selector.addItems([str(i) for i in range(10)])
        self.segment_selector.setFixedHeight(30)
        self.control_layout.addWidget(self.segment_selector)

        self.cell_picking_mode_checkbox = QCheckBox("Through Cell Picking Mode")
        self.cell_picking_mode_checkbox.setChecked(False)
        self.cell_picking_mode_checkbox.stateChanged.connect(self.toggle_cell_picking_mode)
        self.control_layout.addWidget(self.cell_picking_mode_checkbox)

        self.save_button = QPushButton("Save Selected Faces")
        self.save_button.setFixedHeight(30)
        self.save_button.clicked.connect(self.save_selected_faces)
        self.control_layout.addWidget(self.save_button)

        self.unselect_button = QPushButton("Unselect All Faces")
        self.unselect_button.setFixedHeight(30)
        self.unselect_button.clicked.connect(self.unselect_all_faces)
        self.control_layout.addWidget(self.unselect_button)

        self.delete_button = QPushButton("Delete Selected Points")
        self.delete_button.setFixedHeight(30)
        self.delete_button.clicked.connect(self.delete_selected_points)
        self.control_layout.addWidget(self.delete_button)

        self.bbox_button = QPushButton("Create Bounding Box")
        self.bbox_button.setFixedHeight(30)
        self.bbox_button.clicked.connect(self.create_closed_polygon)
        self.control_layout.addWidget(self.bbox_button)

    def create_closed_polygon(self):
        if not self.selected_faces:
            return

        selected_points = np.array([coord for face in self.selected_faces for coord in face['coordinates']])
        polygon = pv.PolyData(selected_points)
        convex_hull = polygon.delaunay_3d()
        self.plotter.add_mesh(convex_hull, color='blue', opacity=0.5)
        self.plotter.show()

    def convert_to_voxel_space(self, coord):
        x, y, z = coord
        voxel_x = int(
            (x - self.mesh.bounds[0]) / (self.mesh.bounds[1] - self.mesh.bounds[0]) * self.voxel_data.shape[0])
        voxel_y = int(
            (y - self.mesh.bounds[2]) / (self.mesh.bounds[3] - self.mesh.bounds[2]) * self.voxel_data.shape[1])
        voxel_z = int(
            (z - self.mesh.bounds[4]) / (self.mesh.bounds[5] - self.mesh.bounds[4]) * self.voxel_data.shape[2])
        return voxel_x, voxel_y, voxel_z

    def toggle_cell_picking_mode(self, state):
        self.plotter.enable_cell_picking(callback=self.pick_callback, show_message=True, color='red', point_size=10,
                                             through=state == Qt.Checked)

    def unselect_all_faces(self):
        print("unselecting not totally working yet")
        if self.mesh is not None:
            print('asdf')
            self.plotter.update()

        # Clear the selected faces list
        self.selected_faces.clear()
        print("All faces unselected.")

    def load_voxel_data(self):
        vol_id = self.vol_id.text()
        vol_type = self.vol_type.text()
        vol_timestamp = self.vol_timestamp.text()
        dim_x = int(self.dim_x.text())
        dim_y = int(self.dim_y.text())
        dim_z = int(self.dim_z.text())
        chunk_x = int(self.chunk_x.text())
        chunk_y = int(self.chunk_y.text())
        chunk_z = int(self.chunk_z.text())
        reduce = int(self.block_reduce.text())
        print("Chunking")
        volman = VolMan('D:/vesuvius.volman')
        self.voxel_data = volman.chunk(vol_id, vol_type, vol_timestamp, [dim_x, dim_y, dim_z],
                                       [chunk_x, chunk_y, chunk_z])

        if self.delete_mask is not None:
            self.voxel_data[self.delete_mask] = 0

        isolevel = int(self.isolevel_input.text())

        labels = label(self.voxel_data > isolevel)
        component_sizes = np.bincount(labels.flatten())
        mask = np.isin(labels, np.where(component_sizes >= 32)[0])
        self.voxel_data = self.voxel_data * mask
        print("Voxel data shape after reduction:", self.voxel_data.shape)

        print(f"Using isolevel: {isolevel}")
        verts, faces, normals, values = measure.marching_cubes(self.voxel_data, level=isolevel, allow_degenerate=False,
                                                               step_size=reduce)
        values = rescale_array(values)
        values = equalize_adapthist(values)
        print("Marching cubes completed successfully.")

        faces = np.hstack([[3] + list(face) for face in faces])
        self.mesh = pv.PolyData(verts, faces)
        print("PyVista mesh created")

        self.mesh["values"] = values
        print('Colorizing')
        cmap = plt.get_cmap("viridis")
        colors = cmap(values / values.max())
        colors = (colors[:, :3] * 255).astype(np.uint8)
        self.mesh.point_data["colors"] = colors
        print('Adding mesh')
        self.plotter.clear()
        self.plotter.add_mesh(self.mesh, scalars="colors", rgb=True, show_scalar_bar=True, opacity="linear",
                              render_points_as_spheres=False, lighting=True, ambient=0.8, specular=0.6)

        light = pv.Light(position=(0, 0, 10), focal_point=(0, 0, 0), color='white', intensity=0.8)
        self.plotter.add_light(light)

        through_mode = self.cell_picking_mode_checkbox.isChecked()
        self.plotter.enable_cell_picking(callback=self.pick_callback, show_message=True, color='red', line_width=10,
                                         through=through_mode)
        print("Showing mesh")
        self.plotter.show()

    def pick_callback(self, picked_cells):
        if picked_cells is None:
            return
        if 'vtkOriginalCellIds' in picked_cells.cell_data.keys():
            cell_ids = picked_cells.cell_data['vtkOriginalCellIds']
            print()
        else:
            cell_ids = picked_cells.cell_data['orig_extract_id']

        for cell_id in cell_ids:
            cell = self.mesh.get_cell(cell_id)
            point_ids = cell.point_ids
            self.selected_faces.append([self.mesh.points[pid] for pid in point_ids])

        self.plotter.update()

    def delete_selected_points(self):
        if not self.selected_faces:
            print("No faces selected for deletion.")
            return
        if not self.cell_picking_mode_checkbox.isChecked():
            raise ValueError("point deletion only supported for through cell picking mode")
        selected_points = np.array([coord for face in self.selected_faces for coord in face])
        min_coords = np.min(selected_points, axis=0)
        max_coords = np.max(selected_points, axis=0)

        delete_mask = np.zeros_like(self.voxel_data, dtype=bool)

        for i in range(self.voxel_data.shape[0]):
            for j in range(self.voxel_data.shape[1]):
                for k in range(self.voxel_data.shape[2]):
                    voxel_coord = (i, j, k)
                    world_coord = self.mesh.bounds[0] + i * (self.mesh.bounds[1] - self.mesh.bounds[0]) / \
                                  self.voxel_data.shape[0], \
                                  self.mesh.bounds[2] + j * (self.mesh.bounds[3] - self.mesh.bounds[2]) / \
                                  self.voxel_data.shape[1], \
                                  self.mesh.bounds[4] + k * (self.mesh.bounds[5] - self.mesh.bounds[4]) / \
                                  self.voxel_data.shape[2]

                    if np.all(min_coords <= world_coord) and np.all(world_coord <= max_coords):
                        delete_mask[voxel_coord] = True

        self.delete_mask = delete_mask
        self.selected_faces.clear()
        self.load_voxel_data()

    def save_selected_faces(self):
        side = self.side_selector.currentText()
        segment = self.segment_selector.currentText()
        with open("selected_faces.txt", "w") as f:
            for face in self.selected_faces:
                f.write(f"Side: {side}\n")
                f.write(f"Segment: {segment}\n")
                f.write(f"Coordinates: {face}\n\n")
        print("Selected faces saved to selected_faces.txt")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())