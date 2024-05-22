import sys
import numpy as np
import pyvista as pv
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QSplitter, QLabel, QLineEdit, QPushButton, QComboBox
from PyQt5.QtCore import Qt
from skimage import measure
from skimage.measure import block_reduce
import matplotlib.pyplot as plt
from volman import VolMan
from pyvistaqt import QtInteractor
from skimage.exposure import equalize_adapthist

def rescale_array(arr):
    min_val = arr.min()
    max_val = arr.max()
    rescaled_arr = (arr - min_val) / (max_val - min_val)
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

        self.voxel_data = None
        self.mesh = None
        self.colors = None
        self.selected_faces = []

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

        self.chunk_x = QLineEdit("256")
        self.chunk_x.setFixedHeight(30)
        self.control_layout.addWidget(self.chunk_x)

        self.chunk_y = QLineEdit("256")
        self.chunk_y.setFixedHeight(30)
        self.control_layout.addWidget(self.chunk_y)

        self.chunk_z = QLineEdit("256")
        self.chunk_z.setFixedHeight(30)
        self.control_layout.addWidget(self.chunk_z)

        self.label_block_size = QLabel("Reduce Block size (x,y,z):")
        self.control_layout.addWidget(self.label_block_size)

        self.block_x = QLineEdit("1")
        self.block_x.setFixedHeight(30)
        self.control_layout.addWidget(self.block_x)

        self.block_y = QLineEdit("1")
        self.block_y.setFixedHeight(30)
        self.control_layout.addWidget(self.block_y)

        self.block_z = QLineEdit("1")
        self.block_z.setFixedHeight(30)
        self.control_layout.addWidget(self.block_z)

        self.load_button = QPushButton("Load Data")
        self.load_button.setFixedHeight(30)
        self.load_button.clicked.connect(self.load_voxel_data)
        self.control_layout.addWidget(self.load_button)

        # Isolevel input box
        self.label_isolevel = QLabel("Isolevel:")
        self.control_layout.addWidget(self.label_isolevel)

        self.isolevel_input = QLineEdit("100")
        self.isolevel_input.setFixedHeight(30)
        self.control_layout.addWidget(self.isolevel_input)

        # Verso/Recto selector
        self.label_side = QLabel("Select Side:")
        self.control_layout.addWidget(self.label_side)

        self.side_selector = QComboBox()
        self.side_selector.addItems(["Verso", "Recto"])
        self.side_selector.setFixedHeight(30)
        self.control_layout.addWidget(self.side_selector)

        # Segment number input
        self.label_segment = QLabel("Segment Number:")
        self.control_layout.addWidget(self.label_segment)

        self.segment_input = QLineEdit()
        self.segment_input.setFixedHeight(30)
        self.control_layout.addWidget(self.segment_input)

        # Save button
        self.save_button = QPushButton("Save Selected Faces")
        self.save_button.setFixedHeight(30)
        self.save_button.clicked.connect(self.save_selected_faces)
        self.control_layout.addWidget(self.save_button)

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
        block_x = int(self.block_x.text())
        block_y = int(self.block_y.text())
        block_z = int(self.block_z.text())

        print("Chunking")
        volman = VolMan('D:/vesuvius.volman')
        voxel_data = volman.chunk(vol_id, vol_type, vol_timestamp, [dim_x, dim_y, dim_z], [chunk_x, chunk_y, chunk_z])

        # Apply block reduction
        block_size = (block_x, block_y, block_z)
        self.voxel_data = block_reduce(voxel_data, func=np.mean, block_size=block_size)

        isolevel = int(self.isolevel_input.text())

        voxel_data_np = np.array(self.voxel_data)
        print("Voxel data shape:", voxel_data_np.shape)

        print(f"Using isolevel: {isolevel}")

        verts, faces, normals, values = measure.marching_cubes(voxel_data_np, level=isolevel)
        values = rescale_array(values)
        values = equalize_adapthist(values)
        print("Marching cubes completed successfully.")

        faces = np.hstack([[3] + list(face) for face in faces])
        self.mesh = pv.PolyData(verts, faces)
        print("PyVista mesh created.")

        self.mesh["values"] = values

        cmap = plt.get_cmap("viridis")
        self.colors = cmap(values / values.max())
        self.colors = (self.colors[:, :3] * 255).astype(np.uint8)
        self.mesh.point_data["colors"] = self.colors

        self.plotter.clear()
        self.plotter.add_mesh(self.mesh, scalars=self.colors, rgb=True, show_scalar_bar=False)

        # Define the callback function for face picking

        # Define the callback function for face picking
        def pick_callback(picked_cells):
            for cell_id in picked_cells['vtkOriginalCellIds']:
                print(f"Selected cell ID: {cell_id}")

                point_ids = self.mesh.get_cell(cell_id).point_ids

                for point_id in point_ids:
                    self.colors[point_id] = [255, 0, 0]

                side = self.side_selector.currentText()
                segment = self.segment_input.text()
                face_data = {
                    "cell_id": cell_id,
                    "side": side,
                    "segment": segment,
                    "coordinates": [self.mesh.points[pid] for pid in point_ids]
                }
                self.selected_faces.append(face_data)

                self.mesh.point_data["colors"] = self.colors
                self.plotter.update()

        # Enable face picking with the callback function, only selecting visible cells
        self.plotter.enable_cell_picking(callback=pick_callback, show_message=True, color='red', point_size=10, through=False)

        self.plotter.show()

    def save_selected_faces(self):
        with open("selected_faces.txt", "w") as f:
            for face in self.selected_faces:
                f.write(f"Cell ID: {face['cell_id']}\n")
                f.write(f"Side: {face['side']}\n")
                f.write(f"Segment: {face['segment']}\n")
                f.write(f"Coordinates: {face['coordinates']}\n\n")
        print("Selected faces saved to selected_faces.txt")



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
