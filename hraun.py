import os
import io
import numpy as np
import tifffile
from skimage import measure, exposure
from matplotlib import cm
from skimage.restoration import denoise_tv_chambolle
from skimage.filters import gaussian
from skimage.measure import block_reduce
from scipy.ndimage import uniform_filter


#https://github.com/awangenh/fastaniso
def anisodiff3(stack,niter=1,kappa=50,gamma=0.1,step=(1.,1.,1.),option=1,ploton=False):
    """
    3D Anisotropic diffusion.

    Usage:
    stackout = anisodiff(stack, niter, kappa, gamma, option)

    Arguments:
            stack  - input stack
            niter  - number of iterations
            kappa  - conduction coefficient 20-100 ?
            gamma  - max value of .25 for stability
            step   - tuple, the distance between adjacent pixels in (z,y,x)
            option - 1 Perona Malik diffusion equation No 1
                     2 Perona Malik diffusion equation No 2
            ploton - if True, the middle z-plane will be plotted on every
                 iteration

    Returns:
            stackout   - diffused stack.

    kappa controls conduction as a function of gradient.  If kappa is low
    small intensity gradients are able to block conduction and hence diffusion
    across step edges.  A large value reduces the influence of intensity
    gradients on conduction.

    gamma controls speed of diffusion (you usually want it at a maximum of
    0.25)

    step is used to scale the gradients in case the spacing between adjacent
    pixels differs in the x,y and/or z axes

    Diffusion equation 1 favours high contrast edges over low contrast ones.
    Diffusion equation 2 favours wide regions over smaller ones.

    Reference:
    P. Perona and J. Malik.
    Scale-space and edge detection using ansotropic diffusion.
    IEEE Transactions on Pattern Analysis and Machine Intelligence,
    12(7):629-639, July 1990.

    Original MATLAB code by Peter Kovesi
    School of Computer Science & Software Engineering
    The University of Western Australia
    pk @ csse uwa edu au
    <http://www.csse.uwa.edu.au>

    Translated to Python and optimised by Alistair Muldal
    Department of Pharmacology
    University of Oxford
    <alistair.muldal@pharm.ox.ac.uk>

    June 2000  original version.
    March 2002 corrected diffusion eqn No 2.
    July 2012 translated to Python
    """

    # ...you could always diffuse each color channel independently if you
    # really want
    if stack.ndim == 4:
        #warnings.warn("Only grayscale stacks allowed, converting to 3D matrix")
        stack = stack.mean(3)

    # initialize output array
    stack = stack.astype('float32')
    stackout = stack.copy()

    # initialize some internal variables
    deltaS = np.zeros_like(stackout)
    deltaE = deltaS.copy()
    deltaD = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    UD = deltaS.copy()
    gS = np.ones_like(stackout)
    gE = gS.copy()
    gD = gS.copy()

    # create the plot figure, if requested
    if ploton:
        import pylab as pl
        from time import sleep

        showplane = stack.shape[0]//2

        fig = pl.figure(figsize=(20,5.5),num="Anisotropic diffusion")
        ax1,ax2 = fig.add_subplot(1,2,1),fig.add_subplot(1,2,2)

        ax1.imshow(stack[showplane,...].squeeze(),interpolation='nearest')
        ih = ax2.imshow(stackout[showplane,...].squeeze(),interpolation='nearest',animated=True)
        ax1.set_title("Original stack (Z = %i)" %showplane)
        ax2.set_title("Iteration 0")

        fig.canvas.draw()

    for ii in range(niter):

        # calculate the diffs
        deltaD[:-1,: ,:  ] = np.diff(stackout,axis=0)
        deltaS[:  ,:-1,: ] = np.diff(stackout,axis=1)
        deltaE[:  ,: ,:-1] = np.diff(stackout,axis=2)

        # conduction gradients (only need to compute one per dim!)
        if option == 1:
            gD = np.exp(-(deltaD/kappa)**2.)/step[0]
            gS = np.exp(-(deltaS/kappa)**2.)/step[1]
            gE = np.exp(-(deltaE/kappa)**2.)/step[2]
        elif option == 2:
            gD = 1./(1.+(deltaD/kappa)**2.)/step[0]
            gS = 1./(1.+(deltaS/kappa)**2.)/step[1]
            gE = 1./(1.+(deltaE/kappa)**2.)/step[2]

        # update matrices
        D = gD*deltaD
        E = gE*deltaE
        S = gS*deltaS

        # subtract a copy that has been shifted 'Up/North/West' by one
        # pixel. don't as questions. just do it. trust me.
        UD[:] = D
        NS[:] = S
        EW[:] = E
        UD[1:,: ,: ] -= D[:-1,:  ,:  ]
        NS[: ,1:,: ] -= S[:  ,:-1,:  ]
        EW[: ,: ,1:] -= E[:  ,:  ,:-1]

        # update the image
        stackout += gamma*(UD+NS+EW)

        if ploton:
            iterstring = "Iteration %i" %(ii+1)
            ih.set_data(stackout[showplane,...].squeeze())
            ax2.set_title(iterstring)
            fig.canvas.draw()
            # sleep(0.01)

    return stackout

def avg_pool_3d(arr, pool_size):
    return uniform_filter(arr, size=pool_size, mode='nearest')
def load_cropped_tiff_slices(tiff_directory, start_slice, end_slice, crop_start, crop_end):
    slices_data = []
    for slice_index in range(start_slice, end_slice):
        tiff_filename = f"{slice_index:02d}.tif"
        print(f"loading {slice_index}")
        tiff_path = os.path.join(tiff_directory, tiff_filename)
        tiff_data = tifffile.memmap(tiff_path)
        slices_data.append(tiff_data[crop_start[0]:crop_end[0], crop_start[1]:crop_end[1]])
    return slices_data

def process_chunk(tiff_directory, chunk_size, chunk_offset):
    start_slice = chunk_offset[2]
    end_slice = start_slice + chunk_size[2]

    crop_start = (chunk_offset[0], chunk_offset[1])
    crop_end = (chunk_offset[0] + chunk_size[0], chunk_offset[1] + chunk_size[1])

    slices_data = load_cropped_tiff_slices(tiff_directory, start_slice, end_slice, crop_start, crop_end)

    # Convert slices_data to numpy array
    combined_chunk = np.stack(slices_data, axis=-1)

    # Perform data type scaling and conversion
    if combined_chunk.dtype == np.uint16:
        combined_chunk //= 256
        combined_chunk = combined_chunk.astype(np.uint8)
    elif combined_chunk.dtype == np.uint8:
        pass
    else:
        raise ValueError("invalid input dtype from tiff files")

    print("preprocessing")
    combined_chunk &= 0xff
    combined_chunk = combined_chunk.astype(np.float32)
    #combined_chunk = gaussian(combined_chunk)
    #combined_chunk = exposure.equalize_adapthist(combined_chunk)
    #combined_chunk = denoise_tv_chambolle(combined_chunk,weight=1)
    #combined_chunk = (combined_chunk - combined_chunk.min()) / (combined_chunk.max() - combined_chunk.min())
    #combined_chunk = block_reduce(combined_chunk,block_size=(8,8,1),func=np.sum)
    #combined_chunk = avg_pool_3d(combined_chunk,(4,4,4))

    combined_chunk = (combined_chunk - combined_chunk.min()) / (combined_chunk.max() - combined_chunk.min())
    #combined_chunk = anisodiff3(combined_chunk)

    p2, p98 = np.percentile(combined_chunk, (2, 98))
    combined_chunk = exposure.rescale_intensity(combined_chunk, in_range=(p2, p98))
    print("marching cubes")
    verts, faces, normals, values = measure.marching_cubes(combined_chunk, level=.5, allow_degenerate=False)
    print("normalizing values")
    values = (values - values.min()) / (values.max() - values.min())
    #values = exposure.equalize_adapthist(values)
    #values = denoise_tv_chambolle(values)
    #p2, p98 = np.percentile(values, (10, 90))
    #values = exposure.rescale_intensity(values, in_range=(p2, p98))
    #mask = values > .3
    #values[mask] = 1.0
    #mask = values < .3
    #values[mask] = 0.0
    print("colorizing")
    colors = cm.get_cmap('viridis')(values)
    colors = (colors*256).astype(np.uint8)

    print("writing to file")
    ply_filename = f"chunk_{chunk_offset[0]}_{chunk_offset[1]}_{chunk_offset[2]}_pool_{pool_size[0]}_{pool_size[1]}_{pool_size[2]}.ply"
    ply_path = os.path.join(output_directory, ply_filename)
    num_verts = len(verts)
    num_faces = len(faces)
    with open(ply_path, 'w',buffering=1024*1024*128) as ply_file:
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write(f"element vertex {num_verts}\n")
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("property uchar red\n")
        ply_file.write("property uchar green\n")
        ply_file.write("property uchar blue\n")
        ply_file.write(f"element face {num_faces}\n")
        ply_file.write("property list uchar int vertex_index\n")
        ply_file.write("end_header\n")

        chunk_size = 1000000
        for i in range(0, num_verts, chunk_size):
            chunk_verts = verts[i:i + chunk_size]
            chunk_colors = colors[i:i + chunk_size]
            vert_lines = [f"{v[0]} {v[1]} {v[2]} {c[0]} {c[1]} {c[2]}\n" for v, c in zip(chunk_verts, chunk_colors)]
            ply_file.write(''.join(vert_lines))

        for i in range(0, num_faces, chunk_size):
            chunk_faces = faces[i:i + chunk_size]
            face_lines = [f"3 {f[0]} {f[1]} {f[2]}\n" for f in chunk_faces]
            ply_file.write(''.join(face_lines))


def convert_to_8bit():
    import os
    import tifffile
    from skimage import exposure
    from skimage.restoration import denoise_tv_chambolle

    def process_tiff_stack(tiff_directory):
        # Get a list of TIFF files in the directory
        tiff_files = [f for f in os.listdir(tiff_directory) if f.endswith('.tif')]

        for tiff_file in tiff_files:
            tiff_path = os.path.join(tiff_directory, tiff_file)

            # Read the 16-bit unsigned TIFF image
            tiff_data = tifffile.imread(tiff_path)
            if tiff_data.dtype == np.uint8:
                continue
            tiff_data //=256
            tiff_data = tiff_data.astype(np.uint8)
            tifffile.imwrite(tiff_path, tiff_data)

            print(f"Processed: {tiff_file}")

    # Specify the directory containing the TIFF stack
    tiff_directory = r'C:\Users\forrest\dev\Hraun\dl.ash2txt.org\full-scrolls\PHerc1667.volpkg\volumes\20231117161658'

    # Process the TIFF stack
    process_tiff_stack(tiff_directory)

if __name__ == '__main__':
    tiff_directory = r"C:\Users\forrest\dev\Hraun\dl.ash2txt.org\full-scrolls\Scroll1.volpkg\paths\20231022170901\layers"
    chunk_size = (1000, 1000, 65)
    chunk_offset = (1000, 1000, 0)
    pool_size = (1, 1, 1)

    output_directory = "generated_ply"
    os.makedirs(output_directory, exist_ok=True)

    process_chunk(tiff_directory, chunk_size, chunk_offset)
    print("Processing completed.")