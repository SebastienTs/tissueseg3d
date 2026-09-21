import numpy as np
import ctypes
from napari import Viewer
from magicgui import magicgui
from skimage.io import imread
from skimage.morphology import reconstruction
from skimage.measure import label, regionprops
from scipy.ndimage import binary_fill_holes, maximum_filter
from napari.utils.notifications import show_warning

# Default image file
imagefile_default = ''

# Create seed mask from seeds coordinates
# !! first label is used for background and seeded from the image borders !!
def seedcoords2mask(coords, shape):
    seeds = np.zeros(shape, dtype=np.uint16)
    seeds[:, 0, :] = 1
    seeds[:, :, 0] = 1
    seeds[:, -1, :] = 1
    seeds[:, :, -1] = 1
    for i, coord in enumerate(coords):
        seeds[coord] = i+2
    return seeds

# Returns distance between two 3D points
def distance_pt2pt(pt1, pt2, zratio):
    dst = np.sqrt(((np.array(pt1)*np.array((zratio,1,1))-np.array(pt2)*np.array((zratio,1,1)))**2).sum())
    return dst

# Return coordinates of points making up a line segment between two 3D points
def interpolate_3d_line(start, end):
    start = np.array(start)
    end = np.array(end)
    vector = end - start
    num_steps = int(np.ceil(np.linalg.norm(vector)))
    t = np.linspace(0, 1, num_steps)
    points = start[np.newaxis, :] + t[:, np.newaxis] * vector[np.newaxis, :]
    return np.round(points).astype(int)

# Remove seed pairs within distance dstthr which intensity line profile range is below deltamin
def remove_seeds(img, seeds, dstthr, deltamin, zratio):
    mergelst = [[] for _ in range(len(seeds))]
    for i, seed1 in enumerate(seeds):
        for j, seed2 in enumerate(seeds[i+1:], start=i+1):
                if  distance_pt2pt(seed1, seed2, zratio) < dstthr:
                    profile = np.array([img[tuple(point)] for point in interpolate_3d_line(seed1, seed2)])
                    delta = profile.max() - profile.min()
                    if delta < deltamin:
                        mergelst[i].append(j)

    # Index of the seeds to keep (all but the ones that are part of a cluster)
    idx = set(range(1, len(seeds))) - set(sum(mergelst, []))

    # Move seeds at their associated clusters' centers of mass
    seeds = [tuple(np.round(np.mean(np.array(seeds)[[i]+lst], axis=0)).astype(int)) for i, lst in enumerate(mergelst)]

    return [seeds[i] for i in idx]

# Impose regional intensity minima at nonnull mask locations
def imposemin(img, mask):
    marker = np.full(img.shape, np.inf)
    marker[mask == 1] = 0
    mask = np.minimum((img + 1), marker)
    return reconstruction(marker, mask, method='erosion')

# Impose regional intensity maxima at nonnull mask locations
def imposemax(img, mask):
    marker = np.full(img.shape, -np.inf)
    marker[mask == 1] = 0
    mask = np.maximum((img + 1), marker)
    return reconstruction(marker, mask, method='dilation')

# Remove objects outside volume range
def remove_lbl_size(lbl, minvol, maxvol):
    labels, counts = np.unique(lbl, return_counts=True)
    mask = (counts >= minvol) & (counts <= maxvol)
    mapping = np.zeros(labels.max() + 1, dtype=lbl.dtype)
    mapping[labels[mask]] = labels[mask]
    return mapping[lbl]

# Remove objects touching image borders
def remove_lbl_edge(lbl):
    regions = regionprops(lbl)
    for region in regions:
        bbox = region.bbox
        if bbox[1] <= 1 or bbox[4] >= (lbl.shape[1] - 2) or bbox[2] <= 1 or bbox[5] >= (lbl.shape[2] - 2):
            x, y, z = zip(*region.coords)
            lbl[x, y, z] = 0
    return lbl

# Fill label mask holes
def fill_lbl_holes(lbl):
    lbl_holes = label(binary_fill_holes(lbl>0) ^ (lbl>0))
    regions = regionprops(lbl_holes, intensity_image=maximum_filter(lbl, size=(1,3,3)))
    for region in regions:
        reglbl = region.intensity_max
        x, y, z = zip(*region.coords)
        lbl[x, y, z] = reglbl
    return lbl

# Relabel label mask with integers from 1 to N (no gap)
def relabel(lbl):
    unique_labels, inverse = np.unique(lbl, return_inverse=True)
    new_labels = np.arange(len(unique_labels))
    new_labels[unique_labels == 0] = 0
    return new_labels[inverse].reshape(lbl.shape)

# Check if viewer layer with specific name exists
def viewer_is_layer(vw: Viewer, layername):
    found = False
    if len(vw.layers) > 0:
        for i, ly in enumerate(vw.layers):
            if str(ly) == layername: found = True
    return found

# Display message dialog box
def dialogboxmes(message, title):
    return ctypes.windll.user32.MessageBoxW(0, title, message, 0)

# Image loader widget
@magicgui(call_button='Load', imagefile={'widget_type': 'FileEdit', 'label': 'Image'},
          zratio={'widget_type': 'FloatSlider', 'min': 1, 'max': 9}, dualchan={'widget_type': 'CheckBox', 'label': 'Two channels (nucleus+membrane)'})
def load_image_tiff(vw:Viewer, imagefile=imagefile_default, zratio=6.92, dualchan=True):

    # Close all layers
    vw.layers.clear()

    # Load image, split channels and display in viewer
    if imagefile.is_file() and str(imagefile).endswith('.tif'):

        img = imread(imagefile).astype(np.uint16)

        if dualchan:
            nuclei = img[:, 0, :, :]
            membrane = img[:, 1, :, :]
            vw.add_image(nuclei, name=f"Nuclei", scale=(zratio, 1, 1))
        else:
            membrane = img
        vw.add_image(membrane, name=f"Membrane", scale=(zratio, 1, 1), blending='additive', colormap='green')
        if not dualchan:
            vw.add_points([], name=f"Seeds_Kept", size=15, face_color='green', blending="additive", scale=(zratio, 1, 1))
    else:

        show_warning('Select a valid TIFF file!')

    return None

@magicgui(call_button='Remove Label', label={'widget_type': 'SpinBox', 'min': 0})
def remove_label(vw: Viewer, label):

    if viewer_is_layer(vw, 'CellsLbl'):
        lbl = vw.layers['CellsLbl'].data
        lbl[lbl==label] = 0
        vw.layers['CellsLbl'].data = lbl
        if viewer_is_layer(vw, 'NucleiLbl'):
            lbl = vw.layers['NucleiLbl'].data
            lbl[lbl == label] = 0
            vw.layers['NucleiLbl'].data = lbl
    else:
        print('!! No "CellsLbl" layer found !!')

    return None


@magicgui(call_button='Merge Labels', label1={'widget_type': 'SpinBox', 'min': 1}, label2={'widget_type': 'SpinBox', 'min': 1})
def merge_labels(vw: Viewer, label1, label2):

    if viewer_is_layer(vw, 'CellsLbl'):
        lbl = vw.layers['CellsLbl'].data
        lbl[lbl==label2] = label1
        vw.layers['CellsLbl'].data = lbl
        if viewer_is_layer(vw, 'NucleiLbl'):
            lbl = vw.layers['NucleiLbl'].data
            lbl[lbl == label2] = label1
            vw.layers['NucleiLbl'].data = lbl
    else:
        print('!! No "CellsLbl" layer found !!')

    return None
