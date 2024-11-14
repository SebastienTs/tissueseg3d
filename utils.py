import numpy as np
from skimage.morphology import reconstruction
from skimage.measure import label, regionprops, marching_cubes
from scipy.ndimage import binary_fill_holes, maximum_filter
from napari import Viewer
from skimage.io import imread
from magicgui import magicgui
from napari.utils.colormaps import Colormap

# Image file from Airy scan (0.25 XY downscaled)
imagefile_default = 'D:/Projects/UPF/Berta_Lucas/CAAXinjH2B 12 hpf_025_crop.tif'

def distance(pt1, pt2, zratio):
    dst = np.sqrt(((np.array(pt1)*np.array((zratio,1,1))-np.array(pt2)*np.array((zratio,1,1)))**2).sum())
    return dst

def interpolate_3d_line(start, end):
    start = np.array(start)
    end = np.array(end)
    vector = end - start
    num_steps = int(np.ceil(np.linalg.norm(vector)))
    t = np.linspace(0, 1, num_steps)
    points = start[np.newaxis, :] + t[:, np.newaxis] * vector[np.newaxis, :]
    return np.round(points).astype(int)

def remove_seeds(img, seeds, dstthr, deltamax, zratio):
    mergelst = [[] for _ in range(len(seeds))]
    for i, seed1 in enumerate(seeds):
        for j, seed2 in enumerate(seeds[i+1:], start=i+1):
                if  distance(seed1, seed2, zratio) < dstthr:
                    profile = np.array([img[tuple(point)] for point in interpolate_3d_line(seed1, seed2)])
                    delta = profile.max() - profile.min()
                    if delta < deltamax:
                        mergelst[i].append(j)

    # Seeds to be kept (all but the ones that are part of a cluster)
    idx = set(range(1, len(seeds))) - set(sum(mergelst, []))

    # Recenter seeds at clusters' centers of mass
    seeds = [tuple(np.round(np.mean(np.array(seeds)[[i]+lst], axis=0)).astype(int)) for i, lst in enumerate(mergelst)]

    return [seeds[i] for i in idx]

def imposemin(img, minima):
    marker = np.full(img.shape, np.inf)
    marker[minima == 1] = 0
    mask = np.minimum((img + 1), marker)
    return reconstruction(marker, mask, method='erosion')

def remove_components_size(lbl, minvol, maxvol):
    labels, counts = np.unique(lbl, return_counts=True)
    mask = (counts >= minvol) & (counts <= maxvol)
    mapping = np.zeros(labels.max() + 1, dtype=lbl.dtype)
    mapping[labels[mask]] = labels[mask]
    return mapping[lbl]

def remove_components_edge(lbl):
    regions = regionprops(lbl)
    for region in regions:
        bbox = region.bbox
        if bbox[1] <= 1 or bbox[4] >= (lbl.shape[1] - 2) or bbox[2] <= 1 or bbox[5] >= (lbl.shape[2] - 2):
            x, y, z = zip(*region.coords)
            lbl[x, y, z] = 0
    return lbl

def fill_lbl_holes(lbl):
    lbl_holes = label(binary_fill_holes(lbl>0) ^ (lbl>0))
    regions = regionprops(lbl_holes, intensity_image=maximum_filter(lbl, size=(1,3,3)))
    for region in regions:
        reglbl = region.intensity_max
        x, y, z = zip(*region.coords)
        lbl[x, y, z] = reglbl
    return lbl

def relabel_consecutive(lbl):
    # Get unique labels, excluding 0 if it's present
    unique_labels = np.unique(lbl)
    unique_labels = unique_labels[unique_labels != 0]

    # Create a mapping from old labels to new labels
    label_map = {old: new for new, old in enumerate(unique_labels, start=1)}

    # Create a vectorized function to apply the mapping
    vectorized_map = np.vectorize(lambda x: label_map.get(x, 0))

    # Apply the mapping to the entire mask
    return vectorized_map(lbl)


def lbl2mesh(lbl):
    unique_labels = np.unique(lbl)
    unique_labels = unique_labels[unique_labels != 0]
    num_labels = len(unique_labels)

    all_verts = []
    all_faces = []
    all_values = []
    face_offset = 0
    colors = [np.random.random(3) for _ in range(num_labels)]
    custom_colormap = Colormap(colors=colors, name='custom_colormap', controls=np.linspace(0, 1, num_labels))
    for i, label in enumerate(range(1, num_labels)):
        verts, faces, _, _ = marching_cubes(lbl == label)
        all_verts.append(verts)
        all_faces.append(faces + face_offset)
        color_value = i + 1
        all_values.extend([color_value] * len(verts))
        face_offset += len(verts)
    combined_verts = np.vstack(all_verts)
    combined_faces = np.vstack(all_faces)
    combined_values = np.array(all_values)
    return combined_verts, combined_faces, combined_values, custom_colormap

# Image loader widget
@magicgui(call_button='Load', imagefile={'widget_type': 'FileEdit', 'label': 'Image'},
          zratio={'widget_type': 'FloatSlider', 'min': 1, 'max': 9})
def load_image_tiff(vw:Viewer, imagefile=imagefile_default, zratio=6):

    # Close all layers
    vw.layers.clear()

    # Load image, split channels and display in viewer
    img = imread(imagefile).astype(np.uint16)
    nuclei = img[:, 0, :, :]
    membrane = img[:, 1, :, :]
    vw.add_image(nuclei, name=f"Nuclei", scale=(zratio, 1, 1))
    vw.add_image(membrane, name=f"Membrane", scale=(zratio, 1, 1), blending='additive', colormap='green')

    return None