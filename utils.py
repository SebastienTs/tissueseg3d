import numpy as np
from skimage.morphology import reconstruction
from skimage.measure import label, regionprops
from scipy.ndimage import binary_fill_holes, maximum_filter
from qtpy.QtWidgets import QTableView
from qtpy.QtCore import QAbstractTableModel, Qt

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
