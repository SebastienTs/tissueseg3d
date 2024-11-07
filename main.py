# 3D cells segmentation example with two channels (membrane only and nuclei)

import napari
from os import chmod
from napari import Viewer
from magicgui import magicgui
from skimage.io import imread, imsave
from skimage.filters import gaussian
from skimage.feature import blob_dog as dog
from skimage.segmentation import watershed
from skimage.measure import regionprops_table
from scipy.ndimage import zoom, minimum_filter, median_filter
from utils import *
import pandas as pd

# Image file from Airy scan (0.25 XY downscaled)
imagefile_default = 'D:/Projects/UPF/Berta_Lucas/CAAXinjH2B 12 hpf_025.tif'

@magicgui(call_button='Run',
          imagefile={'widget_type': 'FileEdit', 'label': 'Image Stack'},
          nuc_prescale={'widget_type': 'FloatSlider', 'min': 0, 'max': 1},
          nuc_scale_min={'widget_type': 'IntSlider', 'min': 1, 'max': 9},
          nuc_scale_max={'widget_type': 'IntSlider', 'min': 1, 'max': 9},
          nuc_det={'widget_type': 'FloatSlider', 'min': 0, 'max': 1e-1},
          memb_gaussrad={'widget_type': 'FloatSlider', 'min': 0, 'max': 1.5},
          memb_maxdelta={'widget_type': 'IntSlider', 'min': 1, 'max': 100},
          cell_maxdst={'widget_type': 'IntSlider', 'min': 1, 'max': 100},
          cell_regrad={'widget_type': 'IntSlider', 'min': 1, 'max': 9},
          cell_minvol={'widget_type': 'IntSlider', 'min': 1, 'max': 1e3},
          cell_maxvol={'widget_type': 'IntSlider', 'min': 1, 'max': 2e5},
          zratio={'widget_type': 'FloatSlider', 'min': 0, 'max': 9})
def tissueseg3d(vw: Viewer, imagefile=imagefile_default, nuc_prescale=0.5, nuc_scale_min = 2, nuc_scale_max = 6,
                nuc_det = 5e-2, memb_gaussrad=0.5, memb_maxdelta=25, cell_maxdst=50, cell_regrad=5, cell_minvol=5e2,
                cell_maxvol=1e5, zratio = 6):

    # Close all layers
    vw.layers.clear()

    # Load image
    print('Loading image stack and preprocessing...')
    stack = imread(imagefile)

    # Split channels
    nuclei = stack[:, 0, :, :]
    membrane = stack[:, 1, :, :]
    membrane_flt = gaussian(membrane.astype(float), sigma=memb_gaussrad, preserve_range=True).astype('uint16')

    #### Detect nuclei
    # Multiscale DoG
    print('Performing nuclei detection...')
    blobs = dog(zoom(nuclei, (1, nuc_prescale, nuc_prescale), order=1), min_sigma=nuc_scale_min,
                max_sigma=nuc_scale_max, sigma_ratio=1.6, threshold=nuc_det*1e-3, exclude_border=True)
    coords = [(int(blob[0]), int(blob[1]/nuc_prescale), int(blob[2]/nuc_prescale)) for blob in blobs]
    # Remove spurious seeds
    #nuclei_msk = (nuclei>=nuc_thr).astype(int)
    coords_kept = remove_seeds(membrane_flt, coords, cell_maxdst, memb_maxdelta, zratio)

    #### Segment cells
    # Watershed
    print('Performing cells segmentation...')
    seeds = np.zeros(nuclei.shape, dtype=np.uint16)
    seeds[:, 0, :] = 1
    seeds[:, :, 0] = 1
    seeds[:, -1, :] = 1
    seeds[:, :, -1] = 1
    for i, sd in enumerate(coords_kept):
        seeds[sd] = i+2
    membrane_flt = imposemin(membrane_flt, seeds>0)
    cell_lbl = watershed(membrane_flt, seeds, compactness=0)
    # Remove small and large regions
    print('Postprocessing...')
    cell_lbl = remove_components_size(cell_lbl, cell_minvol, cell_maxvol)
    cell_lbl = remove_components_edge(cell_lbl)
    # Regularize, fill holes and optionally shrink label mask
    cell_lbl = median_filter(cell_lbl, size=(np.ceil(cell_regrad/zratio).astype(int),cell_regrad,cell_regrad))
    cell_lbl = fill_lbl_holes(cell_lbl)
    # Split cell regions
    #cell_lbl = cell_lbl * (cell_lbl == minimum_filter(cell_lbl, size=(1,3,3)))

    #### Export results
    print('Exporting results...')
    imsave(str(imagefile).replace('.tif', '_lbl.tif'), cell_lbl)
    chmod(str(imagefile).replace('.tif', '_lbl.tif'), 0o666)
    properties = regionprops_table(cell_lbl, properties=['label', 'centroid', 'area', 'MajorAxisLength', 'MinorAxisLength'])
    df = pd.DataFrame(properties)
    df.to_csv(str(imagefile).replace('.tif', '_lbl.csv'), index=False)
    chmod(str(imagefile).replace('.tif', '_lbl.csv'), 0o666)
    print(f'Number of segmented cells: {len(df)}')

    #### Display results
    print('Displaying results...')
    np.random.seed(0)
    vw.add_image(nuclei, name=f"Nuclei", scale=(zratio, 1, 1))
    vw.add_image(membrane, name=f"Membrane", scale=(zratio, 1, 1), blending='additive', colormap='green')
    vw.add_labels(cell_lbl, name=f"CellsLbl", blending="additive", scale=(zratio, 1, 1))
    vw.add_points(coords, name=f"Seeds", size=15, face_color='black', blending="additive", scale=(zratio, 1, 1), visible=False)
    vw.add_points(coords_kept, name=f"Seeds_Kept", size=15, face_color='green', blending="additive", scale=(zratio, 1, 1), visible=False)

viewer = napari.Viewer()
dw = viewer.window.add_dock_widget(tissueseg3d, area='right', name='Process')
dw.setMinimumWidth(360)
napari.run()
