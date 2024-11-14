# 3D cells segmentation example with two channels (membrane only and nuclei)

import napari
import pandas as pd
from os import chmod
from napari import Viewer
from magicgui import magicgui
from skimage.io import imsave
from skimage.filters import gaussian
from skimage.feature import blob_dog as dog
from skimage.segmentation import watershed
from skimage.measure import regionprops_table
from scipy.ndimage import zoom, minimum_filter, median_filter
from utils import *

@magicgui(call_button='Process',
          nuc_prescale={'widget_type': 'FloatSlider', 'min': 0, 'max': 1},
          nuc_scale_min={'widget_type': 'FloatSlider', 'min': 1, 'max': 5},
          nuc_scale_max={'widget_type': 'FloatSlider', 'min': 1, 'max': 9},
          nuc_det_thr={'widget_type': 'FloatSlider', 'min': 0, 'max': 1e-1},
          nuc_maxdst={'widget_type': 'IntSlider', 'min': 1, 'max': 100},
          memb_maxdelta={'widget_type': 'IntSlider', 'min': 1, 'max': 100},
          memb_gaussrad={'widget_type': 'FloatSlider', 'min': 0, 'max': 1.5},
          cell_regrad={'widget_type': 'IntSlider', 'min': 1, 'max': 9},
          cell_minvol={'widget_type': 'IntSlider', 'min': 1, 'max': 5e3},
          cell_maxvol={'widget_type': 'IntSlider', 'min': 1, 'max': 2e5})
def tissueseg3d(vw: Viewer, nuc_prescale=0.5, nuc_scale_min = 2, nuc_scale_max = 6, nuc_det_thr = 5e-2, 
                nuc_maxdst=50, memb_maxdelta=25, memb_gaussrad=0.5, cell_regrad=5, cell_minvol=2e3, cell_maxvol=1e5):

    # Retrieve data
    imagefile = load_image_tiff.imagefile.value
    zratio = load_image_tiff.zratio.value
    nuclei = vw.layers['Nuclei'].data
    membrane = vw.layers['Membrane'].data

    # Filter membrane channel
    membrane_flt = gaussian(membrane.astype(float), sigma=memb_gaussrad, preserve_range=True).astype('uint16')

    #### Detect nuclei
    # Multiscale DoG
    print('Performing nuclei detection...')
    blobs = dog(zoom(nuclei, (1, nuc_prescale, nuc_prescale), order=1), min_sigma=nuc_scale_min,
                max_sigma=nuc_scale_max, sigma_ratio=1.6, threshold=nuc_det_thr*1e-3, exclude_border=False)
    coords = [(int(blob[0]), int(blob[1]/nuc_prescale), int(blob[2]/nuc_prescale)) for blob in blobs]
    # Remove spurious seeds
    #nuclei_msk = (nuclei>=nuc_thr).astype(int)
    coords_kept = remove_seeds(membrane, coords, nuc_maxdst, memb_maxdelta, zratio)

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
    # Regularize, fill holes and optionally shrink label mask
    cell_lbl = median_filter(cell_lbl, size=(np.ceil(cell_regrad/zratio).astype(int),cell_regrad,cell_regrad))
    cell_lbl = remove_components_size(cell_lbl, cell_minvol, cell_maxvol)
    cell_lbl = remove_components_edge(cell_lbl)
    cell_lbl = fill_lbl_holes(cell_lbl)
    cell_lbl = relabel_consecutive(cell_lbl)
    properties = regionprops_table(cell_lbl, properties=['label', 'centroid', 'area', 'MajorAxisLength', 'MinorAxisLength'])
    #cell_lbl = cell_lbl * (cell_lbl == minimum_filter(cell_lbl, size=(1,3,3)))

    # Compute object surfaces from label mask
    #combined_verts, combined_faces, combined_values, custom_colormap = lbl2mesh(cell_lbl)

    #### Export results
    print('Exporting results...')
    imsave(str(imagefile).replace('.tif', '_lbl.tif'), cell_lbl, check_contrast=False)
    print(str(imagefile).replace('.tif', '_lbl.tif'))
    chmod(str(imagefile).replace('.tif', '_lbl.tif'), 0o666)
    df = pd.DataFrame(properties)
    df.columns = ['Cell', 'CZ', 'CY', 'CX', 'Volume (vox)', 'MajorAxis (pix)', 'MinorAxis (pix)']
    df.to_csv(str(imagefile).replace('.tif', '_lbl.csv'), index=False)
    print(str(imagefile).replace('.tif', '_lbl.csv'))
    chmod(str(imagefile).replace('.tif', '_lbl.csv'), 0o666)
    print(f'Number of segmented cells: {len(df)}')

    #### Display results
    print('Displaying results...')
    np.random.seed(0)
    vw.add_labels(cell_lbl, name=f"CellsLbl", blending="additive", scale=(zratio, 1, 1))
    #viewer.add_surface((combined_verts, combined_faces, combined_values), name='Combined Surface', colormap=custom_colormap, scale=(zratio, 1, 1))
    vw.add_points(coords, name=f"Seeds", size=15, face_color='black', blending="additive", scale=(zratio, 1, 1), visible=False)
    vw.add_points(coords_kept, name=f"Seeds_Kept", size=15, face_color='green', blending="additive", scale=(zratio, 1, 1), visible=False)

viewer = napari.Viewer()
dw1 = viewer.window.add_dock_widget(load_image_tiff, area='right', name='Load')
dw1.setMinimumHeight(120);dw1.setMaximumHeight(120)
dw1.setMinimumWidth(360)
dw2 = viewer.window.add_dock_widget(tissueseg3d, area='right', name='Process')
napari.run()
