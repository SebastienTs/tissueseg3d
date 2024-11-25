# 3D cells segmentation from membrane and nuclei channels

import napari
import pandas as pd
from os import chmod
from skimage.io import imsave
from skimage.filters import gaussian
from skimage.feature import blob_dog as dog
from skimage.segmentation import watershed
from skimage.measure import regionprops_table
from scipy.ndimage import zoom, minimum_filter, median_filter
from utils import *

@magicgui(call_button='Seed',
          nuc_prescale={'widget_type': 'FloatSlider', 'min': 0, 'max': 1},
          nuc_scale_min={'widget_type': 'FloatSlider', 'min': 1, 'max': 5},
          nuc_scale_max={'widget_type': 'FloatSlider', 'min': 1, 'max': 9},
          nuc_det_thr={'widget_type': 'FloatSlider', 'min': 0, 'max': 10},
          nuc_merge_maxdst={'widget_type': 'IntSlider', 'min': 1, 'max': 100},
          memb_maxdelta={'widget_type': 'IntSlider', 'min': 1, 'max': 200},
          memb_maxthick={'widget_type': 'IntSlider', 'min': 1, 'max': 100})
def seed_nuclei(vw: Viewer, nuc_prescale=0.5, nuc_scale_min = 2, nuc_scale_max = 6, nuc_det_thr = 1,
                nuc_merge_maxdst=50, memb_maxdelta=5, memb_maxthick=15):

    if viewer_is_layer(vw, 'Nuclei') and viewer_is_layer(vw, 'Membrane'):

        # Retrieve data from Napari objects
        zratio = load_image_tiff.zratio.value
        nuclei = vw.layers['Nuclei'].data
        membrane = vw.layers['Membrane'].data

        #### Detect nuclei
        # Multiscale DoG
        print('-------------------------------')
        print('Performing nucleus detection...')
        print('DoG detection')
        blobs = dog(zoom(nuclei, (1, nuc_prescale, nuc_prescale), order=1), min_sigma=nuc_scale_min,
                    max_sigma=nuc_scale_max, sigma_ratio=1.6, threshold=nuc_det_thr*1e-3, exclude_border=False)
        coords = [(int(blob[0]), int(blob[1]/nuc_prescale), int(blob[2]/nuc_prescale)) for blob in blobs]
        print(f"Found {len(coords)} candidate seeds")
        print('Membrane raytracing')
        coords_kept = remove_seeds(membrane, coords, nuc_merge_maxdst, memb_maxdelta, memb_maxthick, zratio)
        print(f"Kept {len(coords_kept)} seeds ({len(coords_kept)/len(coords):0.3f})")
        vw.add_points(coords, name=f"Seeds", size=15, face_color='black', blending="additive", scale=(zratio, 1, 1))
        vw.add_points(coords_kept, name=f"Seeds_Kept", size=15, face_color='green', blending="additive", scale=(zratio, 1, 1))

@magicgui(call_button='Label',
          memb_gaussrad={'widget_type': 'FloatSlider', 'min': 0, 'max': 1.5},
          cell_regrad={'widget_type': 'IntSlider', 'min': 1, 'max': 9},
          cell_minvol={'widget_type': 'IntSlider', 'min': 1, 'max': 5e3},
          cell_maxvol={'widget_type': 'IntSlider', 'min': 1, 'max': 2e5})
def segment_cells(vw: Viewer, memb_gaussrad=0.5, cell_regrad=5, cell_minvol=2e3, cell_maxvol=1e5):

    if viewer_is_layer(vw, 'Seeds_Kept') and viewer_is_layer(vw, 'Membrane'):

        # Retrieve data from Napari objects
        imagefile = load_image_tiff.imagefile.value
        zratio = load_image_tiff.zratio.value
        membrane = vw.layers['Membrane'].data
        coords_kept = [tuple(row) for row in vw.layers['Seeds_Kept'].data]
        vw.layers['Seeds'].visible = False
        vw.layers['Seeds_Kept'].visible = False

        #### Segment cells
        print('-------------------------------')
        print('Performing cell segmentation...')
        print('Filter membrane signal')
        membrane_flt = gaussian(membrane.astype(float), sigma=(memb_gaussrad, memb_gaussrad, memb_gaussrad), preserve_range=True).astype('uint16')
        seeds = np.zeros(membrane.shape, dtype=np.uint16)
        seeds[:, 0, :] = 1
        seeds[:, :, 0] = 1
        seeds[:, -1, :] = 1
        seeds[:, :, -1] = 1
        for i, sd in enumerate(coords_kept):
            seeds[sd] = i+2
        print('Imposing regional minima')
        membrane_imp = imposemin(membrane_flt, seeds>0)
        print('Watersheding')
        cell_lbl = watershed(membrane_imp, seeds, compactness=0)
        print('-------------------------------')
        print('Postprocessing...')
        # Regularize, remove small, large and border regions, fill holes, relabel and optionally shrink label mask
        print('Smoothing')
        cell_lbl = median_filter(cell_lbl, size=(1, cell_regrad, cell_regrad))
        print('Removing regions')
        cell_lbl = remove_components_size(cell_lbl, cell_minvol, cell_maxvol)
        cell_lbl = remove_components_edge(cell_lbl)
        print('Filling holes')
        cell_lbl = fill_lbl_holes(cell_lbl)
        print('Relabeling')
        cell_lbl = relabel_consecutive(cell_lbl)
        print('Measuring regions')
        properties = regionprops_table(cell_lbl, properties=['label', 'centroid', 'area', 'MajorAxisLength', 'MinorAxisLength'])
        #cell_lbl = cell_lbl * (cell_lbl == minimum_filter(cell_lbl, size=(1,3,3)))

        # Compute object surfaces from label mask
        #combined_verts, combined_faces, combined_values, custom_colormap = lbl2mesh(cell_lbl)

        #### Export results
        print('-------------------------------')
        print('Exporting results...')
        imsave(str(imagefile).replace('.tif', '_lbl.tif'), np.uint16(cell_lbl), check_contrast=False)
        chmod(str(imagefile).replace('.tif', '_lbl.tif'), 0o666)
        print(str(imagefile).replace('.tif', '_lbl.tif'))
        df = pd.DataFrame(properties)
        df.columns = ['Cell', 'CZ', 'CY', 'CX', 'Volume (vox)', 'MajorAxis (pix)', 'MinorAxis (pix)']
        df.to_csv(str(imagefile).replace('.tif', '_lbl.csv'), index=False)
        chmod(str(imagefile).replace('.tif', '_lbl.csv'), 0o666)
        print(str(imagefile).replace('.tif', '_lbl.csv'))
        print('-------------------------------')
        print(f'Number of segmented cells: {len(df)}')

        #### Display results
        np.random.seed(0)
        vw.add_labels(cell_lbl, name=f"CellsLbl", blending="additive", scale=(zratio, 1, 1))
        #viewer.add_surface((combined_verts, combined_faces, combined_values), name='Combined Surface', colormap=custom_colormap, scale=(zratio, 1, 1))
        vw.layers.selection.active = viewer.layers['CellsLbl']

# Instantiate Napari viewer and add widgets
viewer = napari.Viewer()
dw1 = viewer.window.add_dock_widget(load_image_tiff, area='right', name='Load')
dw1.setMinimumHeight(120);dw1.setMaximumHeight(120);dw1.setMinimumWidth(360)
dw2 = viewer.window.add_dock_widget(seed_nuclei, area='right', name='Seed')
dw2.setMinimumHeight(260);dw2.setMaximumHeight(260)
dw3 = viewer.window.add_dock_widget(segment_cells, area='right', name='Label')
dw3.setMinimumHeight(240);dw3.setMaximumHeight(240)
dw4 = viewer.window.add_dock_widget(remove_label, area='right', name='Remove')
dw4.setMinimumHeight(120);dw4.setMaximumHeight(120)
dw5 = viewer.window.add_dock_widget(merge_labels, area='right', name='Merge')
dw5.setMinimumHeight(160);dw5.setMaximumHeight(160)
napari.run()
