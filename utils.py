import numpy as np
import ctypes
from napari import Viewer
from magicgui import magicgui
from skimage.io import imread
from skimage.morphology import reconstruction
from skimage.measure import label, regionprops
from scipy.ndimage import binary_fill_holes, maximum_filter
from napari.utils.notifications import show_warning
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

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
    idx = set(range(len(seeds))) - set(sum(mergelst, []))

    # Move seeds at their associated clusters' centers of mass
    seeds = [tuple(np.round(np.mean(np.array(seeds)[[i]+lst], axis=0)).astype(int)) for i, lst in enumerate(mergelst)]

    return [seeds[i] for i in idx]

# Merge seeds connected by line profiles (within distance dstthr) with a range below deltamin (single linkage)
def merge_seeds(img, seeds, dstthr, deltamin, zratio):
    seeds = np.array(seeds)
    if len(seeds) < 2:
        return [tuple(s) for s in seeds]

    # Seed pairs within distance dstthr with low range line profile
    clean = []
    for i, j in cKDTree(seeds*np.array((zratio, 1, 1))).query_pairs(dstthr):
        profile = img[tuple(interpolate_3d_line(seeds[i], seeds[j]).T)].astype(float)
        if profile.max() - profile.min() < deltamin:
            clean.append((i, j))

    # Groups of connected seeds, moved to their centers of mass
    rows, cols = zip(*clean) if clean else ((), ())
    graph = coo_matrix((np.ones(len(clean)), (rows, cols)), shape=(len(seeds), len(seeds)))
    ncomp, comp = connected_components(graph, directed=False)

    return [tuple(np.round(seeds[comp == c].mean(axis=0)).astype(int)) for c in range(ncomp)]

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

# Label layers edited by the remove / merge tools (cell labels first)
edit_layers = ('CellsLbl', 'NucleiLbl')

# Undo history: one entry per action, holding for each edited layer the flat indices of changed voxels and their old label
edit_history = []

# Replace label src by dst in cell (and nuclei) label masks, recording the change for undo
def replace_label_masks(vw: Viewer, src, dst):
    entry = []
    for name in edit_layers:
        if viewer_is_layer(vw, name):
            lbl = vw.layers[name].data
            idx = np.flatnonzero(lbl == src)
            if idx.size:
                lbl.ravel()[idx] = dst
                vw.layers[name].data = lbl
                entry.append((name, idx, src))
    if entry:
        edit_history.append(entry)

# Remove a label from cell (and nuclei) label masks
def delete_label_masks(vw: Viewer, label):
    replace_label_masks(vw, label, 0)

# Merge label2 into label1 in cell (and nuclei) label masks
def merge_label_masks(vw: Viewer, label1, label2):
    replace_label_masks(vw, label2, label1)

# Undo last remove / merge action
def undo_label_edit(vw: Viewer):
    if not edit_history:
        show_warning('Nothing to undo')
        return
    for name, idx, old in edit_history.pop():
        if viewer_is_layer(vw, name):
            lbl = vw.layers[name].data
            lbl.ravel()[idx] = old
            vw.layers[name].data = lbl
    print('Undid last label edit')

# Crosshair mouse cursor of a given color (thin white outline to stay visible on dark backgrounds)
def color_crosshair_cursor(rgb, size=33):
    from qtpy.QtCore import Qt
    from qtpy.QtGui import QCursor, QPixmap, QPainter, QPen, QColor
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    c, gap = size // 2, 3
    segs = [(c, 0, c, c-gap), (c, c+gap, c, size-1), (0, c, c-gap, c), (c+gap, c, size-1, c)]
    for pen in (QPen(QColor(255, 255, 255), 4), QPen(QColor(*[int(round(255*v)) for v in rgb[:3]]), 2)):
        p.setPen(pen)
        for seg in segs:
            p.drawLine(*seg)
    p.end()
    return QCursor(pm, c, c)

@magicgui(call_button='Remove Label', label={'widget_type': 'SpinBox', 'min': 0})
def remove_label(vw: Viewer, label):

    if viewer_is_layer(vw, 'CellsLbl'):
        delete_label_masks(vw, label)
    else:
        print('!! No "CellsLbl" layer found !!')

    return None


@magicgui(call_button='Merge Labels', label1={'widget_type': 'SpinBox', 'min': 1}, label2={'widget_type': 'SpinBox', 'min': 1})
def merge_labels(vw: Viewer, label1, label2):

    if viewer_is_layer(vw, 'CellsLbl'):
        merge_label_masks(vw, label1, label2)
    else:
        print('!! No "CellsLbl" layer found !!')

    return None


# Interactive label editing tool: 'm' = merge (two clicks), 'd' = delete (one click), 'u' = undo, right click / Esc = exit
class LabelEditTool:

    def __init__(self, vw: Viewer):
        self.vw = vw
        self.mode = None
        self.picked = []
        vw.mouse_drag_callbacks.append(self._on_click)
        # napari resets the canvas cursor on these events: re-apply the tool cursor afterwards
        vw.cursor.events.style.connect(self._apply_cursor, position='last')
        vw.cursor.events.size.connect(self._apply_cursor, position='last')
        vw.camera.events.zoom.connect(self._apply_cursor, position='last')
        # Bind on the viewer and on every layer: active layer keymaps (e.g. Labels 'm') take precedence
        self._bind(vw)
        for ly in vw.layers:
            self._bind(ly)
        vw.layers.events.inserted.connect(lambda e: self._bind(e.value))

    def _bind(self, provider):
        provider.bind_key('m', lambda *_: self.start('merge'), overwrite=True)
        provider.bind_key('d', lambda *_: self.start('delete'), overwrite=True)
        provider.bind_key('u', lambda *_: self.undo(), overwrite=True)
        # Leave the tool if the user switches a layer to another mode (paint, fill...)
        if hasattr(provider, 'events') and hasattr(provider.events, 'mode'):
            provider.events.mode.connect(self._on_layer_mode)

    def _on_layer_mode(self, event):
        if self.mode is not None and str(event.source.mode) != 'pan_zoom':
            self.stop()

    def undo(self):
        undo_label_edit(self.vw)
        if self.mode == 'merge':
            self.picked = []
        if self.mode is not None:
            self._update_overlay()

    # Tool cursor: delete = red, merge = black or color of first picked label
    def _apply_cursor(self, event=None):
        if self.mode is None:
            return
        if self.mode == 'delete':
            rgb = (1, 0, 0)
        elif self.picked:
            rgb = self.vw.layers['CellsLbl'].get_color(self.picked[0])
        else:
            rgb = (0, 0, 0)
        self.vw.window._qt_viewer.canvas.cursor = color_crosshair_cursor(rgb)

    def start(self, mode):
        if not viewer_is_layer(self.vw, 'CellsLbl'):
            show_warning('No "CellsLbl" layer found!')
            return
        self.mode = mode
        self.picked = []
        # Make sure the label layers are in move (pan/zoom) mode so clicks do not paint
        for name in edit_layers:
            if viewer_is_layer(self.vw, name):
                self.vw.layers[name].mode = 'pan_zoom'
        # Esc exits the tool; only bound while active so napari's own Esc (e.g. reset polygon) still works otherwise
        for provider in [self.vw, *self.vw.layers]:
            provider.bind_key('Escape', lambda *_: self.stop(), overwrite=True)
        self._update_overlay()

    def stop(self):
        if self.mode is None:
            return
        for provider in [self.vw, *self.vw.layers]:
            try:
                provider.bind_key('Escape', None)
            except (KeyError, ValueError):
                pass
        self.mode = None
        self.picked = []
        self.vw.text_overlay.visible = False
        self.vw.cursor.events.style()  # let napari restore its own cursor

    def _update_overlay(self):
        if self.mode == 'merge':
            step = 'first' if not self.picked else f'second (merge into {self.picked[0]})'
            txt = f'MERGE: click {step} label - right click to exit'
        else:
            txt = 'DELETE: click label to remove - right click to exit'
        self.vw.text_overlay.text = txt
        self.vw.text_overlay.visible = True
        self._apply_cursor()

    # Label under the mouse pointer (2D or 3D ray picking), cell mask first then nuclei mask
    def _pick_label(self, event):
        for name in edit_layers:
            if viewer_is_layer(self.vw, name):
                val = self.vw.layers[name].get_value(event.position, view_direction=event.view_direction,
                                                     dims_displayed=event.dims_displayed, world=True)
                if val is not None and int(val) > 0:
                    return int(val)
        return 0

    def _on_click(self, vw, event):
        if self.mode is None:
            return
        if event.button == 2:
            self.stop()
            return
        if event.button != 1:
            return
        # Ignore drags (camera rotation / panning)
        dragged = False
        yield
        while event.type == 'mouse_move':
            dragged = True
            yield
        if dragged:
            return
        label = self._pick_label(event)
        if label == 0:
            show_warning('No label under pointer')
            return
        if self.mode == 'delete':
            delete_label_masks(self.vw, label)
            print(f'Removed label {label}')
        else:
            if label in self.picked:
                # Clicking the same cell again cancels the pending merge
                self.picked = []
                self._update_overlay()
                return
            self.picked.append(label)
            if len(self.picked) == 2:
                merge_label_masks(self.vw, self.picked[0], self.picked[1])
                print(f'Merged label {self.picked[1]} into {self.picked[0]}')
                self.picked = []
        self._update_overlay()
