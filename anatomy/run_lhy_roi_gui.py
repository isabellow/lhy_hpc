#!/usr/bin/env python
"""
run_lhy_roi_gui.py
==================

Set the parameters below and run:

    python run_lhy_roi_gui.py

Once per bird
-------------
    * put each section in cutting order on its slide: 'order on slide' box, or
      [ / ] to move it earlier / later.  The list shows index, slide, #order.
    * 'lost before' = sections missing between this one and the previous one
    * 'a' on the anterior commissure reference section

Per section
-----------
    1. y / n / ?          is LHy on this section?
    2. x                  if the section was mounted flipped
    3. m                  midline at the cursor; drag D to the dorsal midline
    4. s, click, Enter    dorsal brain surface (span the ML range you need)
    5. e                  ellipse around LHy; drag / resize / rotate
    6. 1-8, t, click, Enter   trace the probe scar for that shank, on every
                          section where it appears -- follow it to its very
                          end on the deepest section
    7. -> or u            next section (autosaves)

Other keys
----------
    left / right    previous / next section     u   next unreviewed
    click ellipse   select;  Del removes it (or the scar under the cursor)
    v               copy midline / surface / ellipses from the previous section
    h               hide / show overlays        f   fit view
    ctrl+s          save                        q   quit
    on a traced line: click a segment to add a vertex, right-click a vertex
    to remove it

Loading for analysis
--------------------
    from lhy_roi_tools import load_lhy_rois
    rois = load_lhy_rois(annotation_path)
    rois.scar_tracks()                               # tip, insertion, slopes, fit
    insert_coords, tip_coords = rois.probe_inputs(shanks=['A', 'B'])
    d = rois.distance(cell_pos)                      # (n, 3) [ML, AP, DV] um
    in_lhy = rois.label(cell_pos, tol_um=150)
"""

from lhy_roi_gui import launch


# --------------------------------------------------------------------------- #
#  PARAMETERS                                                                  #
# --------------------------------------------------------------------------- #

bird = 'LMN86'
root_dir = 'Z:/Isabel/data/lhy_implants/'

# folder holding the Slide1-N_Region000M_Channel395 nm_Seq0008.nd2 files
hist_dir = f'{root_dir}{bird}/histology/'

# where the annotations go (JSON; *_sections.csv, *_ellipses.csv and
# *_scar_tracks.csv are rewritten next to it on every save for a quick look)
annotation_path = f'{hist_dir}{bird}_lhy_rois.json'

# only use files from this channel ('395 nm' and '395_nm' both match; None = any)
channel = '395 nm'

# probe shank labels, as in the anatomy sheet (shank A = smallest x in
# channel_positions.npy).  Keys 1-8 pick the shank to trace.
shank_labels = ['A']
# shank_labels = ['A', 'B', 'C', 'D']

# --- series ---------------------------------------------------------------- #
# These override whatever is stored in the JSON whenever they are not None, so
# set one to None if you would rather edit it in the GUI.

# cut thickness of each section, um   (EDIT -- placeholder)
section_thickness_um = 40.0

# cut sections per step in the series: 1 if every section was mounted, 2 if
# every other section went on these slides, etc.
section_interval = 1

# slides are imaged anterior -> posterior, so a higher index is more posterior
ap_sign = -1

# anterior commissure reference section as (slide, region), or None to set it
# in the GUI with 'a'
ac_section = None

# AP of the AC relative to lambda, um -- ANT_COM_AP in get_probe_coords_lhy.py
ac_ap_um = 900

# implanted hemisphere as seen on an UNFLIPPED, dorsal-up section: 'right' or
# 'left' of the image.  None = infer it from the scar traces.  ML comes out
# signed with + = implanted hemisphere, as in get_probe_coords_lhy.
implant_side = None

# multiply in-plane distances by this to undo tissue shrinkage (1.0 = none)
inplane_scale = 1.0


# --------------------------------------------------------------------------- #
#  OPTIONAL EXTRAS  (fine to leave alone)                                      #
# --------------------------------------------------------------------------- #

# regex with named groups slide / region (channel / seq optional)
file_pattern = (r'Slide\d+-(?P<slide>\d+)_Region(?P<region>\d+)'
                r'_Channel(?P<channel>.+?)_Seq(?P<seq>\d+)\.nd2$')

# um per pixel (x, y) if the nd2 metadata is missing or wrong; None = metadata
pixel_size_override = None

# display downsampling (annotations are always stored in full-res pixels).
# 4 turns a 16790 x 11876 scan into ~4200 x 3000 for display.
display_downsample = 4

# local folder for downsampled copies of the images.  Reading one full scan
# takes several seconds; with a cache every later visit is instant.  None = off
cache_dir = None
# cache_dir = 'C:/temp/lhy_hist_cache/'

# semi-axes of a new ellipse, um
default_ellipse_axes_um = (250., 200.)

# DV convention for the cursor readout and the csv export only (the JSON never
# stores DV): 'local', 'ref_ml' (needs ref_ml_um) or 'midline'
dv_mode = 'local'
ref_ml_um = None

# reuse the contrast from the previous section
keep_levels = True


# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    launch(bird=bird,
           hist_dir=hist_dir,
           annotation_path=annotation_path,
           file_pattern=file_pattern,
           channel=channel,
           section_thickness_um=section_thickness_um,
           section_interval=section_interval,
           ap_sign=ap_sign,
           ac_section=ac_section,
           ac_ap_um=ac_ap_um,
           implant_side=implant_side,
           inplane_scale=inplane_scale,
           shank_labels=shank_labels,
           pixel_size_override=pixel_size_override,
           display_downsample=display_downsample,
           cache_dir=cache_dir,
           default_ellipse_axes_um=default_ellipse_axes_um,
           dv_mode=dv_mode,
           ref_ml_um=ref_ml_um,
           keep_levels=keep_levels)
