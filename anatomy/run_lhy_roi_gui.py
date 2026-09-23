#!/usr/bin/env python
"""
run_lhy_roi_gui.py
==================

Set the parameters below and run:

    python run_lhy_roi_gui.py

Once per bird
-------------
    * check the cutting order: it is taken from where each section sits on the
      slide overview (down each column, then left to right).  Fix any section
      with the 'order on slide' box or [ / ] to move it earlier / later; the
      list shows index, slide, #order.
    * 'lost before' = sections missing between this one and the previous one
    * 'a' on the anterior commissure reference section
    * 'd' on anything that is not a section at all: a speck of detritus, a
      bubble or a smear that got picked up as a region and imaged.  It leaves
      the AP count without being counted as a lost section, so its neighbours
      end up adjacent.  Press d again to bring it back.
    * 'c' on a section whose high-resolution scan does not cover the whole
      slice: it opens the same section on the slide overview (~8 um/px) as a
      companion entry, so you can annotate the missing part instead of losing
      it.  The two share one step of the AP series; draw a midline and a
      surface on the companion too, since it has its own pixels.  'c' again
      goes back to the high-res scan.
    * 'g' on each extra piece of a section that broke up during mounting, to
      join it to the piece before it.  The whole group takes ONE step of the
      AP series, and every piece keeps its own midline, surface, ellipses and
      scars -- annotate each piece separately, because once the pieces have
      shifted apart, a pixel distance measured across a break means nothing.
      shift+G splits a piece off again.

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
    7. w                  on sections where the scar is near the surface, mark
                          the DM/DL boundary where it meets the dorsal surface
                          (one per hemisphere).  Its distance from the midline
                          is the hippocampal width, which gives an AP from a
                          landmark at the same depth as the entry point
    7. -> or u            next section (autosaves)

Other keys
----------
    left / right    previous / next section     u   next unreviewed
    + / -           zoom in / out at the cursor     z (or f)  whole section
                    (sections taken from the slide overview open zoomed on the
                     section, and z returns there)
    click ellipse   select;  Del removes it (or the scar under the cursor)
    v               copy midline / surface / ellipses from the previous section
    d               not a section (detritus)    g / shift+G  piece of a broken
                                                section / split it off again
    h               hide / show overlays        f   fit view
    ctrl+s          save                        q   quit
    on a traced line: click a segment to add a vertex, right-click a vertex
    to remove it

Loading for analysis
--------------------
    from lhy_roi_tools import load_lhy_rois
    rois = load_lhy_rois(annotation_path)
    rois.scar_tracks()                               # tip, insertion, slopes, fit
    rois.ap_calibration()                            # AC vs hippocampal width
    rois.shear                                       # per-hemisphere shift + pitch
    rois.hp_marks                                    # one row per DM/DL mark
    rois.discarded                                   # entries that were not sections
    insert_coords, tip_coords = rois.probe_inputs(shanks=['A', 'B'])
    d = rois.distance(cell_pos)                      # (n, 3) [ML, AP, DV] um
    in_lhy = rois.label(cell_pos, tol_um=150)

    rois.sections has one row per ENTRY: a broken section contributes one row
    per piece, all sharing a section_index and an AP, with piece / n_pieces
    saying which is which.
"""

from lhy_roi_gui import launch


# --------------------------------------------------------------------------- #
#  PARAMETERS                                                                  #
# --------------------------------------------------------------------------- #

bird = 'TRQ82'
hist_folder = 'bad_model'
hist_root = 'Z:/Isabel/histology/lhy_implants/'
data_root = 'Z:/Isabel/data/lhy_implants/'

# folder holding the Slide1-N_Region000M_Channel395 nm_Seq0008.nd2 files
hist_dir = f'{hist_root}{bird}/{hist_folder}/'

# where the annotations go: this JSON, plus *_sections.csv, *_ellipses.csv and
# *_scar_tracks.csv beside it, rewritten on every save for a quick look.
annotation_path = f'{data_root}{bird}/histology_annotations/{bird}_lhy_rois.json'

# only use files from this channel ('395 nm' and '395_nm' both match; None = any)
channel = '395 nm'

# --- slide overview inset ---------------------------------------------------
# small bird's-eye view of the whole slide in the top-left corner, with a dot
# per section on that slide and the current one circled.  'i' hides it.
show_slide_inset = True

# whole-slide scans: Slide1-N_Channel395 nm_Seq0000.nd2
slide_pattern = (r'Slide\d+-(?P<slide>\d+)_Channel(?P<channel>.+?)'
                 r'_Seq(?P<seq>\d+)\.nd2$')

# sections are mounted down each column, then left to right, so the overview
# also gives the cutting order:   1 | 3 | 5
#                                 2 | 4 | 6
# Set False to keep the order the files come in (slide, then region).  A slide
# whose order you set by hand is never overwritten; shift+O re-reads it anyway.
auto_slide_order = True

# an image that covers two sections at once (sections mounted touching) is
# detected from its frame on the slide overview and split into one entry per
# section, each cropped out of the shared image, so both are shown in order and
# each takes its own step in the AP series.  This happens automatically; no
# parameter.
#
# The opposite case -- one section that BROKE into several pieces -- cannot be
# detected automatically, because a piece looks just like a small section.
# Group the pieces by hand with 'g' so they share a single AP step.

# sections that were mounted but never imaged at high resolution become their
# own entries, annotated on a zoomed-in crop of the slide overview (~8 um/px --
# coarse, but better than leaving a section with the scar unannotated).  Set
# False to only count them as gaps in the AP series.
#
# Keep this on if the slides have junk on them that was never imaged: it is
# what gives that junk an entry you can mark with 'd'.  With it off, a speck
# of detritus is an anonymous gap and you have to correct 'lost before' by hand.
annotate_unimaged = True

# sections are placed on the overview from the stage coordinates in the nd2
# metadata.  The stage axes can run either way relative to the image, so the
# signs are worked out from how well the sections line up with the tissue on
# the overview and then reused for the other slides.  Pin them here as
# (x_sign, y_sign) if that ever gets it wrong; click in the inset to place one
# section by hand, 'o' to undo that.
slide_axis_signs = None

# how big the inset is, in screen pixels, and how much the overview is
# downsampled for it
inset_width_px = 380
slide_downsample = 4

# probe shank labels, as in the anatomy sheet (shank A = smallest x in
# channel_positions.npy).  Keys 1-8 pick the shank to trace.
shank_labels = ['A', 'B']
# shank_labels = ['A', 'B', 'C', 'D']

# --- series ---------------------------------------------------------------- #
# These override whatever is stored in the JSON whenever they are not None, so
# set one to None if you would rather edit it in the GUI.

# cut thickness of each section, um   (EDIT -- placeholder)
section_thickness_um = 100.0

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

# depth of the AC below the brain surface, um.  Only used to turn the offset
# between the two AP estimates (AC vs hippocampal width) into a section-angle
# estimate in rois.ap_calibration().  None = report the offset only.
ac_dv_um = None

# which landmark sets AP:
#   'ac'    the anterior commissure, via section counting
#   'hp'    shift the whole series onto the DM/DL marks, which sit at the
#           surface and so are much less sensitive to the section angle
#   'shear' use BOTH: the AC pins the deep end, the DM/DL marks pin the
#           surface, and every annotation is corrected in proportion to its
#           own depth, per hemisphere.  This is the one to use for comparing
#           AP across birds.  Needs ac_dv_um; falls back to 'hp' without it.
ap_anchor = 'ac'

# the AC in the OTHER hemisphere, if slicing yaw puts it on a different
# section: (slide, region), or None to use ac_section for both.  With
# ap_anchor='shear' each hemisphere is then pinned to its own AC.
ac_section_other = None

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
           ac_section_other=ac_section_other,
           ac_dv_um=ac_dv_um,
           ap_anchor=ap_anchor,
           implant_side=implant_side,
           inplane_scale=inplane_scale,
           shank_labels=shank_labels,
           pixel_size_override=pixel_size_override,
           show_slide_inset=show_slide_inset,
           slide_pattern=slide_pattern,
           slide_downsample=slide_downsample,
           slide_axis_signs=slide_axis_signs,
           auto_slide_order=auto_slide_order,
           annotate_unimaged=annotate_unimaged,
           inset_width_px=inset_width_px,
           display_downsample=display_downsample,
           cache_dir=cache_dir,
           default_ellipse_axes_um=default_ellipse_axes_um,
           dv_mode=dv_mode,
           ref_ml_um=ref_ml_um,
           keep_levels=keep_levels)
