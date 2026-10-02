"""
Drive the GUI offscreen: navigate, discard, group, ungroup, save.  Catches
wiring mistakes (missing widgets, None section_index reaching a %d format).
"""
import os
import sys
import tempfile
import warnings
from collections import OrderedDict

import numpy as np
from pyqtgraph.Qt import QtWidgets

import lhy_roi_tools as lrt
import lhy_roi_gui as gui

warnings.simplefilter('ignore')


class FakeImages(object):
    def get(self, fname):
        return np.zeros((200, 200), np.float32), (800, 800), (1.0, 1.0)


def build(n=5):
    secs = OrderedDict()
    for r in range(1, n + 1):
        k = lrt.section_key(1, r)
        secs[k] = lrt.empty_section(dict(file='s%d.nd2' % r, slide=1, region=r), r)
        secs[k]['pixel_size_um'] = [1.0, 1.0]
        secs[k]['image_shape'] = [800, 800]
        secs[k]['midline_px'] = [[400.0, 100.0], [400.0, 600.0]]
        secs[k]['surface_px'] = [[100.0, 100.0], [700.0, 100.0]]
        secs[k]['ellipses'] = [dict(center_px=[550.0, 350.0],
                                    axes_px=[60.0, 50.0], angle_deg=0.0)]
        secs[k]['lhy'] = 'yes'
    ann = OrderedDict([('format_version', lrt.FORMAT_VERSION), ('bird', 'TRQ82'),
                       ('hist_dir', ''),
                       ('series', OrderedDict(lrt.DEFAULT_SERIES_PARAMS)),
                       ('sections', secs)])
    ann['series'].update(section_thickness_um=40.0, ac_section=lrt.section_key(1, 1),
                         ac_ap_um=900.0, implant_side='right')
    lrt.assign_section_indices(ann)
    return ann


fails = []


def check(name, got, want):
    ok = got == want
    print('%-54s %s   got %s' % (name, 'ok ' if ok else 'FAIL', got))
    if not ok:
        fails.append('%s: got %s, want %s' % (name, got, want))


app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
tmp = tempfile.mkdtemp()
path = os.path.join(tmp, 'TRQ82_lhy_rois.json')

ann = build()
w = gui.LHyROIGUI(ann, path, FakeImages(), export_csv=True, shank_labels=['A', 'B'])

check('opened on a section', w.cur, 0)
check('list has every entry', w.section_list.count(), 5)

# --- discard the entry we are sitting on ----------------------------------
w.goto(2)
w.set_discarded(True)
check('discarded flag set', w.sec(lrt.section_key(1, 3))['discarded'], True)
check('AP index dropped',
      [w.sec(k)['section_index'] for k in w.keys], [0, 1, None, 2, 3])
check('still listed', w.section_list.count(), 5)
check('list row renders', 'not a section' in w.section_list.item(2).text(), True)
check('banner shown', w.discard_label.isVisible(), True)
check('checkbox reflects it', w.discard_box.isChecked(), True)
w._update_status()
check('status says so', 'NOT A SECTION' in w.status.text(), True)

# navigating onto and off a discarded entry must not blow up
w.goto(3)
w.goto(2)
w.goto(1)
check('navigation survived', w.cur, 1)

# next-unreviewed must skip it
for k in w.keys:
    w.sec(k)['lhy'] = None
w.goto(1)
w.next_unreviewed()
check('u skips the discarded entry', w.keys[w.cur], lrt.section_key(1, 4))

w.goto(2)
w.set_discarded(False)
check('un-discarded', [w.sec(k)['section_index'] for k in w.keys], [0, 1, 2, 3, 4])

# --- group two entries as pieces of one broken section --------------------
w.goto(2)
w.group_with_previous()
k2, k3 = lrt.section_key(1, 2), lrt.section_key(1, 3)
check('both pieces in a group',
      w.sec(k2)['piece_group'] == w.sec(k3)['piece_group'] is not None, True)
check('one AP step for the group',
      [w.sec(k)['section_index'] for k in w.keys], [0, 1, 1, 2, 3])
check('piece letters in the list', w._index_text(k2) + w._index_text(k3),
      '  1a  1b')
w._refresh_panel()
check('panel shows the piece', w.group_label.text(), 'piece 2/2')

# a third piece
w.goto(3)
w.group_with_previous()
check('three pieces, still one step',
      [w.sec(k)['section_index'] for k in w.keys], [0, 1, 1, 1, 2])

# copying across a fracture warns
w.goto(3)
w.midline_roi = None
w.copy_previous()
check('copy across a fracture warns', 'SAME SECTION' in w.message.text(), True)

# ungroup the middle piece: the remaining pieces are no longer adjacent,
# which is contradictory and should be flagged
w.goto(2)
w.ungroup()
by_key = {k: w.sec(k)['section_index'] for k in w.keys}
check('ungrouped piece is its own section',
      by_key[lrt.section_key(1, 3)] != by_key[lrt.section_key(1, 2)], True)
check('remaining pieces still share a step',
      by_key[lrt.section_key(1, 2)] == by_key[lrt.section_key(1, 4)], True)
check('still five sections in the series', len(set(by_key.values())), 4)
w._refresh_list()
check('non-adjacent pieces flagged', 'sits between the pieces' in w.message.text(),
      True)

# put it back so the rest of the test starts clean
w.goto(w.keys.index(lrt.section_key(1, 4)))
w.ungroup()
w.goto(w.keys.index(lrt.section_key(1, 2)))
w.ungroup()
check('all ungrouped',
      [w.sec(k)['section_index'] for k in w.keys], [0, 1, 2, 3, 4])

# --- discarding the AC reference clears it --------------------------------
w.goto(0)
w.set_ac_here()
check('AC set', w.ann['series']['ac_section'], lrt.section_key(1, 1))
w.set_discarded(True)
check('AC cleared when discarded', w.ann['series']['ac_section'], None)
w.set_discarded(False)

# --- save and reload -------------------------------------------------------
w.goto(w.keys.index(lrt.section_key(1, 3)))
w.group_with_previous()                       # one broken section on disk
w.save()
check('json written', os.path.exists(path), True)
reloaded = lrt.load_annotation(path)
check('groups survive a round trip',
      sum(1 for s in reloaded['sections'].values() if s.get('piece_group')), 2)
check('csv exported', os.path.exists(os.path.splitext(path)[0] + '_sections.csv'),
      True)


# ---------------------------------------------------------------------------
# overlay items must not leak between sections.  A list left out of
# _clear_items keeps its items after they are taken off the ViewBox, so they
# are re-removed forever (Qt: "item's scene (0x0) is different from this
# scene") and _pull_items copies them into every section that is opened.
# ---------------------------------------------------------------------------
check('every overlay list is cleared',
      sorted(gui.LHyROIGUI.ITEM_LISTS),
      sorted(['midline_labels', 'ellipse_rois', 'scar_rois',
              'dmdl_items', 'ac_items']))

w.goto(0)
w.mouse_xy = (402.0, 300.0)
w.add_ac_point()
w.mouse_xy = (398.0, 300.0)
w.add_ac_point()
home = w.keys[0]
check('two AC marks placed', len(w.ann['sections'][home]['ac_px']), 2)

for i in range(6):                      # walk the series and come back
    w.goto((i + 1) % len(w.keys))

check('AC marks do not follow the section',
      [k for k, s in w.ann['sections'].items() if s.get('ac_px')], [home])
check('AC marks did not multiply', len(w.ann['sections'][home]['ac_px']), 2)
check('no overlay items left over after a clear',
      [len(getattr(w, n)) for n in gui.LHyROIGUI.ITEM_LISTS
       if n != 'midline_labels'][:3], [1, 0, 0])

w.goto(w.keys.index(home))
check('AC marks reload on their own section', len(w.ac_items), 2)
w.goto(1 if home == w.keys[0] else 0)
check('and are gone again elsewhere', len(w.ac_items), 0)

w.close()
print()
if fails:
    print('%d FAILURE(S)' % len(fails))
    for f in fails:
        print('  ' + f)
    sys.exit(1)
print('all checks passed')
