#!/usr/bin/env python
"""
Repair an annotation whose AC marks leaked across sections.

A GUI version before this fix never emptied `ac_items` when the section
changed, so every AC mark ever placed stayed in the list, was re-added on each
section load, and got written into whichever section happened to be open.  The
file ends up with thousands of duplicated AC points spread over many sections,
the AC depth (`ac_dv_um`) is averaged over sections it has nothing to do with,
and the GUI slows to a halt under the weight of the leaked overlay items.

This script collapses the duplicates and keeps the marks only on the AC
reference section(s) named in `series` -- it does not invent anything: the
distinct coordinates are exactly the points that were clicked.

    python repair_ac_px.py BIRD_lhy_rois.json              # report only
    python repair_ac_px.py BIRD_lhy_rois.json --write      # repair in place
    python repair_ac_px.py BIRD_lhy_rois.json --write \
        --ac-section slide09_region001_lowres              # override the home

The original is copied to <name>.prerepair.bak before anything is written.
"""
import os
import sys
import json
import shutil
import argparse


def unique_points(pts):
    """The distinct points of a list, in the order they first appear."""
    seen, out = set(), []
    for xy in pts or []:
        t = (float(xy[0]), float(xy[1]))
        if t not in seen:
            seen.add(t)
            out.append([float(xy[0]), float(xy[1])])
    return out


def repair(ann, ac_section=None, ac_section_other=None):
    """
    Collapse duplicate AC marks and keep them only on the AC reference
    section(s).  Returns (report, changed) and edits `ann` in place.
    """
    series = ann.get('series', {})
    home = ac_section or series.get('ac_section')
    other = ac_section_other if ac_section_other is not None \
        else series.get('ac_section_other')
    keep = {k for k in (home, other) if k}

    report, changed = [], False
    for key, s in ann.get('sections', {}).items():
        pts = s.get('ac_px') or []
        if not pts:
            continue
        uniq = unique_points(pts)
        if key in keep:
            action = 'kept on the AC reference section'
            new = uniq
        elif not keep:
            action = ('NO ac_section set in series -- left alone, rerun with '
                      '--ac-section')
            new = uniq
        else:
            action = 'cleared (not an AC reference section)'
            new = []
        report.append(dict(key=key, had=len(pts), unique=len(uniq),
                           now=len(new), action=action,
                           points=[tuple(round(c, 2) for c in p) for p in uniq]))
        if new != pts:
            s['ac_px'] = new
            changed = True
    return report, changed


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('path', help='BIRD_lhy_rois.json')
    ap.add_argument('--write', action='store_true',
                    help='write the repair back (default: report only)')
    ap.add_argument('--ac-section', default=None,
                    help='override the AC reference section key')
    ap.add_argument('--ac-section-other', default=None,
                    help='override the other-hemisphere AC section key')
    a = ap.parse_args(argv)

    with open(a.path) as fh:
        ann = json.load(fh)

    report, changed = repair(ann, a.ac_section, a.ac_section_other)

    if not report:
        print('no AC marks in %s -- nothing to do' % a.path)
        return 0

    print('%-28s %9s %7s %5s  %s' % ('section', 'ac_px', 'unique', 'now', 'action'))
    for r in report:
        print('%-28s %9d %7d %5d  %s'
              % (r['key'], r['had'], r['unique'], r['now'], r['action']))
    print()
    for r in report:
        if r['now']:
            print('  %s keeps %s' % (r['key'], r['points']))

    if not changed:
        print('\nalready clean -- nothing written')
        return 0
    if not a.write:
        print('\n(report only -- rerun with --write to apply)')
        return 0

    backup = os.path.splitext(a.path)[0] + '.prerepair.bak'
    shutil.copyfile(a.path, backup)
    with open(a.path, 'w') as fh:
        json.dump(ann, fh, indent=1)
    print('\noriginal copied to %s' % backup)
    print('repaired %s (%.1f kB)' % (a.path, os.path.getsize(a.path) / 1e3))
    return 0


if __name__ == '__main__':
    sys.exit(main())
