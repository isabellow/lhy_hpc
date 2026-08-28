import numpy as np
import estimate_target_coords as est

''' Calculate the 3D brain location of LHy (empirically determined) '''
# known insertion params
AP = 1.4    # mm anterior of the lambdoid sinus
ML = 0.48   # mm lateral of midline
DV = 5.8    # mm travelled by the probe

# tested head angle: beak bar, degrees below horizontal
pitch = 49
roll = 0.0  # radians

lhy_loc = est.get_target_loc(AP_entry=AP,
                             ML_entry=ML,
                             DV_probe=DV,
                             pitch_deg=pitch,
                             roll_rad=roll)

rough_loc = np.round(lhy_loc, 2)
print(f"empirically estimated LHy location in the brain: AP = {rough_loc[0]}, ML = {rough_loc[1]}, DV = {rough_loc[2]}\n")


''' Calculate the beak bar and probe angle needed to hit LHy from a given insertion point '''
# estimate ML roll given DV offset L/R
DV_left = float(input("DV left = "))
DV_right = float(input("DV right = "))
ML_offset = float(input("ML offset = "))
roll_new = est.estimate_roll(DV_left, DV_right, ml_offset=ML_offset)

# insertion coordinates
AP_entry = float(input("desired AP = "))  # mm
ML_entry = float(input("desired ML = "))  # mm

# get new targeting angles
new_angles = est.get_new_angles(lhy_loc, AP_entry, ML_entry, roll_new)

print("\n***************\nnew targeting params:")
print(f"beak bar angle: {new_angles['pitch_deg']} degrees below horizontal")
print(f"probe angle towards midline: {new_angles['probe_angle_deg']} degrees")
print(f"\nmanipulator dials from landmark:\nAP = {new_angles['dial_AP']} mm, ML = {new_angles['dial_ML']} mm")
print(f"DV travel distance: {new_angles['travel_probe']} mm")
print("***************")
