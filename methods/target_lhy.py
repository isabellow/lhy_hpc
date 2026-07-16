import numpy as np
import estimate_target_coords as est

''' Calculate the 3D brain location of LHy (empirically determined) '''
# known insertion params
AP = 1.4
ML = 0.48
DV = 5.8

# tested head angle
pitch = 49 # degrees
roll = 0 

lhy_loc = est.get_target_loc(AP_entry=AP, 
                            ML_entry=ML, 
                            DV_probe=DV, 
                            pitch_deg=90-pitch, 
                            roll_rad=0)

rough_loc = np.round(lhy_loc, 2)
print(f"empirically estimated LHy location in the brain: AP = {rough_loc[0]}, ML = {rough_loc[1]}, DV = {rough_loc[2]}\n")


''' Calculate the new probe targeting coordinates '''
# estimate ML roll given DV offset L/R
DV_left = input("DV left = ")
DV_right = input("DV right = ")
ML_offset = input("ML offset = ")
roll_new = est.estimate_roll(float(DV_left), float(DV_right), ml_offset=float(ML_offset))

# new head angle
pitch = input("beak bar angle = ") # degrees

# get new targeting coords
new_coords = est.get_new_coords(lhy_loc, 90-int(pitch), roll_new)

print("\n***************\nupdated targeting coordinates:")
print(new_coords)
print("***************")