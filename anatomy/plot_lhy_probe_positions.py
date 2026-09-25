from lhy_roi_tools import load_lhy_rois
import lhy_3d_plots
import sys
sys.path.append("..//utils/")
import color_utils

# set paths
root_dir = 'Z:/Isabel/data/lhy_implants/'
annotation_folder = 'histology_annotations'
save_figs_dir = f"../figures/basic_neural_analysis/"

# set bird list
bird_ids = ['LMN88', 'TRQ82']

# get the annotation data per bird
rois_by_bird = {}
for bird in bird_ids: 
    rois_by_bird[bird] = load_lhy_rois(f'{root_dir}{bird}/histology_annotations/{bird}_lhy_rois.json')

# plot for all birds
bird_colors_list = color_utils.get_bird_colors_da(bird_ids)
bird_colors_dict = {}
for bird, color in zip(bird_ids, bird_colors_list):
    bird_colors_dict[bird] = color
fig, ax, avg = lhy_3d_plots.plot_tracks_and_lhy(rois_by_bird, 
                                                hemisphere='mirror',
                                                colors=bird_colors_dict, 
                                                show_bird_lhy=False)

fig.savefig(f'{save_figs_dir}probe_tracks_lhy_folded.png', dpi=400, bbox_inches='tight')