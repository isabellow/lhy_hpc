import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.image as mpimg

from skimage.measure import regionprops, label
from scipy.spatial import ConvexHull

import sys
sys.path.append("../utils/")
import load_matlab_data

def load_ref_obj(file_path):
    ref_obj = load_matlab_data.loadmat_sbx(file_path)
    arena_ref = ref_obj['arena_ref']
    print(arena_ref.keys())
    return arena_ref

def sort_arena_items(arena_im, arena_ref):
    '''
    Use a binary arena model image and reference info about the arena to ID
    key objects in the arena (perches, cache sites, feeders, water dish).
    Sort them by type and store their location, bounding box etc. in
    normalized arena coords (center is (0, 0), corners are +/-(1, 1)).

    Based on SC RigControl sort_arena_items.mat v2.4, using ChatGPT with IL
    proofreading/modifying to convert from matlab to python.

    Params
    ------
    arena_im : binary image, shape (n_pixels, n_pixels)
    	binary image of the arena floor made from the L1 laser cutting file
    arena_ref : dict
    	dictionary of arena properties enabling conversion b/w pixels and
    	normalized arena coordinates

    Returns
    -------
    arena_items : dict
    	dictionary of sorted arena objects:
    	"all_perches", "perch_no_site", "perch_w_site", "caches", "feeders", "water_dish"
    	each object is a dictionary with info about those objects:
    	"area", "centroid", "bbox", "orientation"
    '''
    ''' To convert pixels to normalized coordinates '''
    n_pixels = arena_ref['ImageSize'][0]
    intrinsic_to_world = np.linspace(arena_ref['XWorldLimits'][0], arena_ref['XWorldLimits'][1], n_pixels)
    intrinsic_to_world_grid = np.stack(np.meshgrid(intrinsic_to_world, intrinsic_to_world), axis=-1)

    ''' Extract the properties of all arena features'''
    labeled_image = label(arena_im > 0.5)
    regions = regionprops(labeled_image)
    stats = []
    for region in regionprops(labeled_image):
        # get the coordinates of the region's pixels
        coords = np.column_stack(region.coords)
        
        # get the ConvexHull vertices
        hull = ConvexHull(coords)
        convex_hull_points = coords[hull.vertices]
        
        # collect the region's properties
        stats.append({
            'Area': region.area,
            'Centroid': region.centroid,
            'Orientation': region.orientation,
            'ConvexHull': convex_hull_points
        })

    ''' Sort features by area '''
    perches = [region for region in stats if 2e5 < region['Area'] < 9e5]
    caches = [region for region in stats if 6e4 < region['Area'] < 2e5]
    feeders = [region for region in stats if region['Area'] > 4e6]
    water_dish = [region for region in stats if 9e5 < region['Area'] < 4e6]
    n_caches = len(caches)
    n_feeders = len(feeders)

    ''' Convert to normalized coordinates '''
    for perch in perches:
        # Convert Centroid coordinates
        y_centroid, x_centroid = int(perch['Centroid'][0]), int(perch['Centroid'][1])
        perch['Centroid']= intrinsic_to_world_grid[x_centroid, y_centroid]

        # Convert ConvexHull coordinates
        convex_hull_indices = perch['ConvexHull'].astype(int)
        perch['ConvexHull'] = intrinsic_to_world_grid[convex_hull_indices[:, 0], convex_hull_indices[:, 1]]

    for cache in caches:
        # Convert Centroid coordinates
        y_centroid, x_centroid = int(cache['Centroid'][0]), int(cache['Centroid'][1])
        cache['Centroid']= intrinsic_to_world_grid[x_centroid, y_centroid]

        # Convert ConvexHull coordinates
        convex_hull_indices = cache['ConvexHull'].astype(int)
        cache['ConvexHull'] = intrinsic_to_world_grid[convex_hull_indices[:, 0], convex_hull_indices[:, 1]]

    for feeder in feeders:
        # Convert Centroid coordinates
        y_centroid, x_centroid = int(feeder['Centroid'][0]), int(feeder['Centroid'][1])
        feeder['Centroid']= intrinsic_to_world_grid[x_centroid, y_centroid]

        # Convert ConvexHull coordinates
        convex_hull_indices = feeder['ConvexHull'].astype(int)
        feeder['ConvexHull'] = intrinsic_to_world_grid[convex_hull_indices[:, 0], convex_hull_indices[:, 1]]

    # Convert Centroid coordinates
    y_centroid, x_centroid = int(water_dish['Centroid'][0]), int(water_dish['Centroid'][1])
    water_dish['Centroid']= intrinsic_to_world_grid[x_centroid, y_centroid]

    # Convert ConvexHull coordinates
    convex_hull_indices = water_dish['ConvexHull'].astype(int)
    water_dish['ConvexHull'] = intrinsic_to_world_grid[convex_hull_indices[:, 0], convex_hull_indices[:, 1]]

    ''' Distinguish feeder perches from cache perches '''
    orientations = np.array([perch['Orientation'] for perch in perches])
    feeder_perches = np.abs(np.abs(orientations) - 45) < 2
    perch_w_site = [perches[i] for i in range(len(perches)) if not feeder_perches[i]]
    perch_no_site = [perches[i] for i in range(len(perches)) if feeder_perches[i]]
    assert len(perch_w_site) == n_caches, (
        f"error: {len(perch_w_site)} perches for {n_caches} cache sites!"
    )

    ''' Sort caches and perches so they are paired '''
    # Extract centroids
    perch_centers = np.array([perch['Centroid'] for perch in perch_w_site])
    cache_centers = np.array([cache['Centroid'] for cache in caches])

    # Compute histogram and define bin edges
    h = np.histogram(cache_centers[:, 0], bins=11)
    edges = h[1]  # Bin edges
    bin_edges = np.concatenate([edges[:5], [0], edges[7:]])  # Adjusting bin edges

    # Sort and match perches and caches column by column
    for col in range(len(bin_edges) - 1):
        # Find indices for perches and caches within the column range
        perch_idx = (perch_centers[:, 0] < bin_edges[col + 1]) & (perch_centers[:, 0] > bin_edges[col])
        cache_idx = (cache_centers[:, 0] < bin_edges[col + 1]) & (cache_centers[:, 0] > bin_edges[col])

        # Assert that the number of perches matches the number of caches
        assert np.sum(perch_idx) == np.sum(cache_idx), "Mismatch in perch-cache count!"

        # Sort perches by their Y-coordinates
        these_perches = [perch_w_site[i] for i in range(len(perch_w_site)) if perch_idx[i]]
        perch_centroids = np.array([perch['Centroid'] for perch in these_perches])
        sort_idx = np.argsort(perch_centroids[:, 1])
        for i, idx in enumerate(np.where(perch_idx)[0]):
            perch_w_site[idx] = these_perches[sort_idx[i]]

        # Sort caches by their Y-coordinates
        these_caches = [caches[i] for i in range(len(caches)) if cache_idx[i]]
        cache_centroids = np.array([cache['Centroid'] for cache in these_caches])
        sort_idx = np.argsort(cache_centroids[:, 1])
        for i, idx in enumerate(np.where(cache_idx)[0]):
            caches[idx] = these_caches[sort_idx[i]]

    # check for improper matches
    distances = np.sqrt(np.sum((perch_centers - cache_centers) ** 2, axis=1))
    improper_matches = np.sum(distances > 0.05)
    print(f"\n Found {improper_matches} improper perch-cache matches \n")

    ''' Match feeders to their closest perch '''
    perch_no_site_centers = np.array([perch['Centroid'] for perch in perch_no_site])
    feeders_centers = np.array([feeder['Centroid'] for feeder in feeders])

    # pairwise distances all feeders/perches
    distances = np.sqrt(np.sum((perch_no_site_centers[:, np.newaxis, :] - feeders_centers[:, np.newaxis, :])**2, axis=2))
    nearest_feeder_indices = np.argmin(distances, axis=1)

    # sort perches by their nearest feeder
    sorted_perch_indices = np.argsort(nearest_feeder_indices)
    perch_no_site = [perch_no_site[i] for i in sorted_perch_indices]

    # check for improper matches
    perch_no_site_centers = np.array([perch['Centroid'] for perch in perch_no_site])
    distances = np.sqrt(np.sum((perch_no_site_centers - feeders_centers) ** 2, axis=1))
    improper_matches = np.sum(distances > 0.2)
    print(f"\n Found {improper_matches} improper perch-feeder matches \n")


    ''' Plot everything to check the alignment '''
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    world_extent = np.concatenate(arena_ref['XWorldLimits'], arena_ref['YWorldLimits'])
    ax.imshow(arena_model, extent=world_extent)
    ax.set_title('Alignment Check')
    cache_colormap = cm.get_cmap('turbo', n_caches)
    feeder_colors = ['r', 'y', 'g', 'b']

    # Plot caches and cache perches
    caches_centroids = np.array([cache['Centroid'] for cache in caches])
    perch_w_site_centroids = np.array([perch['Centroid'] for perch in perch_w_site])
    for i in range(n_caches):
        ax.plot(caches_centroids[i, 0], caches_centroids[i, 1], '*', color=cache_colormap[i])
        ax.plot(perch_w_site_centroids[i, 0], perch_w_site_centroids[i, 1], '*', color=cache_colormap[i])

    # Plot feeders and feeder perches
    perch_no_site_centroids = np.array([perch['Centroid'] for perch in perch_no_site])
    feeder_centroids = np.array([feeder['Centroid'] for feeder in feeders])
    for i in range(n_feeders):
        ax.plot(perch_no_site_centroids[i, 0], perch_no_site_centroids[i, 1], '*', color=feeder_colors[i])
        ax.plot(feeder_centroids[i, 0], feeder_centroids[i, 1], '*', color=feeder_colors[i])

    # Plot water dish
    ax.plot(water_dish['Centroid'][0], water_dish['Centroid'][1], '*', color='k')

    plt.show()

    ''' Collect everything into a dictionary of arena items '''
    all_perches = perch_w_site + perch_no_site
    arena_items = {
        'all_perches': all_perches,
        'feeder_perches': perch_no_site,
        'cache_perches': perch_w_site,
        'caches': caches,
        'feeders': feeders,
        'water_dish': water_dish
    }
    return arena_items


''' Set paths, load arena image and reference info '''
arena_folder = '../data/arena/'
image_file = 'arena_model-01.png'
ref_file = 'ref_obj.mat'
print("\nLoading arena model image...")
arena_model = mpimg.imread(f'{arena_folder}{image_file}')
arena_ref = load_ref_obj(f'{arena_folder}{ref_file}');

''' Sort arena features by type and get their location info '''
arena_items = sort_arena_items(arena_model, arena_ref)
np.save(f'{arena_folder}arena_items.npy', arena_items, allow_pickle=True)