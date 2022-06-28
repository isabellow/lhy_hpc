def get_rainbow_colors(n_colors):
    '''
    based on https://personal.sron.nl/~pault/#fig:scheme_rainbow_discrete
    chooses rainbow colors that are colorblind friendly
    based on the number of desired colors
    '''
    # full list of 28 colors
    all_colors = np.asarray([[209, 187, 215, 255],
                              [174, 118, 163, 255], 
                              [136, 46, 114, 255],
                              [25, 101, 176, 255],
                              [82, 137, 199, 255], 
                              [123, 175, 222, 255],
                              [78, 178, 101, 255],
                              [144, 201, 135, 255],
                              [202, 224, 171, 255],
                              [247, 240, 86, 255],
                              [244, 167, 54, 255],
                              [232, 96, 28, 255],
                              [220, 5, 12, 255]
                             ])/255
    all_colors = list(all_colors)
    for i, c in enumerate(all_colors):
        all_colors[i] = tuple(c)

    # which combos to take for different numbers of colors
