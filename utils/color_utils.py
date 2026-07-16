import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def hex_to_rgb(hex_code):
    # Remove the '#' if it's present
    hex_code = hex_code.lstrip('#')
    
    # Check if the hex code is 3 digits long (shorthand) and expand it
    if len(hex_code) == 3:
        hex_code = ''.join([c*2 for c in hex_code])

    # Convert each 2-character slice to an integer with base 16
    r = int(hex_code[0:2], 16)
    g = int(hex_code[2:4], 16)
    b = int(hex_code[4:6], 16)
    
    return (r, g, b)


def get_bird_colors_da(bird_ids, adjust_colors=True):
    '''
    Given a list of bird IDs, return a list of colors matched to each bird

    Optionally adjust lighter colors for visualization purposes (e.g. for line plots)

    Colors are from the Banditry 2.0 params
    https://sites.google.com/view/banditry/settings/parameters
    '''
    bird_colors = {
        'RBY': '#cc0000',
        'AMB': '#ff6600',
        'LMN': '#ffff00',
        'EMR': '#006600',
        'LIM': '#99ff66',
        'TRQ': '#006666',
        'SPP': '#66ccff',
        'IND': '#000099',
        'LVN': '#cc99ff',
        'ROS': '#ff9999', 
        'CHC': '#663300',
        'ONX': '#000000',
        'SLV': '#999999',
        'PRL': '#ffffff'
    }

    these_colors = []
    for bird in bird_ids:
        if (bird[:3] == 'PRL') & (adjust_colors):
            these_colors.append(np.asarray([232, 236, 251, 255])/255)
        elif (bird[:3] == 'LMN') & (adjust_colors):
            these_colors.append(np.asarray([247, 203, 69, 255])/255)
        elif (bird[:3] == 'LIM') & (adjust_colors):
            these_colors.append(np.asarray([78, 178, 101, 255])/255)
        elif bird[:3] in bird_colors.keys():
            color = hex_to_rgb(bird_colors[bird[:3]])
            these_colors.append(np.asarray(color)/255)
        else:
            print(f'{bird[:3]} is not a known band color')

    return these_colors


def get_bird_colors_tol(bird_ids):
    '''
    Given a list of bird IDs, return a list of colors matched to each bird

    Colors are from Paul Tol and are colorblind friendly
    https://sronpersonalpages.nl/~pault/
    '''
    bird_colors = {
        'RBY': [220, 5, 12, 255],
        'AMB': [238, 128, 38, 255],
        'LMN': [247, 203, 69, 255],
        'EMR': [78, 178, 101, 255],
        'LIM': [144, 201, 135, 255],
        'TRQ': [84, 158, 179, 255],
        'SPP': [123, 175, 222, 255],
        'IND': [25, 101, 176, 255],
        'LVN': [174, 118, 163, 255],
        'ROS': [209, 187, 215, 255], # this is not really pink...
        'CHC': [66, 21, 10, 255],
        'ONX': 'k',
        'SLV': [119, 119, 119, 255],
        'PRL': [232, 236, 251, 255]
    }

    these_colors = []
    for bird in bird_ids:
        if bird[:3] in bird_colors.keys():
            these_colors.append(np.asarray(bird_colors[bird[:3]])/255)
        else:
            print(f'{bird[:3]} is not a known band color')

    return these_colors


def parula_colormap(n: int = 256) -> LinearSegmentedColormap:
    """
    This function written by Claude - use with caution

    Returns a matplotlib colormap identical to MATLAB's Parula colormap.

    Parula is a perceptually uniform colormap introduced in MATLAB R2014b.
    It transitions from dark blue → teal → green → yellow.

    Parameters
    ----------
    n : int
        Number of discrete colour levels in the colormap (default 256).

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        A colormap object ready to pass to any matplotlib function that
        accepts a `cmap` argument (e.g. imshow, pcolormesh, scatter).

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> cmap = parula_colormap()
    >>> plt.imshow(np.random.rand(10, 10), cmap=cmap)
    >>> plt.colorbar()
    >>> plt.show()
    """

    # Full 64-point RGB data extracted from MATLAB R2014b+ by the BIDS project
    # Values are in [0, 1].
    _parula_data = [
        [0.2081,       0.1663,       0.5292      ],
        [0.2116238095, 0.1897809524, 0.5776761905],
        [0.212252381,  0.2137714286, 0.6269714286],
        [0.2081,       0.2386,       0.6770857143],
        [0.1959047619, 0.2644571429, 0.7279      ],
        [0.1707285714, 0.2919380952, 0.779247619 ],
        [0.1252714286, 0.3242428571, 0.8302714286],
        [0.0591333333, 0.3598333333, 0.8683333333],
        [0.0116952381, 0.3875095238, 0.8819571429],
        [0.0059571429, 0.4086142857, 0.8828428571],
        [0.0165142857, 0.4266,       0.8786333333],
        [0.032852381,  0.4430428571, 0.8719571429],
        [0.0498142857, 0.4585714286, 0.8640571429],
        [0.0629333333, 0.4736904762, 0.8554380952],
        [0.0722666667, 0.4886666667, 0.8467      ],
        [0.0779428571, 0.5039857143, 0.8383714286],
        [0.079347619,  0.5200238095, 0.8311809524],
        [0.0749428571, 0.5375428571, 0.8262714286],
        [0.0640571429, 0.5569857143, 0.8239571429],
        [0.0487714286, 0.5772238095, 0.8228285714],
        [0.0343428571, 0.5965809524, 0.819852381 ],
        [0.0265,       0.6137,       0.8135      ],
        [0.0238904762, 0.6286619048, 0.8037619048],
        [0.0230904762, 0.6417857143, 0.7912666667],
        [0.0227714286, 0.6534857143, 0.7767571429],
        [0.0266619048, 0.6641952381, 0.7607190476],
        [0.0383714286, 0.6742714286, 0.743552381 ],
        [0.0589714286, 0.6837571429, 0.7253857143],
        [0.0843,       0.6928333333, 0.7061666667],
        [0.1132952381, 0.7015,       0.6858571429],
        [0.1452714286, 0.7097571429, 0.6646285714],
        [0.1801333333, 0.7176571429, 0.6424333333],
        [0.2178285714, 0.7250428571, 0.6192619048],
        [0.2586428571, 0.7317142857, 0.5954285714],
        [0.3021714286, 0.7376047619, 0.5711857143],
        [0.3481666667, 0.7424333333, 0.5472666667],
        [0.3952571429, 0.7459,       0.5244428571],
        [0.4420095238, 0.7480809524, 0.5033142857],
        [0.4871238095, 0.7490619048, 0.4839761905],
        [0.5300285714, 0.7491142857, 0.4661142857],
        [0.5708571429, 0.7485190476, 0.4493904762],
        [0.609852381,  0.7473142857, 0.4336857143],
        [0.6473,       0.7456,       0.4188      ],
        [0.6834190476, 0.7434761905, 0.4044333333],
        [0.7184095238, 0.7411333333, 0.3904761905],
        [0.7524857143, 0.7384,       0.3768142857],
        [0.7858428571, 0.7355666667, 0.3632714286],
        [0.8185047619, 0.7327333333, 0.3497904762],
        [0.8506571429, 0.7299,       0.3360285714],
        [0.8824333333, 0.7274333333, 0.3217      ],
        [0.9139333333, 0.7257857143, 0.3062761905],
        [0.9449571429, 0.7261142857, 0.2886428571],
        [0.9738952381, 0.7313952381, 0.266647619 ],
        [0.9937714286, 0.7454571429, 0.240347619 ],
        [0.9990428571, 0.7653142857, 0.2164142857],
        [0.9955333333, 0.7860571429, 0.196652381 ],
        [0.988,        0.8066,       0.1793666667],
        [0.9788571429, 0.8271428571, 0.1633142857],
        [0.9697,       0.8481380952, 0.147452381 ],
        [0.9625857143, 0.8705142857, 0.1309      ],
        [0.9588714286, 0.8949,       0.1132428571],
        [0.9598238095, 0.9218333333, 0.0948380952],
        [0.9661,       0.9514428571, 0.0755333333],
        [0.9763,       0.9831,       0.0538      ],
    ]

    data = np.array(_parula_data)
    return LinearSegmentedColormap.from_list("parula", data, N=n)






# todo in progress
def get_rainbow_colors(n_colors):
    '''
    based on https://sronpersonalpages.nl/~pault/
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
