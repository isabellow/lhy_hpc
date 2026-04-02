import numpy as np

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
