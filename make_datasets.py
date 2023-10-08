import numpy as np

def create_dataset_reds(image):
    height, width = image[0].shape
    image_r = image[0]/255.0
    # get the differences between the pixel and the pixel above
    r_top_diffs = np.diff(image_r, axis=0)
    # pad the top row with zeros
    r_top_diffs = np.pad(r_top_diffs, ((1, 0), (0, 0)), 'constant')
    # get the differences between the pixel and the pixel to the left
    r_left_diffs = np.diff(image_r, axis=1)
    # pad the left column with zeros
    r_left_diffs = np.pad(r_left_diffs, ((0, 0), (1, 0)), 'constant')
    data = []
    targets = []
    for y in range(1, height):
        for x in range(1, width):
            features = []
            # get red features
            top = image_r[y - 1, x]
            left = image_r[y, x - 1]
            a = r_top_diffs[y, x-1]  # left - diagonal
            b = r_left_diffs[y-1, x]  # top - diagonal
            c = r_left_diffs[y, x-1]  # left - left_left
            d = r_top_diffs[y-1, x]  # top - top_top
            r = image_r[y, x]
            features.extend([a, b, c, d, top, left])
            target = (r)
            data.append(features)
            targets.append(target)
    return np.array(data), np.array(targets)


def create_dataset_greens(image):
    height, width = image[0].shape
    image_r = image[0]/255.0
    image_g = image[1]/255.0
    # get the differences between the pixel and the pixel above
    r_top_diffs = np.diff(image_r, axis=0)
    # pad the top row with zeros
    r_top_diffs = np.pad(r_top_diffs, ((1, 0), (0, 0)), 'constant')
    # get the differences between the pixel and the pixel to the left
    r_left_diffs = np.diff(image_r, axis=1)
    # pad the left column with zeros
    r_left_diffs = np.pad(r_left_diffs, ((0, 0), (1, 0)), 'constant')
    g_top_diffs = np.diff(image_g, axis=0)
    # pad the top row with zeros
    g_top_diffs = np.pad(g_top_diffs, ((1, 0), (0, 0)), 'constant')
    # get the differences between the pixel and the pixel to the left
    g_left_diffs = np.diff(image_g, axis=1)
    # pad the left column with zeros
    g_left_diffs = np.pad(g_left_diffs, ((0, 0), (1, 0)), 'constant')
    data = []
    targets = []
    for y in range(1, height):
        for x in range(1, width):
            features = []
            # get red features
            top = image_r[y - 1, x]
            left = image_r[y, x - 1]
            a = r_top_diffs[y, x-1]
            b = r_left_diffs[y-1, x]
            r = image_r[y, x]
            # get green features
            a1 = g_top_diffs[y, x - 1]
            b1 = g_left_diffs[y - 1, x]
            c1 = g_left_diffs[y, x - 1]
            d1 = g_top_diffs[y - 1, x]
            g_left = image_g[y, x - 1]
            g_top = image_g[y - 1, x]
            g = image_g[y, x]
            features.extend([a, b, top, left, g_left, g_top,
                             a1, b1, c1, d1, r])
            target = (g)
            data.append(features)
            targets.append(target)
    return np.array(data), np.array(targets)


def create_dataset_blues(image):
    height, width = image[0].shape
    image_r = image[0]/255.0
    image_g = image[1]/255.0
    image_b = image[2]/255.0
    # get the differences between the pixel and the pixel above
    r_top_diffs = np.diff(image_r, axis=0)
    # pad the top row with zeros
    r_top_diffs = np.pad(r_top_diffs, ((1, 0), (0, 0)), 'constant')
    # get the differences between the pixel and the pixel to the left
    r_left_diffs = np.diff(image_r, axis=1)
    # pad the left column with zeros
    r_left_diffs = np.pad(r_left_diffs, ((0, 0), (1, 0)), 'constant')
    # get the differences between the pixel and the pixel above
    g_top_diffs = np.diff(image_g, axis=0)
    # pad the top row with zeros
    g_top_diffs = np.pad(g_top_diffs, ((1, 0), (0, 0)), 'constant')
    # get the differences between the pixel and the pixel to the left
    g_left_diffs = np.diff(image_g, axis=1)
    # pad the left column with zeros
    g_left_diffs = np.pad(g_left_diffs, ((0, 0), (1, 0)), 'constant')
    # get the differences between the pixel and the pixel above
    b_top_diffs = np.diff(image_b, axis=0)
    # pad the top row with zeros
    b_top_diffs = np.pad(b_top_diffs, ((1, 0), (0, 0)), 'constant')
    # get the differences between the pixel and the pixel to the left
    b_left_diffs = np.diff(image_b, axis=1)
    # pad the left column with zeros
    b_left_diffs = np.pad(b_left_diffs, ((0, 0), (1, 0)), 'constant')
    data = []
    targets = []
    for y in range(1, height):
        for x in range(1, width):
            features = []
            # get red features
            top = image_r[y - 1, x]
            left = image_r[y, x - 1]
            a = r_top_diffs[y, x-1]
            b = r_left_diffs[y-1, x]
            r = image_r[y, x]
            # get green features
            a1 = g_top_diffs[y, x - 1]
            b1 = g_left_diffs[y - 1, x]
            g_left = image_g[y, x - 1]
            g_top = image_g[y - 1, x]
            g = image_g[y, x]
            # get blue features
            a2 = b_top_diffs[y, x - 1]
            b2 = b_left_diffs[y - 1, x]
            c2 = b_left_diffs[y, x - 1]
            d2 = b_top_diffs[y - 1, x]
            b_left = image_b[y, x - 1]
            b_top = image_b[y - 1, x]
            blue = image_b[y, x]
            features.extend([a, b, top, left, g_left, g_top,
                             a1, b1,  r, a2, b2, c2, d2, b_left, b_top, g])
            target = (blue)
            data.append(features)
            targets.append(target)
    return np.array(data), np.array(targets)

