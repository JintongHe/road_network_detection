from skimage.morphology import skeletonize, remove_small_objects, remove_small_holes
import numpy as np
from matplotlib.pylab import plt
import cv2
import sknw
import os
import pandas as pd
from functools import partial
from itertools import tee
from scipy.spatial.distance import pdist, squareform
from scipy import ndimage as ndi
from collections import defaultdict, OrderedDict
import sys
from shapely.geometry import LineString, mapping
import json
from osgeo import gdal
from datetime import date
from multiprocessing import Pool
import math
import re
import warnings
warnings.filterwarnings("ignore")

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def remove_sequential_duplicates(seq):
    #todo
    res = [seq[0]]
    for elem in seq[1:]:
        if elem == res[-1]:
            continue
        res.append(elem)
    return res

def remove_duplicate_segments(seq):
    seq = remove_sequential_duplicates(seq)
    segments = set()
    split_seg = []
    res = []
    for idx, (s, e) in enumerate(pairwise(seq)):
        if (s, e) not in segments and (e, s) not in segments:
            segments.add((s, e))
            segments.add((e, s))
        else:
            split_seg.append(idx+1)
    for idx, v in enumerate(split_seg):
        if idx == 0:
            res.append(seq[:v])
        if idx == len(split_seg) - 1:
            res.append(seq[v:])
        else:
            s = seq[split_seg[idx-1]:v]
            if len(s) > 1:
                res.append(s)
    if not len(split_seg):
        res.append(seq)
    return res


def flatten(l):
    return [item for sublist in l for item in sublist]


def get_angle(p0, p1=np.array([0,0]), p2=None):
    """ compute angle (in degrees) for p0p1p2 corner
    Inputs:
        p0,p1,p2 - points in the form of [x,y]
    """
    if p2 is None:
        p2 = p1 + np.array([1, 0])
    v0 = np.array(p0) - np.array(p1)
    v1 = np.array(p2) - np.array(p1)

    angle = math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
    return np.degrees(angle)

def preprocess(img, thresh, ratio):
    img = (img > (255 * thresh)).astype(np.bool_)
    # img = cv2.dilate(img.astype(np.uint8), np.ones((int(5 * ratio), int(5 * ratio)))).astype(np.bool_)
    img = remove_small_objects(img, (10 * ratio)**2)  # Updated line
    img = remove_small_holes(img, (20 * ratio)**2)    # Assuming similar issue might exist here
    # img = cv2.dilate(img.astype(np.uint8), np.ones((7, 7)))
    return img


def graph2lines(G):
    node_lines = []
    edges = list(G.edges())
    if len(edges) < 1:
        return []
    prev_e = edges[0][1]
    current_line = list(edges[0])
    added_edges = {edges[0]}
    for s, e in edges[1:]:
        if (s, e) in added_edges:
            continue
        if s == prev_e:
            current_line.append(e)
        else:
            node_lines.append(current_line)
            current_line = [s, e]
        added_edges.add((s, e))
        prev_e = e
    if current_line:
        node_lines.append(current_line)
    return node_lines


def visualize(img, G, vertices):
    plt.imshow(img, cmap='gray')

    # draw edges by pts
    for (s, e) in G.edges():
        vals = flatten([[v] for v in G[s][e].values()])
        for val in vals:
            ps = val.get('pts', [])
            plt.plot(ps[:, 1], ps[:, 0], 'green', markersize=0.5)

    # draw node by o
    # node, nodes = G.node(), G.nodes
    # deg = G.degree
    # ps = np.array([node[i]['o'] for i in nodes])
    # ps = np.array(vertices)
    # plt.plot(ps[:, 1], ps[:, 0], 'r.', markersize=0.5)

    # title and show
    plt.title('Build Graph')
    plt.savefig('test.png', dpi=300)
    print('save debug image successfully')

def line_points_dist(line1, pts):
    return np.cross(line1[1] - line1[0], pts - line1[0]) / np.linalg.norm(line1[1] - line1[0])

# def remove_small_terminal(G):
#     deg = G.degree()
#     # terminal_points = [i for i, d in deg.items() if d == 1]
#     terminal_points = [i for i, d in G.degree() if d == 1]
#     edges = list(G.edges())
#     for s, e in edges:
#         if s == e:
#             sum_len = 0
#             vals = flatten([[v] for v in G[s][s].values()])
#             for ix, val in enumerate(vals):
#                 sum_len += len(val['pts'])
#             if sum_len < 3:
#                 G.remove_edge(s, e)
#                 continue
#         vals = flatten([[v] for v in G[s][e].values()])
#         for ix, val in enumerate(vals):
#             if s in terminal_points and val.get('weight', 0) < 10:
#                 G.remove_node(s)
#             if e in terminal_points and val.get('weight', 0) < 10:
#                 G.remove_node(e)
#     return
import networkx as nx


def remove_small_terminal(G, ratio):
    # First, identify terminal points to avoid repeated degree calculations
    terminal_points = {i for i, d in G.degree() if d == 1}

    # Check for self-loops and remove if necessary
    for node in list(G.nodes):
        if G.has_edge(node, node):
            sum_len = sum(len(val['pts']) for val in G[node][node].values())
            if sum_len < 25 * ratio:
                G.remove_edge(node, node)

    # Now check terminal points and their connected edges
    for t in list(terminal_points):
        # Ensure the node still exists before proceeding
        if t not in G:
            continue
        for neighbor in G[t]:
            weight = G[t][neighbor][0].get('weight', 0)  # Assuming single-edge between nodes
            if weight < 25 * ratio:
                G.remove_node(t)
                break  # Exit the loop as the node has been removed

    return G

linestring = "LINESTRING {}"
def make_skeleton(root, fn, debug, fix_borders, ratio):
    replicate = 5
    clip = 2
    rec = replicate + clip
    # open and skeletonize
    img = cv2.imread(os.path.join(root, fn), cv2.IMREAD_GRAYSCALE)
    # assert img.shape == (1300, 1300)
    if fix_borders:
        img = cv2.copyMakeBorder(img, replicate, replicate, replicate, replicate, cv2.BORDER_REPLICATE)
    img_copy = None
    if debug:
        if fix_borders:
            img_copy = np.copy(img[replicate:-replicate,replicate:-replicate])
        else:
            img_copy = np.copy(img)
    thresh = 0.5# threshes[fn[2]]
    img = preprocess(img, thresh, ratio)
    if not np.any(img):
        return None, None
    ske = skeletonize(img).astype(np.uint16)
    if fix_borders:
        ske = ske[rec:-rec, rec:-rec]
        ske = cv2.copyMakeBorder(ske, clip, clip, clip, clip, cv2.BORDER_CONSTANT, value=0)
    return img_copy, ske


def add_small_segments(G, terminal_points, terminal_lines, ratio):
    node = G.nodes
    term = [node[t]['o'] for t in terminal_points]
    dists = squareform(pdist(term))
    possible = np.argwhere((dists > 0 * ratio) & (dists < 50 * ratio))
    good_pairs = []
    for s, e in possible:
        if s > e:
            continue
        s, e = terminal_points[s], terminal_points[e]

        if G.has_edge(s, e):
            continue
        good_pairs.append((s, e))

    possible2 = np.argwhere((dists > 50 * ratio) & (dists < 100 * ratio))
    for s, e in possible2:
        if s > e:
            continue
        s, e = terminal_points[s], terminal_points[e]
        if G.has_edge(s, e):
            continue
        l1 = terminal_lines[s]
        l2 = terminal_lines[e]
        d = line_points_dist(l1, l2[0])

        if abs(d) > 50 * ratio:
            continue
        angle = get_angle(l1[1] - l1[0], np.array((0, 0)), l2[1] - l2[0])
        if -20 < angle < 20 or angle < -160 or angle > 160:
            good_pairs.append((s, e))

    dists = {}
    for s, e in good_pairs:
        s_d, e_d = [G.nodes[s]['o'], G.nodes[e]['o']]
        dists[(s, e)] = np.linalg.norm(s_d - e_d)

    dists = OrderedDict(sorted(dists.items(), key=lambda x: x[1]))

    wkt = []
    added = set()
    for s, e in dists.keys():
        if s not in added and e not in added:
            added.add(s)
            added.add(e)
            s_d, e_d = G.nodes[s]['o'], G.nodes[e]['o']
            line_strings = ["{1:.1f} {0:.1f}".format(*c.tolist()) for c in [s_d, e_d]]
            line = '(' + ", ".join(line_strings) + ')'
            wkt.append(linestring.format(line))
    return wkt


def add_direction_change_nodes(pts, s, e, s_coord, e_coord):
    if len(pts) > 3:
        ps = pts.reshape(pts.shape[0], 1, 2).astype(np.int32)
        approx = 2
        ps = cv2.approxPolyDP(ps, approx, False)
        ps = np.squeeze(ps, 1)
        st_dist = np.linalg.norm(ps[0] - s_coord)
        en_dist = np.linalg.norm(ps[-1] - s_coord)
        if st_dist > en_dist:
            s, e = e, s
            s_coord, e_coord = e_coord, s_coord
        ps[0] = s_coord
        ps[-1] = e_coord
    else:
        ps = np.array([s_coord, e_coord], dtype=np.int32)
    return ps

def pixel_2_meter(img_path):
    # Open the raster file using GDAL
    ds = gdal.Open(img_path)

    # Get raster size (width and height)
    width = ds.RasterXSize
    height = ds.RasterYSize

    # Get georeferencing information
    geoTransform = ds.GetGeoTransform()
    pixel_size_x = geoTransform[1]  # Pixel width
    pixel_size_y = abs(geoTransform[5])  # Pixel height (absolute value)

    # Get the top latitude from the geotransform and the height
    # geoTransform[3] is the top left y, which gives the latitude
    latitude = geoTransform[3] - pixel_size_y * height
    # Close the dataset
    ds = None

    # Convert road width from meters to pixels
    # road_width_meters = line_width
    meters_per_degree = 111139 * math.cos(math.radians(latitude))
    thickness_pixels_ratio = 1 / (pixel_size_x * meters_per_degree)
    return thickness_pixels_ratio

def parse_linestring(linestring):
    points_str = linestring.strip()[len('LINESTRING ('):-1]
    points = [tuple(map(int, map(float, point.split(' ')))) for point in points_str.split(', ')]
    return points


def find_connected_linestrings(linestrings):
    # Parse all linestrings to get their start and end points
    points = [parse_linestring(ls) for ls in linestrings]
    flags = [1] * len(points)
    for i, line1 in enumerate(points):
        for j in range(i + 1, len(points)):
            line2 = points[j]
            if line1[0] == line2[0]:
                points[j] = points[j][::-1] + line1[1:]
                flags[i] = 0
                break
            elif line1[0] == line2[-1]:
                points[j] = points[j] + line1[1:]
                flags[i] = 0
                break
            elif line1[-1] == line2[0]:
                points[j] = line1 + points[j][1:]
                flags[i] = 0
                break
            elif line1[-1] == line2[-1]:
                points[j] = points[j] + line1[::-1][1:]
                flags[i] = 0
                break
            else:
                continue
    points = [points[i] for i in range(len(points)) if flags[i]]
    merged_linestrings = ["LINESTRING (" + ", ".join(f"{x} {y}" for x, y in ls) + ")" for ls in points]
    return merged_linestrings

def linestring_length(points):
    length = 0
    for j in range(1, len(points)):
        # Calculate the Euclidean distance between points[i] and points[i-1]
        length += math.sqrt((points[j][0] - points[j - 1][0]) ** 2 + (points[j][1] - points[j - 1][1]) ** 2)
    return length
def remove_short_linestring(linestrings, ratio, threshold=200):
    flags = [1] * len(linestrings)
    for i, linestring in enumerate(linestrings):
        points = parse_linestring(linestring)
        length = linestring_length((points))
        if length < threshold * ratio:
            found = False
            for point in points:
                for j, linestring2 in enumerate(linestrings):
                    if j == i:
                        continue
                    linestring2 = set(parse_linestring(linestring2))
                    if point in linestring2:
                        found = True
                        break
            if not found:
                flags[i] = 0
    new_linestrings = [linestrings[i] for i in range(len(linestrings)) if flags[i]]
    return new_linestrings

def build_graph(root, img_root_dir, fn, debug=False, threshes={'2': .3, '3': .3, '4': .3, '5': .2}, add_small=True, fix_borders=True):
    city = os.path.splitext(fn)[0]
    image_path = os.path.join(img_root_dir, city + '.tif')
    ratio = pixel_2_meter(image_path)
    img_copy, ske = make_skeleton(root, fn, debug, fix_borders, ratio)
    if ske is None:
        return city, [linestring.format("EMPTY")]
    G = sknw.build_sknw(ske, multi=True)
    remove_small_terminal(G, ratio)
    node_lines = graph2lines(G)
    if not node_lines:
        return city, [linestring.format("EMPTY")]
    node = G.nodes
    deg = G.degree()
    wkt = []
    terminal_points = [i for i, d in G.degree() if d == 1]

    terminal_lines = {}
    vertices = []
    for w in node_lines:
        coord_list = []
        additional_paths = []
        for s, e in pairwise(w):
            vals = flatten([[v] for v in G[s][e].values()])
            for ix, val in enumerate(vals):
                s_coord, e_coord = node[s]['o'], node[e]['o']
                pts = val.get('pts', [])
                if s in terminal_points:
                    terminal_lines[s] = (s_coord, e_coord)
                if e in terminal_points:
                    terminal_lines[e] = (e_coord, s_coord)

                ps = add_direction_change_nodes(pts, s, e, s_coord, e_coord)

                if len(ps.shape) < 2 or len(ps) < 2:
                    continue

                if len(ps) == 2 and np.all(ps[0] == ps[1]):
                    continue

                line_strings = ["{1:.1f} {0:.1f}".format(*c.tolist()) for c in ps]
                if ix == 0:
                    coord_list.extend(line_strings)
                else:
                    additional_paths.append(line_strings)

                vertices.append(ps)
        if not len(coord_list):
            continue

        segments = remove_duplicate_segments(coord_list)
        for coord_list in segments:
            if len(coord_list) > 1:
                line = '(' + ", ".join(coord_list) + ')'
                wkt.append(linestring.format(line))
        for line_strings in additional_paths:
            line = ", ".join(line_strings)
            line_rev = ", ".join(reversed(line_strings))
            for s in wkt:
                if line in s or line_rev in s:
                    break
            else:
                wkt.append(linestring.format('(' + line + ')'))
    if add_small and len(terminal_points) > 1:
        wkt.extend(add_small_segments(G, terminal_points, terminal_lines, ratio))
    # print(wkt)
    wkt = find_connected_linestrings(wkt)

    wkt = remove_short_linestring(wkt, ratio, threshold=400)
    # print(wkt)
    if debug:
        vertices = flatten(vertices)
        visualize(img_copy, G, vertices)

    if not wkt:
        return city, [linestring.format("EMPTY")]
    return city, wkt

def string_to_linestring(line_str):
    # Extract the coordinates using regular expression
    coords = re.findall(r'\d+\.?\d*', line_str)
    # Group them in pairs (tuples) and convert to float
    points = [(float(coords[i]), float(coords[i + 1])) for i in range(0, len(coords), 2)]
    # Create a LineString object from these points
    return LineString(points)

def create_geojson(linestrings, image_path, out_file_path):
    if linestrings[0] == 'LINESTRING EMPTY':
        return
    # Use os.path to handle file paths
    image_name = os.path.basename(image_path)
    suf = image_name.split('.')[1]
    show_path = os.path.join(out_file_path, image_name.replace('.' + suf, '_predict.geojson'))

    gdal.AllRegister()
    dataset = gdal.Open(image_path)
    adfGeoTransform = dataset.GetGeoTransform()

    res_dict = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": []
    }
    today = date.today().isoformat()
    linestrings = [string_to_linestring(line) for line in linestrings]
    for linestring in linestrings:
        line = mapping(linestring)

        name = '道路'
        feature = {
            "type": "Feature",
            "properties": {
                "Id": 0,
                "name": name,
                "date": today,
                "area": 0.0,
                "label": 0,
                "result": 1,
                "XMMC": "",
                "HYMC": "",
                "weight": 0,
                "bz": 0
            },
            "geometry": {"type": "LineString", "coordinates": []}
        }
        # Convert coordinates
        coordinates_list = list(line['coordinates'])
        for i, xy in enumerate(coordinates_list):
            location = (int(xy[0]) * adfGeoTransform[1] + adfGeoTransform[0],
                        int(xy[1]) * adfGeoTransform[5] + adfGeoTransform[3])

            coordinates_list[i] = location
        line['coordinates'] = tuple(coordinates_list)
        feature['geometry'] = line
        res_dict['features'].append(feature)

    # Use json.dumps to create a JSON string
    res = json.dumps(res_dict, ensure_ascii=False)

    # Write the JSON string to a file
    with open(show_path, 'w', encoding='utf8') as out_file:
        out_file.write(res)

if __name__ == "__main__":
    # prefix = 'AOI'
    # results_root = r'/results/results'
    # # results_root = r'd:\tmp\roads\albu\results\results'
    # # root = os.path.join(results_root, r'results\2m_4fold_512_30e_d0.2_g0.2')
    # root = os.path.join(results_root, r'2m_4fold_512_30e_d0.2_g0.2_test', 'merged')
    # f = partial(build_graph, root)
    # l = [v for v in os.listdir(root) if prefix in v]
    # l = list(sorted(l))
    # with Pool() as p:
    #     data = p.map(f, l)
    # all_data = []
    # for k, v in data:
    #     for val in v:
    #         all_data.append((k, val))
    # df = pd.DataFrame(all_data, columns=['ImageId', 'WKT_Pix'])
    # df.to_csv(sys.argv[1] + '.txt', index=False)
    # Specify the directory where your grayscale images are stored

    mask_dir = r"/home/zkxq/project/hjt/smp/out"
    image_dir = r"/home/zkxq/project/hjt/smp/datasets/Wind_Turbine/test/test_img"
    # Define the file prefix or suffix that your images have, if any
    # For example, if your images are named like "AOI_1_grayscale.png",
    # you would keep 'AOI' as the prefix.
    image_prefix = 'AOI'  # Change this to match your file naming pattern

    # Create a partial function that includes the directory as an argument
    f = partial(build_graph, mask_dir, image_dir)

    # List all files in the directory that match the prefix condition
    image_files = [filename for filename in os.listdir(mask_dir) if filename.lower().endswith('.tif')]

    # Sort the list of files if necessary
    image_files = sorted(image_files)

    # Use multiprocessing to process all images in parallel
    with Pool() as p:
        data = p.map(f, image_files)

    # Compile all data into a list
    all_data = []
    out_file_path = "/home/zkxq/project/hjt/smp/geojson"
    for k, v in data:
        image_path = os.path.join(image_dir, k + '.tif')
        create_geojson(linestrings=v, image_path=image_path, out_file_path=out_file_path)
        for val in v:
            all_data.append((k, val))

    # Create a DataFrame from the compiled data
    df = pd.DataFrame(all_data, columns=['ImageId', 'WKT_Pix'])

    # Define the output CSV file name
    output_csv_file = '/home/zkxq/project/hjt/smp/datasets/out'  # Change this to your desired file name

    # Save the DataFrame to a CSV file
    df.to_csv(output_csv_file, index=False)

    # # Specify the root directory where your image files are located
    # root = "/home/zkxq/project/hjt/smp/out"
    #
    # # Specify the name of the single image file you want to process
    # single_image_file = 'access_road_zfengji39_20210325.tif'
    #
    # # Call the build_graph function directly with the path to your single image file
    # city, wkt = build_graph(root, single_image_file, debug=False)
    #
    # # If you want to print the output or do something with it, you can do that here
    # # For example, print the wkt data:
    # print(f"City: {city}")
    # for line in wkt:
    #     print(line)
    # # Compile all data into a list
    # all_data = []
    # out_file_path = "/home/zkxq/project/hjt/smp/geojson"
    # image_path = "/home/zkxq/project/hjt/smp/test_images/access_road_zfengji39_20210325.tif"
    # create_geojson(linestrings=wkt, image_path=image_path, out_file_path=out_file_path)
    #
    # # # If you want to save the output to a CSV file
    # # all_data = [(city, val) for val in wkt]
    # # df = pd.DataFrame(all_data, columns=['ImageId', 'WKT_Pix'])
    # #
    # # # Define the output CSV file name
    # # output_csv_file = 'output_filename.csv'  # Change this to your desired file name
    # #
    # # # Save the DataFrame to a CSV file
    # # df.to_csv(output_csv_file, index=False)