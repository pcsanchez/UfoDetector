import sys
import cv2
import numpy as np
import statistics

num_identifier = 0

def brightenImage(im):
    ny, nx, nc = im.shape
    for y in range(ny):
        for x in range(nx):
            b, g, r = im[y, x]
            if r == 0 and g == 0 and b == 0:
                # Background Pixel.
                continue
            elif r > 0 and g == 0 and b == 0:
                # Red Object.
                im[y, x] = (0, 0, 255)
            elif r == 0 and g > 0 and b == 0:
                # Green Object.
                im[y, x] = (0, 255, 0)
            elif r == 0 and g == 0 and b > 0:
                # Blue Object.
                im[y, x] = (255, 0 ,0)
            elif r > 0 and g > 0 and b == 0:
                # Yellow Object.
                im[y, x] = (0, 255, 255)
            elif r > 0 and g == 0 and b > 0:
                # Purple Object.
                im[y, x] = (255, 0, 255)
            elif r == 0 and g > 0 and b > 0:
                # Teal Object.
                im[y, x] = (255, 255, 0)
            elif r > 0 and g > 0 and b > 0:
                # White Object.
                im[y, x] = (255, 255, 255)
            else:
                print("Unexpected Pixel Color", r, g, b)
    return

def computeDistance(left_x, right_x):
    global im_shape
    xl = (left_x - im_shape[1] / 2) * 10e-6
    xr = (right_x - im_shape[1] / 2) * 10e-6
    # Focal length in meters.
    focal_length = 12
    # Baseline length in meters.
    baseline = 3500
    return (focal_length * baseline) / (xl - xr)

def mapColorToIdentificator(color):
    ident = ""
    b,g,r = color
    if b == 255 and g == 255 and r == 255:
        ident = "White"
    elif b == 255 and g == 0 and r == 0:
        ident = "Blue"
    elif b == 0 and g == 255 and r == 0:
        ident = "Green"
    elif b == 0 and g == 0 and r == 255:
        ident = "Red"
    elif b == 255 and g == 255 and r == 0:
        ident = "Teal"
    elif b == 255 and g == 0 and r == 255:
        ident = "Purple"
    elif b == 0 and g == 255 and r == 255:
        ident = "Yellow"
    else:
        ident = "Unknown"
    return ident

def getObjectColor(nx, ny, w, h, im):
    colors = []
    for y in range(ny, ny + h):
        for x in range(nx, nx + w):
            b, g, r = im[y, x]
            if not ((b == 0 and g == 0 and r == 0) or (b == 255 and g == 255 and r == 255)):
                colors.append(mapColorToIdentificator(im[y, x]))
    if len(colors) == 0:
        return "White"
    else:
        return statistics.mode(colors)

def validateDirection(prev_centroids, new_centroid):
    filtered_prev_centroids = list(filter(lambda cent: cent != -1, prev_centroids))
    if len(filtered_prev_centroids) < 2:
        return True
    else:
        past_x_displacement = filtered_prev_centroids[-1]['x'] - filtered_prev_centroids[-2]['x']
        past_y_displacement = filtered_prev_centroids[-1]['y'] - filtered_prev_centroids[-2]['y']
        new_x_displacement = new_centroid['x'] - filtered_prev_centroids[-1]['x']
        new_y_displacement = new_centroid['y'] - filtered_prev_centroids[-1]['y']
        if abs(new_x_displacement- past_x_displacement) > 20 or abs(new_y_displacement - past_y_displacement > 20):
            return False
        else:
            return True

### Process Centroids
# 1. Get the last centroid from all identified objects and sort them by color.
# 2. Get the current centroids based on contours and sort them by color.
# 3. For each color find the new position of the objetcs.
#       - If the lenghts of both lists is equal then just track by minimum distances.
#       - If the lenght of new centroids is larger then that means a new object was found with that color. Match the current previous based on distance and create new object.
#       - If the length of old centroids is larger then one object was lost. Match based on distance and fill the oher point with dummy data.
def processContours(new_contours, image_objects, frame, im):
    global num_identifier
    global n_frames
    # Get object centroids from previous frames.
    # Sample prev_centroids:
    # {"Yellow-1": {"x": 1, "y": 2}}
    prev_centroids = {obj_id: obj["locations"][frame - 1] for obj_id, obj in image_objects.items()}
    new_centroids = dict()
    # Get new centroids for new objects detected.
    # Sample new_centroids:
    # {"Yellow": [{"x": 1: "y": 2}]}
    for new_contour in new_contours:
        nx, ny, w, h = cv2.boundingRect(new_contour)
        curr_centroid = {"x": (nx + nx + w) // 2, "y": (ny + ny + h) // 2}
        object_color = getObjectColor(nx, ny, w, h, im)
        if object_color in new_centroids.keys():
            new_centroids[object_color].append(curr_centroid)
        else:
            new_centroids[object_color] = [curr_centroid]

    for obj_id, prev_centroid in prev_centroids.items():
        centroid_color = obj_id.split("-")[0]
        if centroid_color not in new_centroids.keys():
            candidate_objects = []
        else:
            candidate_objects = new_centroids[centroid_color]
        min_distance = 100000
        min_candidate_index = -1
        for index, candidate in enumerate(candidate_objects):
            distance = ((((candidate["x"] - prev_centroid["x"])** 2) + ((candidate["y"] - prev_centroid["y"])** 2))** 0.5)
            if distance < min_distance and validateDirection(image_objects[obj_id]["locations"], candidate):
                min_distance = distance
                min_candidate_index = index
        # A previous contour could not be matched in the new contours.
        if min_candidate_index == -1:
            image_objects[obj_id]["locations"][frame] = prev_centroid
        else:
            image_objects[obj_id]["locations"][frame] = candidate_objects[min_candidate_index]
            del candidate_objects[min_candidate_index]
    new_objects = {color: centroids for color, centroids in new_centroids.items() if len(centroids) > 0}
    for color in new_objects.keys():
        for centroid in new_objects[color]:
            new_object_id = color + "-" + str(num_identifier)
            new_object = dict(locations=[-1 for _ in range(n_frames)])
            num_identifier = num_identifier + 1
            new_object["locations"][frame] = centroid
            image_objects[new_object_id] = new_object

def findObjects(im, image_objects, frame):
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, im_binary = cv2.threshold(im_gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(im_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    processContours(contours, image_objects, frame, im)


def initialDetection(im, image_objects):
    global n_frames
    global num_identifier
    brightenImage(im)
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, im_binary = cv2.threshold(im_gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(im_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        nx, ny, w, h = cv2.boundingRect(contour)
        curr_center = {"x": (nx+nx+w)//2, "y": (ny+ny+h)//2}
        object_color = getObjectColor(nx, ny, w, h, im)
        new_object_id = object_color + "-" + str(num_identifier)
        new_object = dict(locations=[-1 for _ in range(n_frames)])
        num_identifier = num_identifier + 1
        new_object["locations"][0] = curr_center
        image_objects[new_object_id] = new_object

def fillObjectTragectories(image_objects):
    for k, v in image_objects.items():
        for i in range(len(v["locations"]) - 3, -1, -1):
            if v["locations"][i] == -1:
                x_disp = v["locations"][i + 2]['x'] - v["locations"][i + 1]['x']
                y_disp = v["locations"][i + 2]['y'] - v["locations"][i + 1]['y']
                v["locations"][i] = {'x': v["locations"][i + 1]['x'] - x_disp, 'y': v["locations"][i + 1]['y'] - y_disp}

def findDistances(left_image_objects, right_image_objects):
    object_distances = dict()
    for key in left_image_objects.keys():
        object_distances[key] = []
        for index, _ in enumerate(left_image_objects[key]['locations']):
            z = computeDistance(left_image_objects[key]["locations"][index]['x'], right_image_objects[key]["locations"][index]["x"])
            object_distances[key].append(z)
    return object_distances


def findUFOS(image_objects):
    ufos = []
    for object_id, object_info in image_objects.items():
        # print(object_info["locations"])
        first_point = object_info["locations"][0]
        last_point = object_info["locations"][-1]
        m = (last_point["y"] - first_point["y"]) / (last_point["x"] - first_point["x"])
        b = first_point["y"] - m * first_point["x"]
        for point in object_info["locations"]:
            point_in_line = m * point["x"] + b
            if abs(point_in_line - point["y"]) > 10:
                ufos.append(object_id)
                break
    return ufos

#-------------------------------------------
# Main Program
#-------------------------------------------
if len(sys.argv) < 4:
    print("Usage: ", sys.argv[0], "<numberOfFrames> <leftImageTemplate> <rightImageTemplate>...", file=sys.stderr)
    sys.exit(1)

n_frames = int(sys.argv[1])
left_image_objects = dict()
left_im = cv2.imread(sys.argv[2] % 0)
initialDetection(left_im, left_image_objects)
im_shape = left_im.shape
for frame in range(1,n_frames):
    fn_left = sys.argv[2] % frame
    left_im = cv2.imread(fn_left)
    brightenImage(left_im)
    findObjects(left_im, left_image_objects, frame)

num_identifier = 0
right_image_objects = dict()
right_im = cv2.imread(sys.argv[3] % 0)
initialDetection(right_im, right_image_objects)
for frame in range(1,n_frames):
    fn_right = sys.argv[3] % frame
    right_im = cv2.imread(fn_right)
    brightenImage(right_im)
    findObjects(right_im, right_image_objects, frame)

fillObjectTragectories(left_image_objects)
fillObjectTragectories(right_image_objects)

distances = findDistances(left_image_objects, right_image_objects)
ufos = findUFOS(right_image_objects)

print("Frame Identity Distance")
for frame in range(0, n_frames):
    for ident, key in distances.items():
        print(frame, ident, "{:.2e}".format(distances[ident][frame]))
print("UFO:", " ".join(ufos))
