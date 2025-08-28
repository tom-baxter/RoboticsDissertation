import serial
import cv2
import os
import sys
import math
import time
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(1)  # 0 = default webcam, 1 = external webcam

# Open serial connection for Arduino
arduino = serial.Serial(port='COM7', baudrate=9600, timeout=0.2)
time.sleep(2)  # allow Arduino to reset
arduino.reset_input_buffer()

# Open serial connection for OpenRB150
motor = serial.Serial(port='COM5', baudrate=57600, timeout=0.2)
time.sleep(2)  # wait for connection

tip_points_cm = []
bev_angles_data = []
target_found_data = []

# active = True
# filenameSave = "A1.txt"

# active = False
# filenameSave = "C1.txt"

# active = True
# filenameSave = "A2.txt"

active = False
filenameSave = "C2.txt"

# ==== Globals (cache the figure, axes, and artists) ====
_fig = None
_ax_top = _ax_bottom = None
_sc_top = _sc_bot = None          # tip points (scatter)
_tgt_top = _tgt_bot = None        # target markers (scatter)
_tip_back_top = _tip_fwd_top = None   # tip lines (top view)
_tip_back_bot = _tip_fwd_bot = None   # tip lines (side view)

def log_tip_point(tip_x, tip_y, us_roll, s_view):
    global tip_points_cm
    global bev_angles_data
    global target_found_data
    """Log the tip point in centimeters."""
    roll_rad = math.radians(us_roll)
    s_view_height = s_view.shape[0]
    r_pixels = s_view_height - tip_y
    r_cm = (r_pixels / 28.5) + 1
    x_cm = r_cm * math.sin(roll_rad)
    y_cm = r_cm * math.cos(roll_rad)
    z_cm = (tip_x - 56)/28.5
    tip_points_cm.append([x_cm, y_cm , z_cm])

def find_3d_tip_vector():
    tip_max_length = 2 # max amount of needle used to calculate tip vector
    min_tip_length = 1 # min amount of needle needed to produce a tip vector

    if not tip_points_cm:
        return False, [0, 0, 0], [0, 0, 0]
    tip_x_cm, tip_y_cm, tip_z_cm = tip_points_cm[-1]
    z_min = 999
    z_max = 0
    #check to see if we have suffiect data 
    #return false if not
    #if we do, find tip vector and return
    for point in tip_points_cm:
        if point[2] < z_min:
            z_min = point[2]
        if point[2] > z_max:
            z_max = point[2]
    if z_max - z_min < min_tip_length: # if segment not long enough
        return False, [tip_x_cm, tip_y_cm, tip_z_cm], [0, 0, 0]  
    # otherwise, continue
    filtered_points = [p for p in tip_points_cm if p[2] > (tip_z_cm - tip_max_length)]
    #fit a stright line to filtered_points, produce a vector to describe it
    if len(filtered_points) < 2:
        return False, [tip_x_cm, tip_y_cm, tip_z_cm], [0, 0, 0]
    # GETS VECTOR:
    P = np.asarray(filtered_points, dtype=float)
    # 1) Point on the line = centroid
    centroid = P.mean(axis=0)
    # 2) Direction = first principal component (unit vector)
    X = P - centroid
    # SVD of the Nx3 centered coords
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    direction = Vt[0]                 # principal axis
    direction = direction / np.linalg.norm(direction)  # unit length
    # print tip location and vector
    # print(f"Tip location: ({tip_x_cm:.2f}, {tip_y_cm:.2f}, {tip_z_cm:.2f}), Direction: {direction}")
    return True, [tip_x_cm, tip_y_cm, tip_z_cm], direction.tolist()

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def compute_bevel_angle(target_location, needle_tip_location, needle_direction):
    P = np.array(needle_direction)
    Y = np.array([0,1,0])
    t = np.array(target_location)
    n = np.array(needle_tip_location)

    P_hat = normalize(P)
    if P_hat[2] > 0:
        P_hat = -P_hat # makes P_hat point towards target


    A = Y - (np.dot(P_hat, Y))*P_hat
    if A[1] < 0:
        A = -A # make sure A is pos Y direction
    B = (t - n) - (np.dot(t - n, P_hat))*P_hat
    Theta = np.arctan2(np.dot(P_hat, np.cross(A, B)), np.dot(A, B))
    bev_angle_deg = Theta * 180 / np.pi

    gamma = np.arccos( abs(np.dot(P_hat, (t-n))) / (np.linalg.norm(t-n)) ) # abs P_hat is 1
    magnitude_deg = gamma * 180 / np.pi

    return bev_angle_deg, magnitude_deg





def send_motor_position(position):
    """Send target position to OpenRB-150."""
    motor.write(f"{position}\n".encode('utf-8'))

def get_latest_readings():
    """Get the most recent complete reading, discarding older buffered lines."""
    # Discard stale data if multiple lines are queued
    while arduino.in_waiting > 1:
        arduino.readline()
    
    line = arduino.readline().decode('utf-8').strip()
    if line:
        try:
            values = list(map(float, line.split(',')))
            if len(values) == 6:
                return values
        except ValueError:
            pass
    return None

def get_views(cur_frame):
    rawimage = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
    h, w = rawimage.shape
    Sview = rawimage[60:400, 180:360]
    Eview = rawimage[60:430, 360:540]
    Sview = cv2.rotate(Sview, cv2.ROTATE_90_COUNTERCLOCKWISE)
    Eview = cv2.rotate(Eview, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return Sview, Eview

def get_target_position(cur_frame):
    Sview, Eview = get_views(cur_frame)

  # Guass blur
    img_blur = cv2.GaussianBlur(Eview.copy(), (5,5), 0)

  # Threshold the image to isolate bright areas
    _, thresh = cv2.threshold(img_blur, 80, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Make a color copy so we can draw contours in color
    contour_img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    centers = []  # store centerpoints (x, y)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 30 <= area <= 110:  # area is about 60, was 30 - 100
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                roundness = 4 * np.pi * area / (perimeter ** 2)
                if roundness > 0.3:  # closer to 1 = more circular
                    # Draw contour

                    # Compute center using image moments
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        centers.append((cX, cY, roundness, area))

    return centers, Eview


def segment_distance(seg1, seg2, max_dist=5):
    pts1 = np.linspace(seg1[0], seg1[1], num=10)
    pts2 = np.linspace(seg2[0], seg2[1], num=10)
    for p1 in pts1:
        for p2 in pts2:
            if np.linalg.norm(np.array(p1) - np.array(p2)) < max_dist:
                return True
    return False

def group_segments(segments):
    n = len(segments)
    visited = [False] * n
    groups = []

    for i in range(n):
        if visited[i]:
            continue
        group = [i]
        visited[i] = True
        queue = [i]

        while queue:
            current = queue.pop()
            for j in range(n):
                if not visited[j] and segment_distance(segments[current], segments[j]):
                    visited[j] = True
                    queue.append(j)
                    group.append(j)

        groups.append(group)

    return groups

def segment_length(seg):
    return np.linalg.norm(np.array(seg[0]) - np.array(seg[1]))


def get_needle_position(cur_frame, us_roll):
    tip_found = True
    tip_x = 0
    tip_y = 0
    tip_grad = 0
    pipeline = []

    min_x = 999
    needle_end_x = 999
    needle_end_y = 999
    shadow_bottom = 0

    Sview, Eview = get_views(cur_frame)

    # Preview (original image)
    pipeline.append(cv2.cvtColor(Sview.copy(), cv2.COLOR_GRAY2BGR))

    # Guass blur
    img_blur = cv2.GaussianBlur(Sview.copy(), (5,5), 0)

    # Sobel Edge Detection
    sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
    sobely_scaled = np.clip((sobely - sobely.min()) / (sobely.max() - sobely.min()) * 255, 0, 255).astype(np.uint8)
    #invert
    inverted = cv2.bitwise_not(sobely_scaled)

    # Threshold (tune thresh_val to your data)
    thresh_val = 180  # 0–255, lower = more pixels kept
    _, binary_sobely = cv2.threshold(inverted, thresh_val, 255, cv2.THRESH_BINARY)

    #lines
    Hlines = cv2.HoughLinesP(cv2.convertScaleAbs(binary_sobely), rho=1, theta=np.pi/180, threshold=50, minLineLength=30, maxLineGap=0)


    # Below is vertical sobel aka stage 2
    # Sobel Edge Detection
    sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=7) # Sobel Edge Detection on the Y axis
    sobely_scaled = np.clip((sobely - sobely.min()) / (sobely.max() - sobely.min()) * 255, 0, 255).astype(np.uint8)

    # Threshold (tune thresh_val to your data)
    hist = cv2.calcHist([sobely_scaled], [0], None, [256], [0, 256])
    mode_val = np.argmax(hist)  # value between 0–255
    thresh_val = mode_val + 15
    _, binary_sobely = cv2.threshold(sobely_scaled, thresh_val, 255, cv2.THRESH_BINARY)

    #lines
    Vlines = cv2.HoughLinesP(cv2.convertScaleAbs(binary_sobely), rho=1, theta=np.pi/180, threshold=40, minLineLength=20, maxLineGap=0)


    Hsegments = []
    if Hlines is not None:
        for line in Hlines:
            x1, y1, x2, y2 = line[0]
            if y1 > 15 and y1 < 170 and y2 > 15 and y2 < 170: # crops top and bottom  
                Hsegments.append(((x1, y1), (x2, y2)))
        if Hsegments:  # non-empty
            # Group the segments
            groups = group_segments(Hsegments)
            # Compute total length of each group
            group_lengths = [sum(segment_length(Hsegments[i]) for i in group) for group in groups]
            # Select the largest group (the needle)
            largest_group_idx = np.argmax(group_lengths)
            needle_segments = [Hsegments[i] for i in groups[largest_group_idx]]

            #interpolate and extract points for line fitting
            points = []
            for (x1, y1), (x2, y2) in needle_segments:
                # calculate tip-most point for validation later
                if x1 < needle_end_x:
                    needle_end_x = x1
                if x2 < needle_end_x:
                    needle_end_x = x2
                if y1 < needle_end_y:
                    needle_end_y = y1
                if y2 < needle_end_y:
                    needle_end_y = y2
                # Compute segment length
                length = np.linalg.norm([x2 - x1, y2 - y1])
                # Number of interpolation steps (at least 1 to include both ends)
                num_steps = max(int(length // 20), 1)
                # Generate interpolated points along the line
                for i in range(num_steps + 1):
                    t = i / num_steps
                    xi = int((1 - t) * x1 + t * x2)
                    yi = int((1 - t) * y1 + t * y2)
                    points.append((xi, yi))
            points = np.array(points)
            points = points[np.argsort(points[:, 0])]
            x, y = points[:, 0], points[:, 1]
            x += -3 # moves the line up 3 pixels

            min_x = np.min(x)
            # Mask for points within 100 units of min_x
            mask = x <= min_x + 100
            x_subset = x[mask]
            y_subset = y[mask]
            coeffs = np.polyfit(x_subset, y_subset, deg=1)
            #we now have coeffs for our line describing the needle tip

        else: # do something else if empty
            tip_found = False
    else:
        tip_found = False

    Vsegments = []
    if Vlines is not None:
        for line in Vlines:
            x1, y1, x2, y2 = line[0]
            if x1 < 250 and x2 < 250: # crops left side
                if abs(y2 - y1) >= abs(x2 - x1): # line is at least 45 degrees
                    Vsegments.append(((x1, y1), (x2, y2)))
                    if y1 > shadow_bottom:
                        shadow_bottom = y1
                    if y2 > shadow_bottom:
                        shadow_bottom = y2
        if Vsegments and tip_found:  # non-empty
            # Group the segments
            groups = group_segments(Vsegments)
            # Compute total length of each group
            group_lengths = [sum(segment_length(Vsegments[i]) for i in group) for group in groups]
            # Select the largest group (the tip shaddow)
            largest_group_idx = np.argmax(group_lengths)
            needle_segments = [Vsegments[i] for i in groups[largest_group_idx]]
            # Find smallest x in that group
            min_x = 999
            for (x1, y1), (x2, y2) in needle_segments:
                min_x = min(min_x, x1, x2)  # compare both endpoints
        else: # do something else if empty
            tip_found = False
    else:
        tip_found = False

    # Validation that the needle tip is near the shadow
    if abs(needle_end_x - min_x) > 50:
        # Needle tip is not valid
        tip_found = False
    # Validation that the shadow is close to the needle tip
    if abs(shadow_bottom - needle_end_y) > 20:
        # Shadow is not valid
        tip_found = False

    # -------------------- Draw all lines
    overlay_img = cv2.cvtColor(Sview.copy(), cv2.COLOR_GRAY2BGR)
    if Hsegments:  # non-empty
        for (x1, y1), (x2, y2) in Hsegments:
            cv2.line(overlay_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    if Vsegments:  # non-empty
        for (x1, y1), (x2, y2) in Vsegments:
            cv2.line(overlay_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    pipeline.append(overlay_img)

    # -------------------- Draw on last lines
    overlay_img2 = cv2.cvtColor(Sview.copy(), cv2.COLOR_GRAY2BGR)
    if tip_found:  # non-empty
        # Create a copy of Sview in color so we can draw in color

        # Draw vertical line at min_x
        cv2.line(overlay_img2, (min_x, 0), (min_x, int(np.polyval(coeffs, min_x))), (0, 255, 0), 2)

        # Prepare x values for the polynomial from min_x to the largest x in the original fit
        x_fit_range = np.arange(min_x, x.max() + 1)
        y_fit_range = np.polyval(coeffs, x_fit_range)

        # Draw polynomial curve
        for xi, yi in zip(x_fit_range, y_fit_range.astype(int)):
            if 0 <= yi < overlay_img2.shape[0]:
                overlay_img2[yi, xi] = (0, 0, 255)

        # Append to pipeline
    pipeline.append(overlay_img2)

    # ------------------- Draw on done
    if tip_found:
        tip_x = min_x
        tip_y = np.polyval(coeffs, min_x)
        # Gradient at tip_x
        tip_grad = coeffs[0]  # slope of the line

        log_tip_point(tip_x, tip_y, us_roll, Sview)

    # cv2.imshow('Needle Detection Pipeline', np.hstack(pipeline))
    pipeline_stacked = np.hstack(pipeline)
    return tip_found, tip_x, tip_y, tip_grad, pipeline_stacked

def pad_to_width(img, target_width):
    h, w = img.shape[:2]
    pad_left = (target_width - w) // 2
    pad_right = target_width - w - pad_left
    return cv2.copyMakeBorder(img, 0, 0, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))  # black padding


def _init_tip_plot():
    global _fig, _ax_top, _ax_bottom, _sc_top, _sc_bot, _tgt_top, _tgt_bot
    _fig, (_ax_top, _ax_bottom) = plt.subplots(2, 1, figsize=(3.5, 5.5))
    # Top
    _sc_top  = _ax_top.scatter([], [], c='green', s=10)   # tip points
    _tgt_top = _ax_top.scatter([], [], c='purple', s=100) # target
    _ax_top.set_xlim(0, 12); _ax_top.set_ylim(-4, 4)
    _ax_top.set_xlabel("Z (cm)"); _ax_top.set_ylabel("X (cm)")
    _ax_top.set_title("Tip Location - Top View")
    _ax_top.set_aspect('equal', adjustable='box')
    # Bottom
    _sc_bot  = _ax_bottom.scatter([], [], c='green', s=10)
    _tgt_bot = _ax_bottom.scatter([], [], c='purple', s=100)
    _ax_bottom.set_xlim(0, 12); _ax_bottom.set_ylim(0, 8)
    _ax_bottom.set_xlabel("Z (cm)"); _ax_bottom.set_ylabel("Y (cm)")
    _ax_bottom.set_title("Tip Location - Side View")
    _ax_bottom.set_aspect('equal', adjustable='box')
    plt.tight_layout()


def make_tip_plot(targetCoords):
    """Fast update of the tip scatter plots (hides tip lines)."""
    global _fig, _ax_top, _ax_bottom, _sc_top, _sc_bot, _tgt_top, _tgt_bot
    global _tip_back_top, _tip_fwd_top, _tip_back_bot, _tip_fwd_bot

    if _fig is None or _ax_top is None or _ax_bottom is None:
        _init_tip_plot()

    # tip_points_cm is a Python list-of-[x,y,z] defined elsewhere
    arr = np.asarray(tip_points_cm, dtype=float).reshape(-1, 3)

    # Lazily create artists
    if _sc_top is None:  _sc_top  = _ax_top.scatter([], [], c='green',  s=10)
    if _sc_bot is None:  _sc_bot  = _ax_bottom.scatter([], [], c='green', s=10)
    if _tgt_top is None: _tgt_top = _ax_top.scatter([], [], c='purple', s=100)
    if _tgt_bot is None: _tgt_bot = _ax_bottom.scatter([], [], c='purple', s=100)
    if _tip_back_top is None: (_tip_back_top,) = _ax_top.plot([], [], c='green', linewidth=2)
    if _tip_fwd_top  is None: (_tip_fwd_top,)  = _ax_top.plot([], [], c='green', linestyle='--', linewidth=1)
    if _tip_back_bot is None: (_tip_back_bot,) = _ax_bottom.plot([], [], c='green', linewidth=2)
    if _tip_fwd_bot  is None: (_tip_fwd_bot,)  = _ax_bottom.plot([], [], c='green', linestyle='--', linewidth=1)

    # Show scatters, hide lines
    _sc_top.set_visible(True);  _sc_bot.set_visible(True)
    _tip_back_top.set_visible(False); _tip_fwd_top.set_visible(False)
    _tip_back_bot.set_visible(False); _tip_fwd_bot.set_visible(False)

    # Update tip points (Z, X) and (Z, Y)
    if arr.size:
        _sc_top.set_offsets(np.column_stack((arr[:, 2], arr[:, 0])))
        _sc_bot.set_offsets(np.column_stack((arr[:, 2], arr[:, 1])))
    else:
        _sc_top.set_offsets(np.empty((0, 2)))
        _sc_bot.set_offsets(np.empty((0, 2)))

    # Update targets
    _tgt_top.set_offsets([[targetCoords[2], targetCoords[0]]])
    _tgt_bot.set_offsets([[targetCoords[2], targetCoords[1]]])

    # Render
    _fig.canvas.draw()
    H, W = _fig.canvas.get_width_height()[1], _fig.canvas.get_width_height()[0]
    plot_img = np.frombuffer(_fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(H, W, 4)
    plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGBA2BGR)
    plot_img = 255 - plot_img
    return plot_img


def make_draw_tip(targetCoords, tipPoint, tipVector):
    """Fast update of target + tip direction lines (hides tip scatters)."""
    global _fig, _ax_top, _ax_bottom
    global _sc_top, _sc_bot, _tgt_top, _tgt_bot
    global _tip_back_top, _tip_fwd_top, _tip_back_bot, _tip_fwd_bot

    if _fig is None or _ax_top is None or _ax_bottom is None:
        _init_tip_plot()

    targetCoords = np.asarray(targetCoords, dtype=float)
    tipPoint    = np.asarray(tipPoint,    dtype=float)
    tipVector   = np.asarray(tipVector,   dtype=float)

    z_min, z_max = 0.0, 12.0

    # Backward segment (2 units opposite tipVector)
    back_point = tipPoint - 2.0 * tipVector

    # Forward segment to z boundary in actual vector direction
    if tipVector[2] > 0:
        scale_to_limit = (z_max - tipPoint[2]) / tipVector[2]
    elif tipVector[2] < 0:
        scale_to_limit = (z_min - tipPoint[2]) / tipVector[2]
    else:
        scale_to_limit = 0.0
    forward_point = tipPoint + tipVector * scale_to_limit

    # Lazily create artists (if not already)
    if _sc_top is None:  _sc_top  = _ax_top.scatter([], [], c='green',  s=10)
    if _sc_bot is None:  _sc_bot  = _ax_bottom.scatter([], [], c='green', s=10)
    if _tgt_top is None: _tgt_top = _ax_top.scatter([], [], c='purple', s=100)
    if _tgt_bot is None: _tgt_bot = _ax_bottom.scatter([], [], c='purple', s=100)
    if _tip_back_top is None: (_tip_back_top,) = _ax_top.plot([], [], c='green', linewidth=2)
    if _tip_fwd_top  is None: (_tip_fwd_top,)  = _ax_top.plot([], [], c='green', linestyle='--', linewidth=1)
    if _tip_back_bot is None: (_tip_back_bot,) = _ax_bottom.plot([], [], c='green', linewidth=2)
    if _tip_fwd_bot  is None: (_tip_fwd_bot,)  = _ax_bottom.plot([], [], c='green', linestyle='--', linewidth=1)

    # Hide scatters, show lines
    _sc_top.set_visible(False); _sc_bot.set_visible(False)
    _tip_back_top.set_visible(True); _tip_fwd_top.set_visible(True)
    _tip_back_bot.set_visible(True); _tip_fwd_bot.set_visible(True)

    # Update targets
    _tgt_top.set_offsets([[targetCoords[2], targetCoords[0]]])  # (Z, X)
    _tgt_bot.set_offsets([[targetCoords[2], targetCoords[1]]])  # (Z, Y)

    # Update tip lines
    _tip_back_top.set_data([back_point[2],   tipPoint[2]],   [back_point[0],   tipPoint[0]])
    _tip_fwd_top.set_data( [tipPoint[2],     forward_point[2]], [tipPoint[0],     forward_point[0]])
    _tip_back_bot.set_data([back_point[2],   tipPoint[2]],   [back_point[1],   tipPoint[1]])
    _tip_fwd_bot.set_data( [tipPoint[2],     forward_point[2]], [tipPoint[1],     forward_point[1]])

    # Render
    _fig.canvas.draw()
    H, W = _fig.canvas.get_width_height()[1], _fig.canvas.get_width_height()[0]
    plot_img = np.frombuffer(_fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(H, W, 4)
    plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGBA2BGR)
    plot_img = 255 - plot_img
    return plot_img


def output_data(targetCoords):
    # helper to convert np.float64 -> float and round
    def clean(x, ndigits=2):
        return round(float(x), ndigits)

    # Target
    print(f"target.append({[clean(x, 2) for x in targetCoords]})\n")
    # Needle points
    print("needle_points.append([")
    for p in tip_points_cm:
        rounded = [clean(x, 2) for x in p]
        print(f"    {rounded},")
    print("])\n")
    # Bevel angles (1 decimal place)
    print(f"bevel_angle.append({[clean(x, 1) for x in bev_angles_data]})\n")
    # Steering flags (force to int)
    print(f"actively_steering.append({[int(x) for x in target_found_data]})\n")


def save_data(targetCoords):
    
    
        
    # helper to convert np.float64 -> float and round
    def clean(x, ndigits=2):
        return round(float(x), ndigits)

    # --- Ensure folder exists ---
    folder = os.path.join("user study", "trialX")
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filenameSave)

    with open(filepath, "a") as f:  # append mode so multiple calls stack
        # Target
        f.write(f"target.append({[clean(x, 2) for x in targetCoords]})\n\n")

        # Needle points
        f.write("needle_points.append([\n")
        for p in tip_points_cm:
            rounded = [clean(x, 2) for x in p]
            f.write(f"    {rounded},\n")
        f.write("])\n\n")

        # Bevel angles (1 decimal place)
        f.write(f"bevel_angle.append({[clean(x, 1) for x in bev_angles_data]})\n\n")

        # Steering flags (force to int)
        f.write(f"actively_steering.append({[int(x) for x in target_found_data]})\n\n")

    print(f"Data saved to {filepath}")

















def main():
    global tip_points_cm
    global bev_angles_data
    global target_found_data

    targetCoords = [0,0,0]
    targetxcm = 0
    targetycm = 0
    targetSet = False
    currentTime = time.time()
    updateInterval = 0.1  # seconds
    GUIupdateInterval = 1 # seconds
    nextUpdate = currentTime + updateInterval
    nextGUIUpdate = currentTime + GUIupdateInterval

    vectorFound = False
    tipPoint = [0,0,0]
    tipVector = [0,0,0]

    send_motor_position(0)

    while True:
        

        readings = get_latest_readings()
        # print(f"Latest readings: {readings}")
        if readings is None:
            continue
        gunRoll, gunPitch, gunYaw, usPitch, usRoll, usYaw = readings
        # get camera feed
        ret, frame = cap.read()
        if not ret:
            # Handle camera read error
            print("Failed to read from camera")
            continue

        currentTime = time.time()
        if currentTime >= nextUpdate:
            nextUpdate = currentTime + updateInterval
            #   processing
            if not targetSet:
                targets, endView = get_target_position(frame)
                # find target in 180-220 range
                target = None
                for (tx, ty, roundness, area) in targets:
                    if 160 <= tx <= 210 and ty < 150:
                        target = (tx, ty, roundness, area)
                        break
                if target:
                    xcm = (target[0] - 180)/26
                    ycm = (180 - target[1])/26
                    radiuscm = math.sqrt(xcm**2 + ycm**2)
                    alpha_rad = math.atan2(ycm, xcm)
                    alpha_deg = math.degrees(alpha_rad)
                    theta_deg = alpha_deg - usRoll
                    theta_rad = math.radians(theta_deg)
                    targetxcm = radiuscm * math.cos(theta_rad)
                    targetycm = radiuscm * math.sin(theta_rad)
                    # print(f"Target coordinates in cm: ({targetxcm:.2f}, {targetycm:.2f}), alpha: {alpha_deg:.2f}, usRoll: {usRoll:.2f}, theta: {theta_deg:.2f}")

                endView = cv2.cvtColor(endView, cv2.COLOR_GRAY2BGR)
                # Draw targets on frame
                for (tx, ty, roundness, area) in targets:
                    cv2.circle(endView, (int(tx), int(ty)), 5, (0, 0, 255), -1)
                    # cv2.putText(endView, f"({int(tx)}, {int(ty)}, {roundness:.2f}, {area})", (int(tx) + 5, int(ty) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                if target:
                    cv2.circle(endView, (target[0], target[1]), 5, (0, 255, 0), -1)  # center point

                #output:
                cv2.imshow('Webcam Feed', endView)

            else: # Target is set, time to needle hunt
                
                tipfound, tipxcm, tipycm, tipgrad, pipeline_stacked = get_needle_position(frame, usRoll)
                bevel_roll = 0
                magnitude = 0
                # tip point is also logged above
                # print(f"Tip found: {tipfound}, x: {tipx}, y: {tipy}, gradient: {tipgrad}")
                if tipfound:
                    vectorFound, tipPoint, tipVector = find_3d_tip_vector()
                if vectorFound:
                    print('calculating bevel angle')
                    print(f"targetCoords: {targetCoords}")
                    print(f"tipPoint:     {tipPoint}")
                    print(f"tipVector:    {tipVector}")
                    bevel_roll, magnitude = compute_bevel_angle(targetCoords, tipPoint, tipVector)
                    print(f'Output: Bevel roll: {bevel_roll}, Magnitude: {magnitude}')

                if tipfound: # logging our data for analysis later
                    bev_angles_data.append(bevel_roll)
                    if vectorFound and magnitude > 1:
                        target_found_data.append(1)
                    else:
                        target_found_data.append(0)

                # if vectorFound and tilt > threshold calculate bevel tip angle from bevel roll, output it
                if vectorFound and magnitude > 1:
                    motorposition = bevel_roll - gunRoll
                    if active:
                        send_motor_position(motorposition)
                    # print(f"Bevel roll is: {bevel_roll}, Motor position set to: {motorposition}")

                if currentTime >= nextGUIUpdate:
                    nextGUIUpdate = currentTime + GUIupdateInterval
                    # we should have a choice between a couple different user interfaces, or combine them.
                    # 1. US images
                    # 2. Sview pipeline
                    # 3. needle tip points in 3D
                    # 4. 3D model reconstruction
                    # OR
                    # 5. all 4 in one screen

                    # this does 1.
                    full_us_with_target = cv2.rotate(frame[60:400, 180:540], cv2.ROTATE_90_COUNTERCLOCKWISE)
                    radius = math.sqrt(targetCoords[0]**2 + targetCoords[1]**2) * 26
                    alpha_rad = math.atan2(targetCoords[1], targetCoords[0])
                    alpha_deg = math.degrees(alpha_rad)
                    theta_deg = alpha_deg + usRoll
                    theta_rad = math.radians(theta_deg)
                    xpixel = 180 + (radius * math.cos(theta_rad))
                    ypixel = 180 - (radius * math.sin(theta_rad))
                    cv2.circle(full_us_with_target, (int(xpixel), int(ypixel)), 4, (0, 255, 0), -1)
                    h, w = full_us_with_target.shape[:2]
                    # Scale factor to make height 550
                    scale = 550 / h
                    new_w = int(w * scale)
                    new_h = 550
                    # Resize while keeping aspect ratio
                    resized_full_us_with_target = cv2.resize(full_us_with_target, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


                    # pipeline_stacked is for 2.

                    # this does 3.
                    tip_plot_scatter_img = make_tip_plot(targetCoords)

                    # this does 4.
                    tip_unit_vector = [0, 0, 0]
                    if vectorFound:
                        tip_unit_vector = normalize(tipVector)
                        if tip_unit_vector[2] > 0:
                            tip_unit_vector = -tip_unit_vector 
                        
                    tip_plot_img = make_draw_tip(targetCoords, tipPoint, tip_unit_vector)

                    #now combine images:
                    stacked = np.hstack((resized_full_us_with_target, tip_plot_scatter_img, tip_plot_img))
                    target_width = stacked.shape[1]
                    # resize pipeline_stacked to match target_width
                    h, w = pipeline_stacked.shape[:2]
                    scale = target_width / w
                    new_h = int(h * scale)
                    pipeline_stacked_resized = cv2.resize(pipeline_stacked, (target_width, new_h), interpolation=cv2.INTER_LINEAR)
                    #combine
                    final_display = np.vstack((stacked, pipeline_stacked_resized))
                    cv2.imshow("GUI", final_display)


            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                sys.exit()
            elif key == ord(' '):
                # Space key pressed
                if target:
                    targetCoords[0] = targetxcm
                    targetCoords[1] = targetycm
                    targetCoords[2] = 0
                    targetSet = True
                    print(f"Target coordinates set: {targetCoords}")
            elif key == ord('d'):
                # output_data(targetCoords)
                save_data(targetCoords)
                sys.exit()
while True:
    if __name__ == "__main__":
        main()
