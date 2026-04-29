import cv2 as cv
import numpy as np

# --- Configuration ---
LOUPE_SIZE = 120  
ZOOM_LEVEL = 4    
REF_HEIGHT = 28.0 
INPUT_IMAGE = 'images/table_bottle03.jpeg'

def to_h(p):
    return np.array([np.float32(p[0]), np.float32(p[1]), 1.0])

def to_p(v):
    if abs(v[2]) < 1e-6: return np.array([v[0], v[1]]) 
    return (v[:2] / v[2]).astype(int)

def vanishing_point_of_two_parallel_lines(p1, p2, p3, p4):
    return np.cross(np.cross(to_h(p1), to_h(p2)), np.cross(to_h(p3), to_h(p4)))


def handle_loupe(x, y):
    h, w = input_image.shape[:2]
    x1 = int(np.clip(x - LOUPE_SIZE // 2, 0, w - LOUPE_SIZE))
    y1 = int(np.clip(y - LOUPE_SIZE // 2, 0, h - LOUPE_SIZE))
    roi = input_image[y1:y1+LOUPE_SIZE, x1:x1+LOUPE_SIZE].copy()
    cv.drawMarker(roi, (x - x1, y - y1), (0, 255, 255), cv.MARKER_CROSS, 20, 1)
    display = cv.resize(roi, (LOUPE_SIZE * ZOOM_LEVEL, LOUPE_SIZE * ZOOM_LEVEL), interpolation=cv.INTER_CUBIC)
    cv.imshow('loupe', display)

def redraw_gizmos():
    global image_gizmos, points
    image_gizmos = input_image.copy()
    
    # visualize selected edges/parallel lines
    if len(points) >= 2: cv.line(image_gizmos, points[0], points[1], (0, 255, 255), 5)
    if len(points) >= 3: cv.line(image_gizmos, points[0], points[2], (0, 255, 255), 5)
    if len(points) >= 4:
        cv.line(image_gizmos, points[2], points[3], (0, 255, 255), 5)
        cv.line(image_gizmos, points[1], points[3], (0, 255, 255), 5)

        # calculate the vanishing points
        vpx = vanishing_point_of_two_parallel_lines(points[0], points[1], points[2], points[3])
        vpy = vanishing_point_of_two_parallel_lines(points[0], points[2], points[1], points[3])
        
        # draw horizon line (this will most likely not be visible since it's very far away.)
        p_vpx, p_vpy = to_p(vpx), to_p(vpy)
        cv.line(image_gizmos, p_vpx, p_vpy, (0, 255, 0), 2)

        # check if we have the reference base/top and target base/top
        if len(points) >= 8:
            rb = np.array(points[4]) # ref base
            rt = np.array(points[5]) # ref top
            tb = np.array(points[6]) # target base
            tt = np.array(points[7]) # target top

            # Horizon line as a vector
            horizon = np.cross(vpx, vpy)

            # vanishing point of the line connecting the two bases on the ground
            ground_line = np.cross(to_h(rb), to_h(tb))
            v_ground = np.cross(ground_line, horizon)

            # Transfer the top of reference to the target's vertical line
            # This line vanishes at v_ground because it's parallel to the ground connector
            transfer_line = np.cross(v_ground, to_h(rt))
            
            # intersect transfer_line with the vertical line passing thru tb
            target_vert_line = np.array([1, 0, -tb[0]])
            r_prime_h = np.cross(transfer_line, target_vert_line)
            r_prime = to_p(r_prime_h)

            # visualisation stuffs
            # reference object (the bottle)
            cv.line(image_gizmos, tuple(rb), tuple(rt), (255, 0, 0), 8)
            # target object (the cup)
            cv.line(image_gizmos, tuple(tb), tuple(tt), (0, 165, 255), 8)
            # projection lines (rb <-> tb and rt <-> r_prime and tb <-> r_prime)
            cv.line(image_gizmos, tuple(rb), tuple(tb), (255, 0, 255), 8)
            cv.line(image_gizmos, tuple(rt), tuple(r_prime), (255, 0, 255), 8)
            cv.line(image_gizmos, tuple(tb), tuple(r_prime), (0, 0, 255), 8)

            # h_target / h_ref = pixel_height_target / pixel_height_ref_at_target
            h_target_px = np.linalg.norm(tt - tb)
            h_ref_at_target_px = np.linalg.norm(r_prime - tb)
            
            # this will be in whatever unit the reference is, which is in this case cm.
            calculated_height = (h_target_px / (h_ref_at_target_px)) * REF_HEIGHT
            print(f"Estimated height: {calculated_height} cm")

            cv.putText(image_gizmos, f"H: {calculated_height:.2f} cm", (tt[0], tt[1] - 50), cv.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 8)

    # selected points
    for i, p in enumerate(points):
        cv.circle(image_gizmos, p, 20, (255, 0, 0), 10)

    cv.imshow('scene', image_gizmos)

# run dis thing
input_image = cv.imread(INPUT_IMAGE)
points = []
cv.namedWindow('scene', cv.WINDOW_NORMAL)
cv.resizeWindow('scene', 1600, 1200)
cv.setMouseCallback('scene', lambda event, x, y, flags, param: (
    points.append((x, y)) if event == cv.EVENT_LBUTTONDOWN else None,
    handle_loupe(x, y) if event == cv.EVENT_MOUSEMOVE else None,
    redraw_gizmos()
))
redraw_gizmos()
cv.waitKey(0)
cv.destroyAllWindows()