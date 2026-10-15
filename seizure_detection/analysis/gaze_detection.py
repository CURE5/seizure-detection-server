import cv2
import numpy as np
from .helpers import relative, relativeT
import math

def gaze(frame, landmarks, do_draw=True):
    """
    The gaze function gets an image and face landmarks from mediapipe framework.
    The function draws the gaze direction into the frame.
    """

    '''
    2D image points.
    relative takes mediapipe points that is normalized to [-1, 1] and returns image points
    at (x,y) format
    '''
    image_points = np.array([
        relative(landmarks[4], frame.shape),  # Nose tip
        relative(landmarks[152], frame.shape),  # Chin
        relative(landmarks[263], frame.shape),  # Left eye left corner
        relative(landmarks[33], frame.shape),  # Right eye right corner
        relative(landmarks[287], frame.shape),  # Left Mouth corner
        relative(landmarks[57], frame.shape)  # Right mouth corner
    ], dtype="double")

    '''
    2D image points.
    relativeT takes mediapipe points that is normalized to [-1, 1] and returns image points
    at (x,y,0) format
    '''
    image_points1 = np.array([
        relativeT(landmarks[4], frame.shape),  # Nose tip
        relativeT(landmarks[152], frame.shape),  # Chin
        relativeT(landmarks[263], frame.shape),  # Left eye, left corner
        relativeT(landmarks[33], frame.shape),  # Right eye, right corner
        relativeT(landmarks[287], frame.shape),  # Left Mouth corner
        relativeT(landmarks[57], frame.shape)  # Right mouth corner
    ], dtype="double")

    # 3D model points.
    model_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0, -63.6, -12.5),  # Chin
        (-43.3, 32.7, -26),  # Left eye, left corner
        (43.3, 32.7, -26),  # Right eye, right corner
        (-28.9, -28.9, -24.1),  # Left Mouth corner
        (28.9, -28.9, -24.1)  # Right mouth corner
    ])

    '''
    3D model eye points
    The center of the eye ball
    '''
    Eye_ball_center_right = np.array([[-29.05], [32.7], [-39.5]])
    Eye_ball_center_left = np.array([[29.05], [32.7], [-39.5]])  # the center of the left eyeball as a vector.

    '''
    camera matrix estimation
    '''
    focal_length = frame.shape[1]
    center = (frame.shape[1] / 2, frame.shape[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                  dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    # 2d pupil location
    left_pupil = relative(landmarks[468], frame.shape)
    right_pupil = relative(landmarks[473], frame.shape)

    # Transformation between image point to world point
    _, transformation, _ = cv2.estimateAffine3D(image_points1, model_points)  # image to world transformation
    left_eye_gaze = None
    right_eye_gaze = None
    if transformation is not None:  # if estimateAffine3D secsseded
        # project pupil image point into 3d world point 
        pupil_world_cord = transformation @ np.array([[left_pupil[0], left_pupil[1], 0, 1]]).T

        # 3D gaze point (10 is arbitrary value denoting gaze distance)
        S = Eye_ball_center_left + (pupil_world_cord - Eye_ball_center_left) * 10

        # Project a 3D gaze direction onto the image plane.
        (eye_pupil2D, _) = cv2.projectPoints((int(S[0]), int(S[1]), int(S[2])), rotation_vector,
                                             translation_vector, camera_matrix, dist_coeffs)
        # project 3D head pose into the image plane
        (head_pose, _) = cv2.projectPoints((int(pupil_world_cord[0]), int(pupil_world_cord[1]), int(40)),
                                           rotation_vector,
                                           translation_vector, camera_matrix, dist_coeffs)
        # correct gaze for head rotation
        gaze = left_pupil + (eye_pupil2D[0][0] - left_pupil) - (head_pose[0][0] - left_pupil)

        # Draw gaze line into screen
        p1 = (int(left_pupil[0]), int(left_pupil[1]))
        p2 = (int(gaze[0]), int(gaze[1]))
        left_eye_gaze = (p1, p2)
        if do_draw:
            cv2.line(frame, p1, p2, (0, 0, 255), 2)

        # correct gaze for head rotation
        gaze = right_pupil + (eye_pupil2D[0][0] - right_pupil) - (head_pose[0][0] - right_pupil)

        # Draw gaze line into screen
        p1 = (int(right_pupil[0]), int(right_pupil[1]))
        p2 = (int(gaze[0]), int(gaze[1]))
        right_eye_gaze = (p1, p2)
        if do_draw:
            cv2.line(frame, p1, p2, (0, 0, 255), 2)
    return {
        "left_eye_gaze": left_eye_gaze,
        "right_eye_gaze": right_eye_gaze
    }

def calculate_ear(eye_landmarks):
    # Unpack the landmarks
    p1, p2, p3, p4, p5, p6 = eye_landmarks

    # Calculate the Euclidean distances
    vertical_1 = math.sqrt((p2.x - p6.x)**2 + (p2.y - p6.y)**2 + (p2.z - p6.z)**2)
    vertical_2 = math.sqrt((p3.x - p5.x)**2 + (p3.y - p5.y)**2 + (p3.z - p5.z)**2)
    horizontal = math.sqrt((p1.x - p4.x)**2 + (p1.y - p4.y)**2 + (p1.z - p4.z)**2)

    # Calculate EAR
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear

def gaze_direction(frame, landmarks, do_draw=True, ):
    """
    The gaze function gets an image and face landmarks from the MediaPipe framework.
    It calculates horizontal and vertical gaze directions relative to the eye.
    """
# Define indices for eye corners and pupils
    LEFT_EYE_OUTER = 33
    LEFT_EYE_INNER = 133
    LEFT_EYE_TOP = 159
    LEFT_EYE_BOTTOM = 145
    LEFT_PUPIL = 468

    RIGHT_EYE_OUTER = 362
    RIGHT_EYE_INNER = 263
    RIGHT_EYE_TOP = 386
    RIGHT_EYE_BOTTOM = 374
    RIGHT_PUPIL = 473

    CLOSED_EYE_THRESHOLD = 0.15  # 0.2 proved too high in practice
    # Initialize gaze directions dictionary
    

    LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380] 
    left_eye_landmarks = [landmarks[i] for i in LEFT_EYE_INDICES]
    left_eye_ear = calculate_ear(left_eye_landmarks)
    left_eye_is_closed = None
    if left_eye_ear < CLOSED_EYE_THRESHOLD:
        left_eye_is_closed = True
    else:
        left_eye_is_closed = False

    RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    right_eye_landmarks = [landmarks[i] for i in RIGHT_EYE_INDICES]
    right_eye_ear = calculate_ear(right_eye_landmarks)

    right_eye_is_closed = None
    if right_eye_ear < CLOSED_EYE_THRESHOLD:
        right_eye_is_closed = True
    else:
        right_eye_is_closed = False

    is_closed = {
        "right_is_closed": right_eye_is_closed,
        "left_is_closed": left_eye_is_closed
    }

    gaze_directions = {}

    # Calculate horizontal and vertical positions for each eye
    for eye, outer_index, inner_index, top_index, bottom_index, pupil_index in [
        ("left", LEFT_EYE_OUTER, LEFT_EYE_INNER, LEFT_EYE_TOP, LEFT_EYE_BOTTOM, LEFT_PUPIL),
        ("right", RIGHT_EYE_OUTER, RIGHT_EYE_INNER, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, RIGHT_PUPIL)
    ]:
        # Convert eye landmarks to pixel coordinates
        eye_outer = relative(landmarks[outer_index], frame.shape)
        eye_inner = relative(landmarks[inner_index], frame.shape)
        eye_top = relative(landmarks[top_index], frame.shape)[1]
        eye_bottom = relative(landmarks[bottom_index], frame.shape)[1]
        eye_y = (eye_outer[1] + eye_inner[1])/2.0

        # Calculate eye dimensions
        eye_width = abs(eye_inner[0] - eye_outer[0])  # Horizontal distance
        eye_height = abs(eye_top - eye_bottom)        # Vertical distance

        # Get pupil position in pixel coordinates
        pupil = relative(landmarks[pupil_index], frame.shape)

        # Avoid division by zero for eye dimensions
        horizontal_position = 0.5  # Default to center
        vertical_position = 0.5    # Default to center

        if eye_width != 0:
            horizontal_position = (pupil[0] - min(eye_inner[0], eye_outer[0])) / eye_width

        if eye_height != 0:
            #vertical_position = (pupil[1] - eye_top) / eye_height
            vertical_position = (eye_y - eye_top) / eye_height # for some reason this works better

        # Clamp positions to [0, 1]
        horizontal_position = max(0.0, min(1.0, horizontal_position))
        vertical_position = max(0.0, min(1.0, vertical_position))

        # Add the calculated positions to the result dictionary
        gaze_directions[eye] = {
            "horizontal_position": float(horizontal_position),
            "vertical_position": float(vertical_position),
            "is_closed": is_closed[eye + "_is_closed"]
        }

        if do_draw:
            # Visualization (optional): Draw gaze direction
            if is_closed[eye + "_is_closed"]:
                cv2.line(frame, (int(eye_outer[0]), int(eye_outer[1])), (int(eye_inner[0]), int(eye_inner[1])), (255, 0, 0), 2) #BGR not RGB for some fkn reason
            else:
                cv2.circle(frame, (int(pupil[0]), int(pupil[1])), 5, (0, 255, 0), -1)  # Draw pupil
            
            # cv2.line(frame, (int(eye_outer[0]), int(eye_top)), (int(eye_inner[0]), int(eye_top)), (0, 255, 255), 1)
            # cv2.line(frame, (int(eye_outer[0]), int(eye_bottom)), (int(eye_inner[0]), int(eye_bottom)), (255, 0, 255), 1)

    # Return gaze direction as JSON
    return gaze_directions