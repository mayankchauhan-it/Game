# import numpy as np
# import cv2
# import math
# import pyautogui

# # Open Camera
# capture = cv2.VideoCapture(0)

# # Set the finger box dimensions
# box_x = 100
# box_y = 100
# box_width = 200 
# box_height = 200

# # Set the jump trigger threshold (percentage of box area covered by finger)
# jump_threshold = 0.5

# while capture.isOpened():
#     # Capture frames from the camera
#     ret, frame = capture.read()

#     # Flip the frame horizontally to create a mirror effect
#     frame = cv2.flip(frame, 1)

#     # Get hand data from the rectangle sub window
#     cv2.rectangle(frame, (box_x, box_y), (box_x + box_width, box_y + box_height), (0, 255, 0), 0)
#     crop_image = frame[box_y:box_y+box_height, box_x:box_x+box_width]

#     # Convert the cropped image to grayscale
#     gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)

#     # Apply Gaussian blur
#     blur = cv2.GaussianBlur(gray, (3, 3), 0)

#     # Apply thresholding to create a binary image with skin color as white and the rest as black
#     _, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

#     # Find contours
#     contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Find the contour with maximum area
#     if len(contours) > 0:
#         contour = max(contours, key=cv2.contourArea)

#         # Find the convex hull and its defects
#         hull = cv2.convexHull(contour, returnPoints=False)
#         defects = cv2.convexityDefects(contour, hull)

#         # Count the number of fingers raised
#         finger_count = 0

#         if defects is not None:
#             for i in range(defects.shape[0]):
#                 s, e, f, _ = defects[i, 0]
#                 start = tuple(contour[s][0])
#                 end = tuple(contour[e][0])
#                 far = tuple(contour[f][0])

#                 # Calculate the triangle sides
#                 a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
#                 b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
#                 c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)

#                 # Calculate the angle using the cosine rule
#                 angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))

#                 # If the angle is below a certain threshold, consider it as a finger
#                 if angle < math.pi / 2:
#                     finger_count += 1

#                     # Draw circles on the finger points
#                     cv2.circle(crop_image, far, 5, [0, 0, 255], -1)

#             # Trigger jump effect if finger count is 1 and finger is within the box
#             box_area = box_width * box_height
#             finger_area = cv2.contourArea(contour)
#             area_ratio = finger_area / box_area

#             if finger_count == 1 and area_ratio > jump_threshold:
#                 pyautogui.press('space')
#                 cv2.putText(frame, "JUMP", (115, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

#     # Show the mirrored frame with the added rectangle and finger detection
#     cv2.imshow("Gesture", frame)

#     # Close the camera if 'q' is pressed
#     if cv2.waitKey(1) == ord('q'):
#         break

# capture.release()
# cv2.destroyAllWindows()





# Jump while your finger swipe right to left.


# import numpy as np
# import cv2
# import math
# import pyautogui

# # Open Camera
# capture = cv2.VideoCapture(0)

# # Set the parameters for hand gesture detection
# gesture_threshold = 100
# gesture_length = 0
# gesture_detected = False
# gesture_start_x = None
# gesture_end_x = None

# while capture.isOpened():
#     ret, frame = capture.read()
#     frame = cv2.flip(frame, 1)

#     cv2.rectangle(frame, (100, 100), (200, 200), (0, 255, 0), 0)
#     crop_image = frame[100:300, 100:300]

#     gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (3, 3), 0)

#     _, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

#     contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     if len(contours) > 0:
#         contour = max(contours, key=cv2.contourArea)
#         hull = cv2.convexHull(contour, returnPoints=False)
#         defects = cv2.convexityDefects(contour, hull)

#         if defects is not None and defects.shape[0] > 0:
#             s, e, f, _ = defects[0, 0]
#             start = tuple(contour[s][0])
#             end = tuple(contour[e][0])

#             if gesture_start_x is None:
#                 gesture_start_x = start[0]
#             else:
#                 gesture_length = abs(start[0] - gesture_start_x)

#                 if gesture_length > gesture_threshold:
#                     gesture_detected = True
#                     gesture_end_x = start[0]

#             if gesture_detected:
#                 if gesture_end_x < gesture_start_x:
#                     cv2.putText(frame, "JUMP", (115, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
#                     pyautogui.press('space')

#                 # Reset the gesture variables
#                 gesture_detected = False
#                 gesture_start_x = None
#                 gesture_end_x = None

#     cv2.imshow("Gesture", frame)

#     if cv2.waitKey(1) == ord('q'):
#         break

# capture.release()
# cv2.destroyAllWindows()



import numpy as np
import cv2
import math
import pyautogui

# Open Camera
capture = cv2.VideoCapture(0)

# Set the parameters for hand gesture detection
gesture_threshold = 100
is_inside_box = False
continuous_jump = False

while capture.isOpened():
    ret, frame = capture.read()
    frame = cv2.flip(frame, 1)

    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 0)
    crop_image = frame[100:300, 100:300]

    gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    _, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull)

        if defects is not None and defects.shape[0] > 0:
            s, e, f, _ = defects[0, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])

            if start[0] > 100 and start[0] < 300:
                is_inside_box = True
            else:
                is_inside_box = False
                continuous_jump = False

            if is_inside_box and not continuous_jump:
                if start[0] < gesture_threshold:
                    continuous_jump = True
                    pyautogui.press('space')

        else:
            is_inside_box = False
            continuous_jump = False

    cv2.imshow("Gesture", frame)

    if cv2.waitKey(1) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
