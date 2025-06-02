import cv2
import math
import numpy as np

# --- Configuration ---
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_COLOR = (0, 255, 0) # Green
LINE_THICKNESS = 2
CIRCLE_RADIUS = 5
REFERENCE_LINE_COLOR = (0, 0, 255) # Red
OBJECT_LINE_COLOR = (255, 0, 0)   # Blue
INSTRUCTION_TEXT_COLOR = (255, 255, 255) # White

# --- Global Variables ---
ref_points = []  # Stores (x,y) for reference object
obj_points = []  # Stores (x,y) for target object
pixels_per_metric = None # Stores the calculated pixels per metric (e.g., pixels per cm)
known_ref_width_cm = 0.0 # User-defined width of the reference object in cm
measurement_text = "" # Text to display for the current measurement

def calculate_distance(p1, p2):
    """Calculates the Euclidean distance between two points."""
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def mouse_callback(event, x, y, flags, param):
    """Handles mouse clicks to select points for reference and object measurement."""
    global ref_points, obj_points, pixels_per_metric, measurement_text

    if event == cv2.EVENT_LBUTTONDOWN:
        if known_ref_width_cm <= 0:
            print("Please enter the known width of the reference object in the console first.")
            # Potentially display this message on screen too
            return

        if len(ref_points) < 2:
            ref_points.append((x, y))
            print(f"Reference point {len(ref_points)} selected: ({x}, {y})")
            if len(ref_points) == 2:
                pixel_distance_ref = calculate_distance(ref_points[0], ref_points[1])
                if pixel_distance_ref > 0 and known_ref_width_cm > 0:
                    pixels_per_metric = pixel_distance_ref / known_ref_width_cm
                    print(f"Reference object pixel width: {pixel_distance_ref:.2f} pixels")
                    print(f"Known reference real width: {known_ref_width_cm:.2f} cm")
                    print(f"Pixels per CM calculated: {pixels_per_metric:.2f}")
                    measurement_text = "" # Clear previous measurement
                else:
                    print("Error: Reference pixel distance or known width is zero. Resetting reference points.")
                    ref_points = [] # Reset if calculation is not possible
                    pixels_per_metric = None

        elif pixels_per_metric is not None and len(obj_points) < 2:
            obj_points.append((x, y))
            print(f"Object point {len(obj_points)} selected: ({x}, {y})")
            if len(obj_points) == 2:
                pixel_distance_obj = calculate_distance(obj_points[0], obj_points[1])
                if pixels_per_metric > 0:
                    object_real_dimension = pixel_distance_obj / pixels_per_metric
                    measurement_text = f"Measured: {object_real_dimension:.2f} cm"
                    print(measurement_text)
                else:
                    print("Error: Pixels per metric is not valid. Please reset reference object.")
                    measurement_text = "Error: PPM not set"
                # obj_points = [] # Keep points to draw line, reset with 'r'

def draw_instructions(frame):
    """Draws usage instructions on the frame."""
    y0, dy = 30, 20
    instructions = [
        "Instructions:",
        f"1. Ref Width (console): {known_ref_width_cm} cm" if known_ref_width_cm > 0 else "1. Enter Ref. Width (cm) in console.",
        "2. Click 2 pts on REF object.",
        "3. Click 2 pts on TARGET object.",
        "Press 'r' to Reset points.",
        "Press 'q' to Quit."
    ]
    for i, line in enumerate(instructions):
        cv2.putText(frame, line, (10, y0 + i * dy), FONT, 0.5, INSTRUCTION_TEXT_COLOR, 1, cv2.LINE_AA)

def main():
    global ref_points, obj_points, pixels_per_metric, known_ref_width_cm, measurement_text

    # Prompt user for the known width of the reference object
    while True:
        try:
            val = input("Enter the known width of your reference object (e.g., in cm): ")
            known_ref_width_cm = float(val)
            if known_ref_width_cm > 0:
                break
            else:
                print("Please enter a positive value for the width.")
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cv2.namedWindow("Object Measurement")
    cv2.setMouseCallback("Object Measurement", mouse_callback)

    print("\nWebcam opened. Follow instructions in the window.")
    print("1. Ensure the reference object is in view.")
    print("2. Click two points defining its known width.")
    print("3. Click two points on the target object to measure its dimension.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        frame = cv2.flip(frame, 1) # Flip horizontally for a more natural mirror view
        display_frame = frame.copy() # Work on a copy for drawing

        # Draw instructions
        draw_instructions(display_frame)

        # Draw points and lines for reference object
        if len(ref_points) > 0:
            cv2.circle(display_frame, ref_points[0], CIRCLE_RADIUS, REFERENCE_LINE_COLOR, -1)
        if len(ref_points) == 2:
            cv2.circle(display_frame, ref_points[1], CIRCLE_RADIUS, REFERENCE_LINE_COLOR, -1)
            cv2.line(display_frame, ref_points[0], ref_points[1], REFERENCE_LINE_COLOR, LINE_THICKNESS)
            if pixels_per_metric:
                mid_point_ref = ((ref_points[0][0] + ref_points[1][0]) // 2, (ref_points[0][1] + ref_points[1][1]) // 2 - 10)
                cv2.putText(display_frame, f"Ref: {known_ref_width_cm:.2f} cm", mid_point_ref, FONT, FONT_SCALE, REFERENCE_LINE_COLOR, LINE_THICKNESS)
                cv2.putText(display_frame, f"PPM: {pixels_per_metric:.2f}", (10, display_frame.shape[0] - 30), FONT, FONT_SCALE, INSTRUCTION_TEXT_COLOR, LINE_THICKNESS)


        # Draw points and lines for target object and display measurement
        if len(obj_points) > 0:
            cv2.circle(display_frame, obj_points[0], CIRCLE_RADIUS, OBJECT_LINE_COLOR, -1)
        if len(obj_points) == 2:
            cv2.circle(display_frame, obj_points[1], CIRCLE_RADIUS, OBJECT_LINE_COLOR, -1)
            cv2.line(display_frame, obj_points[0], obj_points[1], OBJECT_LINE_COLOR, LINE_THICKNESS)
            if measurement_text:
                # Position text above the midpoint of the object line
                mid_x = (obj_points[0][0] + obj_points[1][0]) // 2
                mid_y = (obj_points[0][1] + obj_points[1][1]) // 2

                # Get text size to position it properly
                (text_width, text_height), _ = cv2.getTextSize(measurement_text, FONT, FONT_SCALE, LINE_THICKNESS)

                text_x = mid_x - text_width // 2
                text_y = mid_y - text_height # Position above the line

                # Ensure text is within frame boundaries
                text_x = max(text_x, 10)
                text_y = max(text_y, 20)

                cv2.putText(display_frame, measurement_text, (text_x, text_y), FONT, FONT_SCALE, OBJECT_LINE_COLOR, LINE_THICKNESS)


        cv2.imshow("Object Measurement", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quitting...")
            break
        elif key == ord('r'):
            print("Resetting points...")
            ref_points = []
            obj_points = []
            pixels_per_metric = None # Also reset PPM, requiring new reference calibration
            measurement_text = ""
            print("Reference points, object points, and PPM have been reset.")
            print("Please re-select reference points if you want to continue measuring.")


    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()