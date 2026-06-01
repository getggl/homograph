# Copyright (C) 2026  Your Name
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free-Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.



import cv2
import numpy as np
import pyautogui

# Performance optimization
pyautogui.FAILSAFE = True 
pyautogui.PAUSE = 0

SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
cap = cv2.VideoCapture(1)  # Using camera index 1 as specified

# Smooth mouse motion variables
prev_screen_x, prev_screen_y = None, None
smooth_factor = 0.22  # smaller = smoother but more lag

# Create a dedicated window for tuning controls
cv2.namedWindow("Tuning Controls")

def nothing(x):
    pass

# Trackbars for Laser / Pen sensitivity
cv2.createTrackbar("Min Red Saturation", "Tuning Controls", 80, 255, nothing)
cv2.createTrackbar("Min Red Value/Bright", "Tuning Controls", 60, 255, nothing)
cv2.createTrackbar("Min Object Area", "Tuning Controls", 8, 100, nothing)

print("Welcome to rayTracing. Press 'q' to quit.")

# Initialize persistent laser lock coordinates
laser_x, laser_y = None, None

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Read live tuning values from sliders
    min_sat = cv2.getTrackbarPos("Min Red Saturation", "Tuning Controls")
    min_val = cv2.getTrackbarPos("Min Red Value/Bright", "Tuning Controls")
    min_area = cv2.getTrackbarPos("Min Object Area", "Tuning Controls")
    
    # Blue HSV range settings for corner markers
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_corners = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 20: 
            M_moments = cv2.moments(cnt)
            if M_moments["m00"] != 0:
                cx = int(M_moments["m10"] / M_moments["m00"])
                cy = int(M_moments["m01"] / M_moments["m00"])
                detected_corners.append([cx, cy])
                
                # Simple, clean indicator for corners (Solid cyan dot)
                cv2.circle(frame, (cx, cy), 6, (255, 255, 0), -1)

    if len(detected_corners) == 4:
        # Sort corners clockwise: TL, TR, BR, BL
        detected_corners = sorted(detected_corners, key=lambda x: x[0])
        left_side = sorted(detected_corners[:2], key=lambda x: x[1])
        right_side = sorted(detected_corners[2:], key=lambda x: x[1])
        
        whiteboard_corners = np.array([
            left_side[0],   # Top-Left
            right_side[0],  # Top-Right
            right_side[1],  # Bottom-Right
            left_side[1]    # Bottom-Left
        ], dtype=np.int32)

        # Draw a subtle bounding zone perimeter line
        cv2.polylines(frame, [whiteboard_corners], True, (0, 255, 0), 1)
        
        # Perspective transform setup
        pts_dst = np.array([[0, 0], [SCREEN_WIDTH, 0], [SCREEN_WIDTH, SCREEN_HEIGHT], [0, SCREEN_HEIGHT]], dtype=float)
        M, _ = cv2.findHomography(whiteboard_corners.astype(float), pts_dst)

        # Build tracking boundary mask
        roi_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(roi_mask, [whiteboard_corners], 255)
        
        # Dynamic red target thresholding from trackbar inputs
        lower_red1 = np.array([0, min_sat, min_val])
        upper_red1 = np.array([12, 255, 255])
        lower_red2 = np.array([155, min_sat, min_val])
        upper_red2 = np.array([180, 255, 255])

        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        color_red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        # Bright blowout core mask 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, brightness_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        laser_combined_mask = cv2.bitwise_or(color_red_mask, brightness_mask)

        masked_red = cv2.bitwise_and(laser_combined_mask, laser_combined_mask, mask=roi_mask)
        masked_red = cv2.GaussianBlur(masked_red, (5, 5), 0)

        contours, _ = cv2.findContours(masked_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        current_laser_x = None
        current_laser_y = None
        max_area = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area and area > max_area:
                M_moments = cv2.moments(cnt)
                if M_moments["m00"] != 0:
                    current_laser_x = int(M_moments["m10"] / M_moments["m00"])
                    current_laser_y = int(M_moments["m01"] / M_moments["m00"])
                    max_area = area

        # Handle persistence lock state
        if current_laser_x is not None and current_laser_y is not None:
            laser_x, laser_y = current_laser_x, current_laser_y
            laser_active = True
        else:
            laser_active = False

        if laser_x is not None:
            # Map raw coordinates to screen perspective
            laser_point = np.array([[[laser_x, laser_y]]], dtype=float)
            screen_point = cv2.perspectiveTransform(laser_point, M)
            target_screen_x = max(0, min(int(screen_point[0][0][0]), SCREEN_WIDTH - 1))
            target_screen_y = max(0, min(int(screen_point[0][0][1]), SCREEN_HEIGHT - 1))

            # Only process interpolation and mouse movement if the target is actively seen
            if laser_active:
                if prev_screen_x is not None and prev_screen_y is not None:
                    # Apply your smoothing factor equation
                    final_screen_x = int(prev_screen_x + smooth_factor * (target_screen_x - prev_screen_x))
                    final_screen_y = int(prev_screen_y + smooth_factor * (target_screen_y - prev_screen_y))
                else:
                    final_screen_x, final_screen_y = target_screen_x, target_screen_y

                pyautogui.moveTo(final_screen_x, final_screen_y)
                prev_screen_x, prev_screen_y = final_screen_x, final_screen_y

            # -----------------------------------------------------------------
            # VISUAL HUD HIGHLIGHTS
            # -----------------------------------------------------------------
            # Red glow if tracking is active; Orange glow if holding position lock
            hud_color = (0, 0, 255) if laser_active else (0, 140, 255)
            
            overlay = frame.copy()
            cv2.circle(overlay, (laser_x, laser_y), 18, hud_color, -1)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
            
            # Sharp precision dot in center core
            cv2.circle(frame, (laser_x, laser_y), 3, (0, 255, 255), -1)
            
    else:
        # Minimalist warning if a corner goes missing
        cv2.putText(frame, "CALIBRATION LOST", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        # Clear smoothing memory if calibration is broken to avoid wild jumps later
        prev_screen_x, prev_screen_y = None, None

    # Output window
    cv2.imshow("Feed", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()