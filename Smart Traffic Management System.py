import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained vehicle classification model
model = load_model('vehicle_classifier.h5')

# Function to preprocess images before classification
def preprocess_image(image):
    image = cv2.resize(image, (17, 38))  # Resize to match training size
    image = image.astype('float32') / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to classify vehicle
def classify_vehicle(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    vehicle_classes = ['car', 'ambulance', 'bike', 'truck']
    return vehicle_classes[np.argmax(predictions)]

# Count of vehicles passing the stop line
vehicle_counts = {'right': 0, 'left': 0, 'down': 0, 'up': 0}

# Function to load images
def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Warning: Could not load image {path}")
    return img

# Load intersection background
intersection_image = load_image('images/mod_int.png')

# Load vehicle images
vehicle_images = {
    'right': {'car': load_image('images/right/car.png'), 
              'truck': load_image('images/right/truck.png'), 
              'bike': load_image('images/right/bike.png'),
              'ambulance': load_image('images/right/bus.png')},
    'left': {'car': load_image('images/left/car.png'), 
             'truck': load_image('images/left/truck.png'), 
             'bike': load_image('images/left/bike.png'),
             'ambulance': load_image('images/left/bus.png')},
    'down': {'car': load_image('images/down/car.png'), 
             'truck': load_image('images/down/truck.png'), 
             'bike': load_image('images/down/bike.png'),
             'ambulance': load_image('images/down/bus.png')},
    'up': {'car': load_image('images/up/car.png'), 
           'truck': load_image('images/up/truck.png'), 
           'bike': load_image('images/up/bike.png'),
           'ambulance': load_image('images/up/bus.png')}
}

# Load traffic light images
red_light = load_image('images/signals/red.png')
yellow_light = load_image('images/signals/yellow.png')
green_light = load_image('images/signals/green.png')

# Traffic light positions for each direction
traffic_light_positions = {
    'top': (550, 220),
    'right': (800, 220),
    'bottom': (800, 550),
    'left': (550, 550)
}

# Map vehicle directions to traffic light directions
direction_mapping = {
    'left': 'bottom',
    'right': 'top',
    'up': 'left',
    'down': 'right'
}

# Define lanes and lane positions
lane_positions = {
    'right': [375, 400, 350],  # y-coordinates for lanes
    'left': [435, 460, 495],
    'down': [695, 720, 750],
    'up': [625, 650, 600]
}

# Vehicle data (starting positions, directions, types, and lanes)
vehicles = [
    
    {'position': [-1000, lane_positions['right'][0]], 'direction': 'right', 'type': 'car', 'speed': 3, 'passed': False},
    {'position': [-50, lane_positions['right'][1]], 'direction': 'right', 'type': 'car', 'speed': 5, 'passed': False},
    {'position': [-50, lane_positions['right'][1]], 'direction': 'right', 'type': 'car', 'speed': 3, 'passed': False},
    {'position': [-150, lane_positions['right'][2]], 'direction': 'right', 'type': 'bike', 'speed': 5, 'passed': False},
    {'position': [-500, lane_positions['right'][2]], 'direction': 'right', 'type': 'bike', 'speed': 5, 'passed': False},
    {'position': [-80, lane_positions['right'][0]], 'direction': 'right', 'type': 'car', 'speed': 3, 'passed': False},
    {'position': [-1200, lane_positions['right'][0]], 'direction': 'right', 'type': 'car', 'speed': 3, 'passed': False},
    {'position': [-80, lane_positions['right'][1]], 'direction': 'right', 'type': 'truck', 'speed': 2, 'passed': False},
    {'position': [-5800, lane_positions['right'][1]], 'direction': 'right', 'type': 'car', 'speed': 3, 'passed': False},
    {'position': [-200, lane_positions['right'][2]], 'direction': 'right', 'type': 'bike', 'speed': 5, 'passed': False},
    {'position': [-600, lane_positions['right'][2]], 'direction': 'right', 'type': 'bike', 'speed': 5, 'passed': False},



    {'position': [1400, lane_positions['left'][0]], 'direction': 'left', 'type': 'ambulance', 'speed': 6, 'passed': False},
    {'position': [1400, lane_positions['left'][1]], 'direction': 'left', 'type': 'truck', 'speed': 2, 'passed': False},
    {'position': [1400, lane_positions['left'][2]], 'direction': 'left', 'type': 'bike', 'speed': 5, 'passed': False},
    {'position': [1550, lane_positions['left'][2]], 'direction': 'left', 'type': 'bike', 'speed': 5, 'passed': False},
    {'position': [2800, lane_positions['left'][0]], 'direction': 'left', 'type': 'car', 'speed': 3, 'passed': False},
    #{'position': [2900, lane_positions['left'][0]], 'direction': 'left', 'type': 'car', 'speed': 3, 'passed': False},
    #{'position': [2800, lane_positions['left'][1]], 'direction': 'left', 'type': 'truck', 'speed': 2, 'passed': False},
    #{'position': [2800, lane_positions['left'][2]], 'direction': 'left', 'type': 'bike', 'speed': 5, 'passed': False},
    #{'position': [3100, lane_positions['left'][2]], 'direction': 'left', 'type': 'bike', 'speed': 5, 'passed': False},
    #{'position': [3150, lane_positions['left'][2]], 'direction': 'left', 'type': 'bike', 'speed': 5, 'passed': False},
    {'position': [6000, lane_positions['left'][0]], 'direction': 'left', 'type': 'car', 'speed': 3, 'passed': False},
    {'position': [6000, lane_positions['left'][1]], 'direction': 'left', 'type': 'truck', 'speed': 2, 'passed': False},
    {'position': [6000, lane_positions['left'][2]], 'direction': 'left', 'type': 'bike', 'speed': 5, 'passed': False},
    {'position': [5000, lane_positions['left'][2]], 'direction': 'left', 'type': 'bike', 'speed': 5, 'passed': False},
    {'position': [5000, lane_positions['left'][0]], 'direction': 'left', 'type': 'car', 'speed': 3, 'passed': False},
    {'position': [5100, lane_positions['left'][0]], 'direction': 'left', 'type': 'car', 'speed': 3, 'passed': False},
    {'position': [5200, lane_positions['left'][0]], 'direction': 'left', 'type': 'car', 'speed': 3, 'passed': False},
    #{'position': [5300, lane_positions['left'][0]], 'direction': 'left', 'type': 'car', 'speed': 3, 'passed': False},
    #{'position': [5400, lane_positions['left'][0]], 'direction': 'left', 'type': 'car', 'speed': 3, 'passed': False},
    #{'position': [5500, lane_positions['left'][0]], 'direction': 'left', 'type': 'car', 'speed': 3, 'passed': False},
    #{'position': [5600, lane_positions['left'][0]], 'direction': 'left', 'type': 'car', 'speed': 3, 'passed': False},
    #{'position': [4100, lane_positions['left'][1]], 'direction': 'left', 'type': 'truck', 'speed': 2, 'passed': False},
    #{'position': [5100, lane_positions['left'][2]], 'direction': 'left', 'type': 'bike', 'speed': 5, 'passed': False},
    #{'position': [5200, lane_positions['left'][2]], 'direction': 'left', 'type': 'bike', 'speed': 5, 'passed': False},
    #{'position': [5300, lane_positions['left'][2]], 'direction': 'left', 'type': 'bike', 'speed': 5, 'passed': False},


    {'position': [lane_positions['down'][1], -50], 'direction': 'down', 'type': 'car', 'speed': 4, 'passed': False},
    {'position': [lane_positions['down'][0], -50], 'direction': 'down', 'type': 'car', 'speed': 3, 'passed': False},
    {'position': [lane_positions['down'][1], -150], 'direction': 'down', 'type': 'truck', 'speed': 3, 'passed': False},
    #{'position': [lane_positions['down'][2], -50], 'direction': 'down', 'type': 'bike', 'speed': 5, 'passed': False},
    #{'position': [lane_positions['down'][2], -200], 'direction': 'down', 'type': 'bike', 'speed': 5, 'passed': False},
    #{'position': [lane_positions['down'][0], -150], 'direction': 'down', 'type': 'car', 'speed': 3, 'passed': False},
    #{'position': [lane_positions['down'][1], -1500], 'direction': 'down', 'type': 'truck', 'speed': 3, 'passed': False},
    #{'position': [lane_positions['down'][2], -1500], 'direction': 'down', 'type': 'bike', 'speed': 5, 'passed': False},
    #{'position': [lane_positions['down'][2], -3000], 'direction': 'down', 'type': 'bike', 'speed': 5, 'passed': False},
    #{'position': [lane_positions['down'][0], -2000], 'direction': 'down', 'type': 'car', 'speed': 3, 'passed': False},
    #{'position': [lane_positions['down'][1], -2000], 'direction': 'down', 'type': 'truck', 'speed': 2, 'passed': False},
    #{'position': [lane_positions['down'][2], -2000], 'direction': 'down', 'type': 'bike', 'speed': 5, 'passed': False},
    #{'position': [lane_positions['down'][2], -4000], 'direction': 'down', 'type': 'bike', 'speed': 5, 'passed': False},
    #{'position': [lane_positions['down'][0], -3000], 'direction': 'down', 'type': 'car', 'speed': 3, 'passed': False},
    #{'position': [lane_positions['down'][1], -3000], 'direction': 'down', 'type': 'truck', 'speed': 2, 'passed': False},
    #{'position': [lane_positions['down'][2], -3000], 'direction': 'down', 'type': 'bike', 'speed': 5, 'passed': False},
    {'position': [lane_positions['down'][2], -9000], 'direction': 'down', 'type': 'bike', 'speed': 5, 'passed': False},
    {'position': [lane_positions['down'][0], -9000], 'direction': 'down', 'type': 'car', 'speed': 3, 'passed': False},


    {'position': [lane_positions['up'][0], 850], 'direction': 'up', 'type': 'car', 'speed': 3, 'passed': False},
    {'position': [lane_positions['up'][1], 850], 'direction': 'up', 'type': 'truck', 'speed': 2, 'passed': False},
    {'position': [lane_positions['up'][1], 1000], 'direction': 'up', 'type': 'truck', 'speed': 2, 'passed': False},
    {'position': [lane_positions['up'][2], 850], 'direction': 'up', 'type': 'bike', 'speed': 5, 'passed': False},
    {'position': [lane_positions['up'][2], 1000], 'direction': 'up', 'type': 'bike', 'speed': 5, 'passed': False},
    {'position': [lane_positions['up'][2], 2500], 'direction': 'up', 'type': 'bike', 'speed': 5, 'passed': False},
    {'position': [lane_positions['up'][2], 2600], 'direction': 'up', 'type': 'bike', 'speed': 5, 'passed': False},
    {'position': [lane_positions['up'][0], 2500], 'direction': 'up', 'type': 'car', 'speed': 3, 'passed': False},
    {'position': [lane_positions['up'][0], 2600], 'direction': 'up', 'type': 'car', 'speed': 3, 'passed': False},
    {'position': [lane_positions['up'][0], 2700], 'direction': 'up', 'type': 'car', 'speed': 3, 'passed': False},
    {'position': [lane_positions['up'][1], 2500], 'direction': 'up', 'type': 'truck', 'speed': 2, 'passed': False},
    {'position': [lane_positions['up'][2], 1100], 'direction': 'up', 'type': 'bike', 'speed': 5, 'passed': False},
    {'position': [lane_positions['up'][2], 2700], 'direction': 'up', 'type': 'bike', 'speed': 5, 'passed': False},
    {'position': [lane_positions['up'][2], 2800], 'direction': 'up', 'type': 'bike', 'speed': 5, 'passed': False},
    {'position': [lane_positions['up'][0], 2800], 'direction': 'up', 'type': 'car', 'speed': 3, 'passed': False},
    {'position': [lane_positions['up'][0], 2900], 'direction': 'up', 'type': 'car', 'speed': 3, 'passed': False},
    {'position': [lane_positions['up'][0], 3000], 'direction': 'up', 'type': 'car', 'speed': 3, 'passed': False},
    {'position': [lane_positions['up'][1], 2600], 'direction': 'up', 'type': 'truck', 'speed': 2, 'passed': False},
    {'position': [lane_positions['up'][1], 2700], 'direction': 'up', 'type': 'truck', 'speed': 2, 'passed': False},
    {'position': [lane_positions['up'][1], 2800], 'direction': 'up', 'type': 'truck', 'speed': 2, 'passed': False},
    {'position': [lane_positions['up'][1], 2900], 'direction': 'up', 'type': 'truck', 'speed': 2, 'passed': False}
]

# Stop positions for traffic lights
stop_positions = {
    'top': 585,
    'right': 330,
    'bottom': 750,
    'left': 495
}

# Minimum gap between vehicles in the same lane
min_gap = 70

# Overlay function with boundary checks
def overlay_object(frame, object_image, position):
    if object_image is None:
        return frame  # Skip if the image is missing

    x, y = position
    h, w, _ = object_image.shape
    frame_h, frame_w, _ = frame.shape

    # Ensure overlay stays within frame bounds
    if y + h > frame_h or x + w > frame_w or x < 0 or y < 0:
        return frame

    alpha_s = object_image[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(3):
        frame[y:y+h, x:x+w, c] = (alpha_s * object_image[:, :, c] +
                                  alpha_l * frame[y:y+h, x:x+w, c])

    return frame

# Function to display traffic lights and change their state
def change_traffic_lights(frame, active_light, light_state):
    # Go through each direction and set the light state
    for direction, position in traffic_light_positions.items():
        # Select the appropriate light image based on the current state
        if direction == active_light:
            if light_state == 'green':
                light_image = green_light  # Show green light for active direction
            elif light_state == 'yellow':
                light_image = yellow_light  # Show yellow light for active direction
            elif light_state == 'red':
                light_image = red_light  # Show red light for active direction
        else:
            light_image = red_light  # All other directions show red light

        # Overlay the selected traffic light on the frame
        frame = overlay_object(frame, light_image, position)

    return frame


# Global variable declarations (at the top level)
emergency_vehicle_detected = False
emergency_direction = None
emergency_mode = False
emergency_time = 0
emergency_lane_active = False
vehicle_counts = {'right': 0, 'left': 0, 'down': 0, 'up': 0} #Initialize the dictionary
next_vehicle_counts = {'right': 0, 'left': 0, 'down': 0, 'up': 0}

# ... (Existing code for traffic light positions, direction mapping, lane positions, vehicle data, stop positions, min_gap, overlay function, etc.)


def move_vehicle(vehicle):
    global emergency_vehicle_detected, emergency_mode, emergency_time, emergency_lane_active, emergency_direction  # Globals at the VERY beginning

    direction = vehicle['direction']
    mapped_direction = direction_mapping[direction]
    stop_position = stop_positions.get(mapped_direction)
    middlestop = 500

    # Count vehicles before the stop line (halfway)
    if stop_position is not None and not vehicle['passed']:
        if direction == 'right' and stop_position - (middlestop - 50) <= vehicle['position'][0] + 50 <= stop_position:
            vehicle_counts[direction] += 1
        elif direction == 'left' and stop_position <= vehicle['position'][0] - 50 <= stop_position + (middlestop - 50):
            vehicle_counts[direction] += 1
        elif direction == 'down' and stop_position - (middlestop - 200) <= vehicle['position'][1] + 50 <= stop_position:
            vehicle_counts[direction] += 1
        elif direction == 'up' and stop_position <= vehicle['position'][1] - 50 <= stop_position + (middlestop - 300):
            vehicle_counts[direction] += 1

    # Emergency Vehicle Handling
    if vehicle['type'] == 'ambulance' and not vehicle['passed']:
        if not emergency_mode:
            emergency_vehicle_detected = True
            emergency_direction = vehicle['direction']  # Set the GLOBAL here
            emergency_mode = True
            emergency_time = 10 * 10
            emergency_lane_active = True

    # Stop at the stop line if the light is red (only if not in emergency mode or ambulance is not in the same direction)
    if stop_position is not None and not vehicle['passed'] and not (emergency_mode and emergency_direction == direction):
        if active_light != mapped_direction:  # Red light for this direction
            if direction == 'right' and vehicle['position'][0] + 50 >= stop_position:
                vehicle['position'][0] = stop_position - 50
                return
            elif direction == 'left' and vehicle['position'][0] - 50 <= stop_position:
                vehicle['position'][0] = stop_position + 50
                return
            elif direction == 'down' and vehicle['position'][1] + 50 >= stop_position:
                vehicle['position'][1] = stop_position - 50
                return
            elif direction == 'up' and vehicle['position'][1] - 50 <= stop_position:
                vehicle['position'][1] = stop_position + 50
                return

    # Check for vehicles ahead in the same lane
    vehicle_can_move = True  # Assume the vehicle can move unless blocked

    for other_vehicle in vehicles:
        if other_vehicle is vehicle:
            continue

        if other_vehicle['direction'] == direction:
            if direction in ['right', 'left']:
                if other_vehicle['position'][1] == vehicle['position'][1]:  # Same Y coordinate (same lane)
                    if direction == 'right' and 0 < other_vehicle['position'][0] - vehicle['position'][0] <= min_gap:
                        vehicle_can_move = False
                    elif direction == 'left' and 0 < vehicle['position'][0] - other_vehicle['position'][0] <= min_gap:
                        vehicle_can_move = False
            elif direction in ['up', 'down']:
                if other_vehicle['position'][0] == vehicle['position'][0]:  # Same X coordinate (same lane)
                    if direction == 'down' and 0 < other_vehicle['position'][1] - vehicle['position'][1] <= min_gap:
                        vehicle_can_move = False
                    elif direction == 'up' and 0 < vehicle['position'][1] - other_vehicle['position'][1] <= min_gap:
                        vehicle_can_move = False

    # Mark vehicle as passed once it crosses the junction
    if stop_position is not None:
        if direction == 'right' and vehicle['position'][0] > stop_position:
            vehicle['passed'] = True
        elif direction == 'left' and vehicle['position'][0] < stop_position:
            vehicle['passed'] = True
        elif direction == 'down' and vehicle['position'][1] > stop_position:
            vehicle['passed'] = True
        elif direction == 'up' and vehicle['position'][1] < stop_position:
            vehicle['passed'] = True

    # Move the vehicle based on its direction and speed if lane is clear
    if vehicle_can_move:
        if direction == 'right':
            vehicle['position'][0] += vehicle['speed']
        elif direction == 'left':
            vehicle['position'][0] -= vehicle['speed']
        elif direction == 'down':
            vehicle['position'][1] += vehicle['speed']
        elif direction == 'up':
            vehicle['position'][1] -= vehicle['speed']


# Function to draw vehicles and remove them once they exit the frame
def draw_vehicles(frame):
    global vehicles
    for vehicle in vehicles[:]:
        direction = vehicle['direction']
        vehicle_image = vehicle_images[direction][vehicle['type']]

        if (direction == 'right' and vehicle['position'][0] > 1350) or \
           (direction == 'left' and vehicle['position'][0] < -100) or \
           (direction == 'down' and vehicle['position'][1] > 850) or \
           (direction == 'up' and vehicle['position'][1] < -100):
            vehicles.remove(vehicle)
        else:
            frame = overlay_object(frame, vehicle_image, vehicle['position'])

    return frame

def draw_timer(frame, active_light, next_light, remaining_time, emergency_time=None):
    red_timer_positions = {
        'top': (500, 250),    # Red light for top lane
        'right': (870, 250),  # Red light for right lane
        'bottom': (870, 580), # Red light for bottom lane
        'left': (500, 580)    # Red light for left lane
    }

    green_timer_positions = {
        'top': (500, 250),    # Green light for top lane
        'right': (870, 250),  # Green light for right lane
        'bottom': (870, 580), # Green light for bottom lane
        'left': (500, 580)    # Green light for left lane
    }

    display_time = f"{remaining_time // 10}"  # Convert simulation steps to real seconds

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2

    # Draw countdown for active green light
    if active_light in green_timer_positions:
        position = green_timer_positions[active_light]
        color = (0, 255, 0)  # Green color
        cv2.putText(frame, display_time, position, font, font_scale, color, thickness, cv2.LINE_AA)

    # Draw countdown for upcoming red light
    if next_light in red_timer_positions:
        position = red_timer_positions[next_light]
        color = (0, 0, 255)  # Red color
        cv2.putText(frame, display_time, position, font, font_scale, color, thickness, cv2.LINE_AA)

    # Draw Emergency Timer (if active)
    if emergency_time is not None and emergency_time > 0:
        emergency_display_time = f"Emergency Vehicle Detected: {emergency_time // 10}"
        # Position Emergency timer - you'll likely want to adjust this position
        emergency_position = (10, 50)  # Example: top-left corner
        color = (0, 0, 255)  # Red for emergency timer (you can change it)
        cv2.putText(frame, emergency_display_time, emergency_position, font, font_scale, color, thickness, cv2.LINE_AA)

    return frame

# Function to display vehicle counts on the frame
def display_vehicle_count(frame):
    font = cv2.FONT_HERSHEY_SIMPLEX
    position = {'right': (50, 50), 'left': (50, 100), 'down': (50, 150), 'up': (50, 200)}
    for direction, count in vehicle_counts.items():
        text = f"{direction.capitalize()} Lane: {count}"
        cv2.putText(frame, text, position[direction], font, 1, (0, 255, 0), 2, cv2.LINE_AA)



# Traffic light sequence
traffic_light_order = ['top', 'right', 'bottom', 'left']
current_light_index = 0
frame_delay = 100  # Adjust for simulation speed

time_yellow = 3 * 10  # 3 seconds for yellow light

light_state = 'green'  # Start with green light

vehicle_counts = {'right': 0, 'left': 0, 'down': 0, 'up': 0}
next_vehicle_counts = {'right': 0, 'left': 0, 'down': 0, 'up': 0}

def get_green_light_time(vehicle_counts):
    """Determine the green light duration based on vehicle count."""
    max_count = max(vehicle_counts.values())  # Get the highest vehicle count
    if 0 <= max_count <= 2:
        return 5 * 10  # 5 seconds
    elif 3 <= max_count <= 7:
        return 10 * 10  # 10 seconds
    elif 8 <= max_count <= 15:
        return 15 * 10  # 15 seconds
    else:
        return 20 * 10  # 20 seconds

# Set initial green light time
remaining_time = get_green_light_time(vehicle_counts)



for t in range(10000):  # Long simulation duration
    frame = intersection_image.copy()

    # Emergency Handling
    if emergency_mode:  # Check the flag in the main loop
        active_light = direction_mapping[emergency_direction]  # Set active light
        light_state = 'green'  # Set light state
        frame = change_traffic_lights(frame, active_light, light_state)  # Update the lights

        emergency_time -= 1  # DECREMENT THE EMERGENCY TIMER HERE!

        if emergency_time <= 0 or not any(v['type'] == 'ambulance' and v['direction'] == emergency_direction for v in vehicles):  # Check if ambulance is still present
            emergency_mode = False
            emergency_lane_active = False

            # Reset traffic light cycle (Important!)
            try:
                next_light_index = (traffic_light_order.index(emergency_direction) + 1) % len(traffic_light_order)
                active_light = traffic_light_order[next_light_index]
            except ValueError:  # Handle if emergency_direction is not in traffic_light_order
                active_light = traffic_light_order[0]  # Default to the first light
                print("Warning: Emergency direction not found in traffic_light_order. Resetting to default light.")
            light_state = 'green'

            # *** Correctly update remaining_time ***
            remaining_time = get_green_light_time(vehicle_counts)  # Get the initial green light time for the new active_light

            vehicle_counts = next_vehicle_counts.copy()
            next_vehicle_counts = {'right': 0, 'left': 0, 'down': 0, 'up': 0}

    elif remaining_time <= 0:
        if light_state == 'green':
            light_state = 'yellow'
            remaining_time = time_yellow  # Set yellow light time
        else:
            # Move to the next traffic light
            current_light_index = (current_light_index + 1) % len(traffic_light_order)
            active_light = traffic_light_order[current_light_index]
            light_state = 'green'
            remaining_time = get_green_light_time(vehicle_counts)  # Reset countdown based on updated vehicle counts

            # Reset counts for the next cycle
            vehicle_counts = next_vehicle_counts.copy()
            next_vehicle_counts = {'right': 0, 'left': 0, 'down': 0, 'up': 0}

    else:
        active_light = traffic_light_order[current_light_index]
        remaining_time -= 1  # DECREMENT THE NORMAL TIMER HERE!

    frame = change_traffic_lights(frame, active_light, light_state)

    # Determine the next light in order
    try:
        next_light_index = (current_light_index + 1) % len(traffic_light_order)
        next_light = traffic_light_order[next_light_index]
    except IndexError:  # handle if index is not available
        next_light = traffic_light_order[0]
        print("Warning: Index is not available. Resetting to default light.")

    frame = draw_timer(frame, active_light, next_light, remaining_time, emergency_time)  # Pass emergency_time

    vehicle_counts = {'right': 0, 'left': 0, 'down': 0, 'up': 0}

    # Move vehicles and count them for the *NEXT* cycle
    for vehicle in vehicles:
        move_vehicle(vehicle)  # Call move_vehicle
        if not vehicle['passed']:
            next_vehicle_counts[vehicle['direction']] += 1

    frame = draw_vehicles(frame)

    # Display vehicle count
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    text_positions = {'right': (450, 200), 'left': (805, 545), 'down': (800, 200), 'up': (470, 546)}
    for direction, count in vehicle_counts.items():
        cv2.putText(frame, f'{direction.capitalize()} Lane: {count}', text_positions[direction], font, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)

    cv2.imshow('Traffic Simulation', frame)
    if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()