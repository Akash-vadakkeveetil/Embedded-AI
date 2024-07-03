# Intelligent Systems: Advanced Embedded AI for Real-World Problems
>

IR Censor code
```PYTHON
import RPi.GPIO as GPIO
import time

# Pin configuration
IR_SENSOR_PIN = 17  # GPIO17 (Pin 11)

# Setup GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(IR_SENSOR_PIN, GPIO.IN)

# IR sensor function
def read_ir_sensor():
    if GPIO.input(IR_SENSOR_PIN) == GPIO.HIGH:
        return True  # Object detected
    else:
        return False  # No object detected

# Main loop
try:
    object_detected = False  # Initialize object detection status
    while True:
        current_state = read_ir_sensor()
        
        if current_state and not object_detected:
            print("Object detected!")
            object_detected = True
        elif not current_state and object_detected:
            print("Object not detected")
            object_detected = False
        
        time.sleep(0.1)  # Delay to avoid CPU load

except KeyboardInterrupt:
    print("Program terminated by user")

finally:
    GPIO.cleanup()
```

## The image processing

We are gonna use opencv for image processing

```PYTHON
import cv2
  
img = cv2.imread('unnamed.png')#the image which we need to process 
print(img)
cv2.imshow('title-of-theOpenCV-window',img)
cv2.waitKey()
cv2.destroyAllWindows()

```

To resize the image add the commad 
```PYTHON
resize_im = cv2.resize(img,(1200,400))
```

Detecting object or movemnts in the screen and saving it to sysstem if the movement detected

```PYTHON
import cv2
import time
import os

# Initialize the camera
camera = cv2.VideoCapture(0)  # Use 0 for the default camera, adjust if using a different camera source

# Initialize variables for motion detection
first_frame = None
motion_detected = False

# Motion detection parameters
MIN_AREA = 20000  # Minimum area size for an object to be considered as motion (adjust as needed)
THRESHOLD_SENSITIVITY = 50  # Threshold sensitivity for motion detection (adjust as needed)
MOVEMENT_DURATION = 2.0  # Minimum duration of continuous movement to be considered as valid motion (adjust as needed)

# Initialize time for image capture delay
last_capture_time = time.time()
capture_interval = 2  # Interval in seconds between each image capture

# Create directory for saving images if it does not exist
save_dir = 'images'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Main loop
while True:
    # Capture frame-by-frame
    ret, frame = camera.read()
    
    # Convert frame to grayscale and blur it
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
    # Initialize the first frame
    if first_frame is None:
        first_frame = gray
        continue
    
    # Compute absolute difference between the current frame and first frame
    frame_delta = cv2.absdiff(first_frame, gray)
    thresh = cv2.threshold(frame_delta, THRESHOLD_SENSITIVITY, 255, cv2.THRESH_BINARY)[1]
    
    # Dilate the thresholded image to fill in holes, then find contours
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Check for motion
    motion_detected = False
    for contour in contours:
        if cv2.contourArea(contour) < MIN_AREA:
            continue
        
        # Calculate bounding box for the contour
        (x, y, w, h) = cv2.boundingRect(contour)
        
        # Draw bounding box around moving object (optional)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Check if enough time has passed since last capture
        current_time = time.time()
        if current_time - last_capture_time >= capture_interval:
            # Save the image when motion is detected
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(save_dir, f"motion_{timestamp}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Motion detected! Image saved as {filename}")
            
            # Update last capture time
            last_capture_time = current_time
        
        # Set flag indicating motion is detected
        motion_detected = True
    
    # Display the frame
    cv2.imshow('Motion Detection', frame)
    
    # Wait for 'q' key to quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Cleanup
camera.release()
cv2.destroyAllWindows()
```
