import cv2

cap = cv2.VideoCapture(0)  # Try 1, 2 if 0 doesn't work

if not cap.isOpened():
    print("ERROR: Camera not accessible")
else:
    print("Camera detected - press 'q' to exit")
    
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame")
        break
    cv2.imshow('Test', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()