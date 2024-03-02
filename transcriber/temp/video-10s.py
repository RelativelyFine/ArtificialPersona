import cv2
import time

def record_video(duration=10, save_path='output.avi'):
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(save_path, fourcc, 20.0, (640, 480))
    
    cap = cv2.VideoCapture(0)
    
    # Set resolution to 480p
    cap.set(3, 640)  # Width
    cap.set(4, 480)  # Height
    
    start_time = time.time()
    while int(time.time() - start_time) < duration:
        ret, frame = cap.read()
        if ret:
            # Write the frame into the file 'output.avi'
            out.write(frame)
            
            # Display the resulting frame
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Quit if 'q' is pressed
                break
        else:
            break
    
    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

record_video(10, 'output.avi')
