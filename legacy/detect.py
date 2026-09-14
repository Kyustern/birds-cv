import cv2
import numpy as np
import time
# import matplotlib.pyplot as plt
# from ultralytics import YOLO

def preprocess_frame(frame, backSub):

    """
    Preprocessing phase for each frame.
    Currently empty - implement as needed.
    """

    fg_mask = backSub.apply(frame)

        # Find contours
    contours, hierarchy = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cv2.findContours
    # print(contours)
    # print("contours", contours)
    frame_ct = cv2.drawContours(frame, contours, -1, (0, 255, 0), 1)
    # return frame_ct

    GLOBAL_THRESHOLD_MIN = 200
    retval, mask_global_thresh = cv2.threshold( fg_mask, GLOBAL_THRESHOLD_MIN, 255, cv2.THRESH_BINARY)

    # set the kernal
    # TODO: Fine tune the kernel
    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # Apply erosion
    mask_eroded = cv2.morphologyEx(mask_global_thresh, cv2.MORPH_OPEN, erosion_kernel)

    min_contour_area = 175  # Define your minimum area threshold

    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    
    frame_out = frame.copy()
    for cnt in large_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        frame_out = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 200), 3)
    
    # Display the resulting frame
    # cv2.imshow('Frame_final', frame_out)

    return frame_out


def main():
    # Open video file
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("sample_vids/8170-207209141_medium.mp4")

    original_fps = cap.get(cv2.CAP_PROP_FPS)

    backSub = cv2.createBackgroundSubtractorMOG2()

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        elapsed_time = time.time() - start_time
        processed_fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        # Preprocess frame
        processed_frame = preprocess_frame(frame, backSub)

        # Display FPS on frame
        cv2.putText(processed_frame, f"Processed: {processed_fps:.1f} FPS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(processed_frame, f"Original: {original_fps:.1f} FPS", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        cv2.imshow("Cam", processed_frame)
        

        # Run YOLO detection
        # results = model(processed_frame, verbose=False)

        # # Display results
        # results[0].show()

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
