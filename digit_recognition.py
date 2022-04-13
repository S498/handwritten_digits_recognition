# Import Libraries
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# MNIST CNN model
cnn_model = load_model("model.h5")
live_result = cv2.VideoWriter('digits_processed_video.mov',
                              cv2.VideoWriter_fourcc(*'MJPG'), 10, (512, 512))


def digit_recognition(frame):
    while frame.isOpened():
        ret, img = frame.read()
        cv2.imshow("Video", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if ret:
            # Resizing the image frame
            image = cv2.resize(img, (512, 512))
            # Converting the image to gray scale
            imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Canny edge detector
            thresh = cv2.Canny(image, 100, 200)

            contours, hier = cv2.findContours(
                thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rectangles = [cv2.boundingRect(contour) for contour in contours]
            # Processing Each ROI
            for rectangle in rectangles:
                area = rectangle[2] * rectangle[3]
                # Considering the ROI's which has area > 1000 and the remaining as noise
                if area < 1000:
                    continue

                # Make the rectangular region around the digit
                leng = int(rectangle[3] * 1.5)
                roi = thresh[int(rectangle[1] + rectangle[3] // 2 - leng // 2):int(rectangle[1] + rectangle[3] // 2 - leng // 2) +
                             leng, int(rectangle[0] + rectangle[2] // 2 - leng // 2):int(rectangle[0] + rectangle[2] // 2 - leng // 2) + leng]

                if roi.shape[1] == 0:
                    leng = rectangle[3]
                    roi = thresh[int(rectangle[1] + rectangle[3] // 2 - leng // 2):int(rectangle[1] + rectangle[3] // 2 - leng // 2) + leng, int(
                        rectangle[0] + rectangle[2] // 2 - leng // 2):int(rectangle[0] + rectangle[2] // 2 - leng // 2) + leng]
                try:
                    # Displaying each ROI
                    cv2.imshow("Region Of Interest", roi)
                    cv2.waitKey(60)
                    # Resizing the ROI to 28*28
                    roi = cv2.resize(
                        roi, (28, 28), interpolation=cv2.INTER_AREA)
                    roi = roi.astype("float32")
                    roi = roi.reshape(-1, 28, 28, 1)
                    # Predicting the label for ROI
                    label = cnn_model.predict(roi)
                    label = np.argmax(label)
                    # Draw rectangle
                    cv2.rectangle(
                        image, (rectangle[0], rectangle[1]), (rectangle[0] + rectangle[2], rectangle[1] + rectangle[3]), (105, 105, 105), 6)
                    cv2.putText(image, str(label),
                                (rectangle[0], rectangle[1]), 1, 2, (255, 255, 255), 2)
                except Exception as e:
                    print(e)
            cv2.imshow("Final Image with ROIs", image)
            live_result.write(image)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        else:
            break


if __name__ == "__main__":
    live_video = input(
        "Do you want to use live Input? (Yes[y]/No[n]): ").lower()
    if live_video == "yes" or live_video == "y":
        frames = cv2.VideoCapture(0)
        frames.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        digit_recognition(frames)
    else:
        file_name = ""
        frames = None
        try:
            file_name = input("Please Enter The Video File Name :")
            frames = cv2.VideoCapture(file_name)
        except Exception as e:
            print("An Exception Occurred while loading the file {}".format(file_name))
            frames = None
        if frames:
            if not frames.isOpened():
                print("Error opening video stream or file")
            digit_recognition(frames)
