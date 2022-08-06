
# Importing the libraries
import cv2

# Loading the cascades
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") # loadding cascade for face # cascade is a series of filters that will apply one after another to detect the face
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml") # loadding cascade for face

# Defining a function that will do the detection 
def detect(gray, frame):  # The input is series of images from a single video which is our webcam # gray is the black and white version of our image # frame is our original image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) # The first thing that we need to do is to get the coordinates of the rectangle that detect the face # So we get the 4 elements of x y w h. x and y is the coordinates of the upper left corner, w is the width of rectangle and h is the height of rectangle  # detectMultiScale method will get us the coordinates of the upper left corner and the width and the height of the rectangle that detecting the faces # the first argument in detectMultiScale is the image in black and white. second argument is the scale factor which tells by how much the size of the image will be reduce. 1.3 means that the image will be redice by the factor of 1.3 times. third argument is the minimum number of neighbours which means in order for a zone of pixels to be accepted at least x neighbour zones must also be accepted. # these number came with exprimenting which they are good for the webcam
    for (x, y, w, h) in faces: # Starting a loop to iterate through these faces and for each of these faces we draw a rectangle and we will detect some eyes
        cv2.rectangle(frame, (x, y), (w+x, y+h), (255, 0, 0), 2) # first we want to draw a rectangle # first argument is our frame which is the image for drawing the rectangle. second argument is the coordinate of upper left corner of the rectangle. third argument is the coordinate of lower right corner of rectangle. forth arument is the color of rectangle which we use RGB. fifth argument is the thickness of the edges of rectangles.
        roi_gray = gray[y:y+h, x:x+w] # we are detecting eyes inside the face in order to improve the computation and it's a much more better algorithm # the first thing to do is to get the two region of interest inside rectangles. one region of interest for black and white image which cascade is goin to be applied to detect the eyes and one region of interest for the original color image to draw the rectangle # inside gray[] we put the zone of the ractangle we use range between two numbers
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3) # The structure of eyes is similar to the faces # in oreder to save some computation time we replace the gray by region of interest which is roi_gray # the numbers are gain by exprimenting 
        for (ex, ey, ew, eh) in eyes: # we want to draw a rectangle for the eyes as well # replacing x by ex which is eye_x and so on for others
            cv2.rectangle(roi_color, (ex, ey), (ew+ex, ey+eh), (0, 255, 0), 2) # we replace frame with roi_color because we want to draw it in there
    return frame # we want to return the frame because we draw the rectangles in there

# Doing some face recognition by webcam
video_capture = cv2.VideoCapture(0) # inside this method: 0 if it's the webcam of computer or 1 if the webcam come from a external device (pluged in webcam)    
while True:  # here we want a infinite loop # by wrting 'break' we will get out
    _, frame = video_capture.read() # first step is to get the last frame coming from the webcam which we use the read method # the reason for using underline is because read method returns two method which we are only interested in the second one which is last frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # now we are ready to apply the detect fuction but since in detect we need two argument which one of them is 'gray' then we need to first make this change # cvtColor takes two arguments. the first one is frame. second argument do an average on red, green, blue and that's to take the scale of gray and to get the right contrast of darkness and lightness
    canvas = detect(gray, frame) # canvas is the result of detect function applied to our frame which is the last frame in webcam
    cv2.imshow('Video', canvas) # this is function to display the processd images in an animated way in a window
    if cv2.waitKey(1) & 0xFF == ord('q'): # to stop the webcam and the face detection process if we press 'q'
        break # the same break as told before
video_capture.release() # to turn off the webcam
cv2.destroyAllWindows() # to destory the window that we used our images



