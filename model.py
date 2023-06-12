import cv2
import torch
import numpy as np

class ParkingManagement:

    """
    This class is used to count the number of cars in the parking.

    Attributes:
        model: The model used to detect the cars.
        area: The area of the parking.
    """


    def __init__(self):

        """
        The constructor for ParkingManagement class.
        """

        # Load the model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

        # Define the area of the parking    
        self.area = [(5, 719), (3, 315), (1245, 274), (1279, 718)]


    def run(self):
        """
        This function is used to run the parking management.
        """

        # Create a window to show the frame
        cv2.namedWindow('FRAME')

        # Set the mouse callback function
        cv2.setMouseCallback('FRAME', self.points)

        # Create a video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 

        # save the video
        out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))

        # Read the video
        cap = cv2.VideoCapture('./video.mp4')

        # Check if the video is opened
        while True:

            # Read the frame
            ret, frame = cap.read()
            if not ret:
                break

            # Pass the frame to the model to detect the cars
            results = self.model(frame) 

            # Get the points of the cars
            points = self.get_points(results.pandas().xyxy[0])

            # Get the number of cars
            num_cars = len(points)

            # Draw the objects
            frame = self.draw_objects(frame, results, num_cars)

            cv2.imshow('FRAME', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()


    def points(self, event, x, y, flags, param):
        """
        This function is used to get the points of the parking.
        Args:
            event: The event of the mouse.
            x: The x coordinate of the mouse.
            y: The y coordinate of the mouse.
            flags: The flags of the mouse.
            param: The param of the mouse.
        """ 

        # Check if the event is mouse click
        if event == cv2.EVENT_MOUSEMOVE:
            colorsBGR = [x, y]
            # print(colorsBGR)

    
    def get_points(self, data):
        """
        This function is used to get the points of the cars.

        Args:
            data: The data of the cars.

        Returns:
            points: The points of the cars.
        """
        points = [] # The points of the cars

        # loop over the data and get the points of the cars
        for _ , row in data.iterrows():
            x1 = int(row['xmin'])
            y1 = int(row['ymin'])
            x2 = int(row['xmax'])
            y2 = int(row['ymax'])

            d = row['name'] # The name of the object
            cx = int((x1 + x2) // 2) # The x coordinate of the center of the car
            cy = int((y1 + y2) // 2) # The y coordinate of the center of the car

            # Check if the object is a car
            if  'car' in d:
                result = cv2.pointPolygonTest(np.array(self.area, np.int32), (cx, cy), False)  # detect if the car is in the parking
                
                # Check if the car is detected in the parking then add it to the points
                if result == 1:
                    points.append(cx)

        return points

    def draw_objects(self, frame, results, num_cars):

        """
        This function is used to draw the  objects.

        Returns:
            frame: The frame with the objects.
        """
        
        for _ , row in results.pandas().xyxy[0].iterrows():
            x1 = int(row['xmin'])
            y1 = int(row['ymin'])
            x2 = int(row['xmax'])
            y2 = int(row['ymax'])
            d = row['name']

            if 'car' in d:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)  
                cv2.putText(frame, str(d), (x1, y1), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        # Draw the area of the parking
        # cv2.polylines(frame, np.array([self.area]), False, (0, 0, 255), 2)
        
        # Write the number of cars in the parking
        cv2.putText(frame, 'number of cars in parking = ' + str(num_cars), (50, 80), cv2.FONT_HERSHEY_PLAIN, 2,
                    (0, 0, 255), 2)

        
        return frame


if __name__ == '__main__':
    
    # Create an object of ParkingManagement class
    parking_management = ParkingManagement()

    # Run the parking management
    parking_management.run()