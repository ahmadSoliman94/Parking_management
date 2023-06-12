import cv2
import torch
import numpy as np


class ParkingManagement:
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.area = [(5, 719), (3, 315), (943, 296), (1279, 718)]

    def run(self):
        cv2.namedWindow('FRAME')
        cv2.setMouseCallback('FRAME', self.points)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
        cap = cv2.VideoCapture('./video.mp4')

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame)
            points = self.get_points(results.pandas().xyxy[0])

            num_cars = len(points)

            frame = self.draw_objects(frame, results, points, num_cars)

            cv2.imshow('FRAME', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def points(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            colorsBGR = [x, y]
            print(colorsBGR)

    def get_points(self, data):
        points = []
        for index, row in data.iterrows():
            x1 = int(row['xmin'])
            y1 = int(row['ymin'])
            x2 = int(row['xmax'])
            y2 = int(row['ymax'])

            d = row['name']
            cx = int((x1 + x2) // 2)
            cy = int((y1 + y2) // 2)

            if 'car' in d:
                result = cv2.pointPolygonTest(np.array(self.area, np.int32), (cx, cy), False)
                if result == 1:
                    points.append(cx)

        return points

    def draw_objects(self, frame, results, points, num_cars):
        for _ , row in results.pandas().xyxy[0].iterrows():
            x1 = int(row['xmin'])
            y1 = int(row['ymin'])
            x2 = int(row['xmax'])
            y2 = int(row['ymax'])
            d = row['name']

            if 'car' in d:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, str(d), (x1, y1), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        # cv2.polylines(frame, np.array([self.area]), True, (0, 0, 255), 2)
        cv2.putText(frame, 'number of cars in parking = ' + str(num_cars), (50, 80), cv2.FONT_HERSHEY_PLAIN, 2,
                    (0, 0, 255), 2)

        return frame


if __name__ == '__main__':
    # Run the parking management
    parking_management = ParkingManagement()
    parking_management.run()
