import numpy as np


class PointOrderer:
    '''
    This class is used to order the points of a polygon by projecting them onto a reference line.
    The coordinates of the points stay the same, but the order of the points is changed.
    The order is crucial for the correct computation of the skeleton. (at least for some methods)
    '''
    def __init__(self, reference_points):
        self.reference_points = reference_points
        self.extended_reference_points = []
        self.extend_reference_line()

    def extend_line_from_point(self, point1, point2, x_min=0, y_min=0, x_max=4032, y_max=3040):
        """Extend a line formed by point1 and point2 from point1 in the opposite direction until it reaches a boundary."""
        m = (point2[1] - point1[1]) / (point2[0] - point1[0]) if point2[0] != point1[0] else float('inf')
        c = point1[1] - m * point1[0]
        
        if m == float('inf'):  # vertical line
            return (point1[0], y_max if point1[1] < point2[1] else y_min)
        
        y_at_x_min = m * x_min + c
        y_at_x_max = m * x_max + c
        x_at_y_min = (y_min - c) / m
        x_at_y_max = (y_max - c) / m

        if y_at_x_min >= y_min and y_at_x_min <= y_max and point1[0] > point2[0]:
            return (x_min, y_at_x_min)
        elif y_at_x_max >= y_min and y_at_x_max <= y_max and point1[0] < point2[0]:
            return (x_max, y_at_x_max)
        elif x_at_y_min >= x_min and x_at_y_min <= x_max and point1[1] > point2[1]:
            return (x_at_y_min, y_min)
        else:
            return (x_at_y_max, y_max)
        
    def extend_reference_line(self):
        start_boundary_point = self.extend_line_from_point(self.reference_points[1], self.reference_points[0])
        end_boundary_point = self.extend_line_from_point(self.reference_points[-2], self.reference_points[-1])
        self.extended_reference_points = [start_boundary_point] + self.reference_points + [end_boundary_point]

    def order_points(self, unordered_points):
        """Order the unordered points by projecting them onto the extended reference line."""
        def custom_sort(point):
            min_distance = float('inf')
            order_value = None
            for i in range(len(self.extended_reference_points) - 1):
                segment_start, segment_end = self.extended_reference_points[i], self.extended_reference_points[i+1]
                segment_vector = np.array(segment_end) - np.array(segment_start)
                point_vector = np.array(point) - np.array(segment_start)
                t = np.dot(point_vector, segment_vector) / np.dot(segment_vector, segment_vector)
                projection = np.array(segment_start) + t * segment_vector
                distance = np.linalg.norm(np.array(point) - projection)
                if distance < min_distance:
                    min_distance = distance
                    order_value = i + t  # Updated this to just t for better ordering
            return order_value

        ordered_points = sorted(unordered_points, key=custom_sort)
        return ordered_points
