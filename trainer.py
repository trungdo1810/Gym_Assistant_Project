import math
import cv2

def findAngle(image, kpts, p1, p2, p3, draw = True):
    coord = []
    no_kpt = len(kpts)//3
    for i in range (no_kpt):
        cx, cy = kpts[3*i], kpts[3*i+1]
        conf = kpts[3*i+2]
        coord.append([i, cx, cy, conf])
        
    point = (p1, p2, p3)
    
    # Get landmarks
    x1, y1 = coord[p1][1:3]
    x2, y2 = coord[p2][1:3]
    x3, y3 = coord[p3][1:3]
    
    # Calculate the Angle
    angle = math.degrees(math.atan2(y3-y2, x3-x2) - math.atan2(y1-y2, x1-x2))
    
    if angle < 0:
        angle += 360
        
    # Draw coordinates
    if draw:
        cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (255,255,255), 3)
        cv2.line(image, (int(x3), int(y3)), (int(x2), int(y2)), (255,255,255), 3)
        
        cv2.circle(image, (int(x1), int(y1)), 10, (255,255,255), cv2.FILLED)
        # cv2.circle(image, (int(x2), int(y2)), 20, (235,235,235), 5)
        cv2.circle(image, (int(x2), int(y2)), 10, (255,255,255), cv2.FILLED)
        # cv2.circle(image, (int(x1), int(y1)), 20, (255,255,255), 5)
        # cv2.circle(image, (int(x3), int(y3)), 10, (255,255,255), cv2.FILLED)
        # cv2.circle(image, (int(x1), int(y1)), 20, (255,255,255), cv2.FILLED)
    return int(angle)

def show_keypoints(image, kpts):
    coord = []
    no_kpt = len(kpts)//3
    for i in range (no_kpt):
        cx, cy = kpts[3*i], kpts[3*i+1]
        conf = kpts[3*i+2]
        coord.append([i, cx, cy, conf])
        
    point = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    # Get landmarks
    
    for i in range(17):
        x, y = coord[point[i]][1:3]
        cv2.circle(image, (int(x), int(y)), 10, (255,255,255), cv2.FILLED) #point
        cv2.putText(image, str(point[i]), (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 1,(255,0,0), 2)

    
    return 0

def point_distance(image, kpts, p1, p2, draw = True):
    coord = []
    no_kpt = len(kpts)//3
    for i in range (no_kpt):
        cx, cy = kpts[3*i], kpts[3*i+1]
        conf = kpts[3*i+2]
        coord.append([i, cx, cy, conf])
        
   
    
    # Get landmarks
    x1, y1 = coord[p1][1:3]
    x2, y2 = coord[p2][1:3]
    
    dist12 = math.sqrt((x1-x2)**2+(y1-y2)**2)
  
    
    # Draw coordinates
    if draw:
        if dist12 > 135.0:
            cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (255,255,255), 3)        
        else:
            cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 3)        

        cv2.circle(image, (int(x1), int(y1)), 10, (255,255,255), cv2.FILLED)
        cv2.circle(image, (int(x1), int(y1)), 20, (235,235,235), 5)
        cv2.circle(image, (int(x2), int(y2)), 10, (255,255,255), cv2.FILLED)
        cv2.circle(image, (int(x2), int(y2)), 20, (255,255,255), 5)
       
        
    return round(dist12,2)