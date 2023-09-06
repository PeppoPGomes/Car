import cv2
import numpy as np
import funcoes as fn
import math


def thresholding(img):
    imgHsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    lowerWhite = np.array([0,0,180])
    upperWhite = np.array([180,255,255])
    maskWhite = cv2.inRange(imgHsv,lowerWhite,upperWhite)
    return  maskWhite

def getLaneCurve(img):

    imgCopy = img.copy()
    imgThresh = thresholding(img)

    h, w, c = img.shape
    points = valTrackbars()
    imgWarp = warpImg(imgThresh,points,w,h)
    imgWarpPoints = drawPoints(imgCopy,points)

    basePoint,imgHist = getHistogram(imgWarp, display=True)


    #cv2.imshow('Thres',imgThresh)
    cv2.imshow('Warp',imgWarp)
    #cv2.imshow('WarpPoints',imgWarpPoints)
    #cv2.imshow('Histogram',imgHist)
    return None

def warpImg(img, points, w,h):
    pts1 = np.float32(points)
    pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgWarp=cv2.warpPerspective(img, matrix,(w,h))

    return imgWarp

def nothing(a):
    pass

def initializeTrackbars(intialTracbarVals, wt=480, ht=240):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360,240)
    cv2.createTrackbar("Width Top",  "Trackbars", intialTracbarVals[0], wt//2, nothing)
    cv2.createTrackbar("Height Top",  "Trackbars", intialTracbarVals[1], ht, nothing)
    cv2.createTrackbar("Width Bottom",  "Trackbars", intialTracbarVals[2], wt//2, nothing)
    cv2.createTrackbar("Height Bottom",  "Trackbars", intialTracbarVals[3], ht, nothing)


def valTrackbars(wt=480, ht=240):
    widthTop = cv2.getTrackbarPos("Width Top", "Trackbars")
    heightTop = cv2.getTrackbarPos("Height Top", "Trackbars")
    widthBottom = cv2.getTrackbarPos("Width Bottom", "Trackbars")
    heightBottom = cv2.getTrackbarPos("Height Bottom", "Trackbars")
    points = np.float32([(widthTop, heightTop), (wt-widthTop, heightTop), (widthBottom, heightBottom), (wt-widthBottom, heightBottom)])
    return points

def drawPoints(img, points):
    for x in range(4):
        cv2.circle(img,(int(points[x][0]),int(points[x][1])),15,(0,0,255), cv2.FILLED)
    
    return img


def getHistogram(img,minPer=0.1, display = False):
    histValues = np.sum(img,axis=0)
    #print(histValues)
    maxValue = np.max(histValues)
    minValue = minPer*maxValue

    indexArray = np.where(histValues >= minValue)
    basePoint = int(np.average(indexArray))
    #print(basePoint)
    if display:
        imgHist = np.zeros((img.shape[0], img.shape[1],3),np.uint8)
        for x,intensity in enumerate(histValues):
            cv2.line(imgHist,(x,img.shape[0]), (x, img.shape[0] - intensity//255),(255,0,255),1)
            cv2.circle(imgHist,(basePoint,img.shape[0]),20, (0,255,255),cv2.FILLED)

        return basePoint, imgHist        
    return basePoint

def find_car_position(img, min_area=1000):
    imgThresh = thresholding(img)
    
    # Encontre os contornos das faixas brancas
    contours, _ = cv2.findContours(imgThresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_contours = []

    for contour in contours:
        # Calcule os momentos apenas para contornos com área não zero e área maior que min_area
        if cv2.contourArea(contour) > min_area:
            moments = cv2.moments(contour)
            centroid_x = int(moments["m10"] / moments["m00"])
            centroid_y = int(moments["m01"] / moments["m00"])
            valid_contours.append((centroid_x, centroid_y))
    
    if len(valid_contours) >= 2:
        # Calcule o ponto médio entre os centroides válidos
        centroid_x_sum = sum(point[0] for point in valid_contours)
        centroid_y_sum = sum(point[1] for point in valid_contours)
        car_position = (int(centroid_x_sum / len(valid_contours)), int(centroid_y_sum / len(valid_contours)))

        # Calcule a distância vertical entre o ponto médio do carro e as faixas esquerda e direita
        distance_to_left_lane = car_position[1] - valid_contours[0][1]
        distance_to_right_lane = car_position[1] - valid_contours[1][1]
        cv2.circle(img, car_position, 5, (0, 255, 255), -1)  # Amarelo: (0, 255, 255)
        return car_position, distance_to_left_lane, distance_to_right_lane
    else:
        return None

def location_car_position(img, min_area=1000):
    imgThresh = thresholding(img)
    
    # Encontre os contornos das faixas brancas
    contours, _ = cv2.findContours(imgThresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_contours = []

    for contour in contours:
        # Calcule os momentos apenas para contornos com área não zero e área maior que min_area
        if cv2.contourArea(contour) > min_area:
            moments = cv2.moments(contour)
            centroid_x = int(moments["m10"] / moments["m00"])
            centroid_y = int(moments["m01"] / moments["m00"])
            valid_contours.append((centroid_x, centroid_y))
    
    if len(valid_contours) >= 2:
        # Calcule o ponto médio entre os centroides válidos
        centroid_x_sum = sum(point[0] for point in valid_contours)
        centroid_y_sum = sum(point[1] for point in valid_contours)
        car_position = (int(centroid_x_sum / len(valid_contours)), int(centroid_y_sum / len(valid_contours)))

        # Calcule a distância vertical entre o ponto médio do carro e as faixas esquerda e direita
        distance_to_left_lane = car_position[1] - valid_contours[0][1]
        distance_to_right_lane = car_position[1] - valid_contours[1][1]
        cv2.circle(img, car_position, 5, (0, 255, 255), -1)  # Amarelo: (0, 255, 255)
        return car_position
    else:
        return None

def draw_center_point(img):
    height, width, _ = img.shape
    center_x, center_y = width // 2, height // 2
    cv2.circle(img, (center_x, center_y), 5, (255, 0, 0), -1)  # Desenha um ponto azul no centro

def location_center_point(img):
    height, width, _ = img.shape
    center_x, center_y = width // 2, height // 2
    return center_x, center_y


def calculate_distance(circle1_center, circle2_center):

    if circle1_center is None or circle2_center is None:
        return None
    
    x1, y1 = circle1_center
    x2, y2 = circle2_center
    
    distance = x1 - x2
    return distance



if __name__ =='__main__':   

    cap = cv2.VideoCapture('teste1.mp4')
    initialTrackBarVals = [0,0,0,100]
    initializeTrackbars(initialTrackBarVals)
    

    while True:
        sucess, img = cap.read()
        img = cv2.resize(img,(480,240))
        
        car_position_info = find_car_position(img)
        
        centerPoint = draw_center_point(img)  # Chama a função para desenhar o ponto azul no centro
        
        circle1_center = location_center_point(img)

        circle2_center = location_car_position(img)

        distance = calculate_distance(circle1_center, circle2_center)       
        
        if car_position_info:
            car_position, distance_to_left_lane, distance_to_right_lane = car_position_info
            
            #print(f"Car Position: {car_position}")
            #print(f"Distance to Left Lane: {distance_to_left_lane}")
            #print(f"Distance to Right Lane: {distance_to_right_lane}")
            print(f"Distancia: {distance}")

        getLaneCurve(img)
        cv2.imshow('Vid',img)
        cv2.waitKey(1)
