import cv2
import numpy as np
import funcoes as fn
import math
import serial
import time

#ESP - Carro
porta_serial = "COM6"  # Substitua "COMX" pela porta serial do seu ESP32
baud_rate = 115200

# Arduino Teste
#porta_serial = "COM4"  # Substitua "COMX" pela porta serial do seu ESP32
#baud_rate = 9600

def extract_roi(image):

 
    roi_coordinates = (0, 200, 640, 480)
    x1, y1, x2, y2 = roi_coordinates
    
    return image[y1:y2, x1:x2]
    

def thresholding(img):
    imgHsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    lowerWhite = np.array([0,0,230])
    upperWhite = np.array([230,255,255])
    maskWhite = cv2.inRange(imgHsv,lowerWhite,upperWhite)
    maskROI = extract_roi(maskWhite)
    return  maskROI


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






def location_car_position(img, min_area=1000):
    
    
    # Encontre os contornos das faixas brancas
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
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
        cv2.circle(img, car_position, 2, (255, 255, 255), -1) 
        return car_position
    else:
        return None
     
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


def draw_center_point(img):

    height, width = img.shape
    center_x, center_y = width // 2, height // 2
    cv2.circle(img, (center_x, center_y), 2, (255, 0, 0), -1)  # Desenha um ponto azul no centro

def location_center_point(img):
    
    height, width = img.shape
    center_x, center_y = width // 2, height // 2
    return center_x, center_y


def calculate_distance_x(circle1_center, circle2_center):

    if circle1_center is None or circle2_center is None:
        return None
    
    x1, y1 = circle1_center
    x2, y2 = circle2_center
    
    distance_x = x1 - x2
    

    return distance_x

def calculate_distance_y(circle1_center, circle2_center):

    if circle1_center is None or circle2_center is None:
        return None
    
    x1, y1 = circle1_center
    x2, y2 = circle2_center
    

    distance_y = y1 - y2

    return distance_y
    
last_command_time = 0  # Variável para controlar o tempo do último comando enviado
historicoComando = None
contadorComandos = 0
""""
# Função para enviar comandos para o ESP32
def enviar_ESP(distance_x, distance_y, ser):
    global last_command_time
    global historicoComando
    global contadorComandos
    current_time = time.time()
    delay_seconds = 1  # Defina o valor desejado para o atraso (delay).

    if distance_x is not None:
        if distance_x > 0:
            if time.time() - last_command_time >= delay_seconds:
                ser.write(b'e')  # Envia o comando 'e'
                print("Enviando 'e' para ESP32")
                last_command_time = current_time
                historicoComando = 'e'
                contadorComandos = contadorComandos - 1
        elif distance_x < 0:
            if time.time() - last_command_time >= delay_seconds:
                ser.write(b'd')  # Envia o comando 'd'
                print("Enviando 'd' para ESP32")
                last_command_time = current_time
                historicoComando = 'd'
                contadorComandos = contadorComandos + 1
    else:
        if historicoComando == 'e':
            if time.time() - last_command_time >= delay_seconds:
                ser.write(b'e') 
                print("Mantendo ultimo comando e")
                last_command_time = current_time
                historicoComando = 'e'
                contadorComandos = contadorComandos - 1

        if historicoComando == 'd':
            if time.time() - last_command_time >= delay_seconds:
                ser.write(b'd') 
                print("Mantendo ultimo comando d")
                last_command_time = current_time  
                historicoComando = 'd'  
                contadorComandos = contadorComandos + 1

"""
historicoComando = 'inicio'
historicoComando2= 'inicio'
historicoComando3= 'inicio'
last_command_time2=0
last_command_time3=0
def enviarComandoESP32(distance_x1, distance_x2,distance_x3,distance_x4, distance_x5, distance_x6, ser):
    current_time = time.time()
    current_time2 = time.time()
    current_time3 = time.time()
    global last_command_time
    global last_command_time2
    global last_command_time3
    delay_seconds = 0.6
    global historicoComando
    global historicoComando2
    global historicoComando3

    if distance_x1 == (0, 0) and distance_x2 == (0, 0) and historicoComando =='inicio':
        pass  # Não há interseção em ambas as linhas, nenhum comando é enviado
        
    elif distance_x1 == (0, 0) and distance_x2 == (0, 0) and historicoComando =='d' and historicoComando3 == 'inicio':
        if time.time() - last_command_time >= delay_seconds:
            ser.write(b'e')  
            print("Enviando 'e' para ESP32")
            last_command_time = current_time
            historicoComando = 'inicio'
    elif distance_x1 == (0, 0) and distance_x2 == (0, 0) and historicoComando =='e'and historicoComando3 == 'inicio':
            if time.time() - last_command_time >= delay_seconds:
                ser.write(b'd')  
                print("Enviando 'd' para ESP32")
                last_command_time = current_time
                historicoComando = 'inicio'
    elif distance_x1 != (0, 0) and distance_x2 == (0, 0) and historicoComando != 'd':
            if time.time() - last_command_time >= delay_seconds:
                ser.write(b'd')  
                print("Enviando 'd' para ESP32")
                last_command_time = current_time
                historicoComando = 'd'
    elif distance_x1 == (0, 0) and distance_x2 != (0, 0) and historicoComando != 'e':
            if time.time() - last_command_time >= delay_seconds:
                ser.write(b'e')  
                print("Enviando 'e' para ESP32")
                last_command_time = current_time
                historicoComando = 'e'
                #----------------------------
    if distance_x3 == (0, 0) and distance_x4 == (0, 0) and historicoComando2 =='inicio':
            pass  # Não há interseção em ambas as linhas, nenhum comando é enviado
            
    elif distance_x3 == (0, 0) and distance_x4 == (0, 0) and historicoComando2 =='r' and historicoComando3 == 'inicio':
        if time.time() - last_command_time2 >= delay_seconds:
            ser.write(b'l')  
            print("Enviando 'l' para ESP32")
            last_command_time2 = current_time2
            historicoComando2 = 'inicio'

    elif distance_x3 == (0, 0) and distance_x4 == (0, 0) and historicoComando2 =='l' and historicoComando3 == 'inicio':
        if time.time() - last_command_time2>= delay_seconds:
            ser.write(b'r')  
            print("Enviando 'r' para ESP32")
            last_command_time2 = current_time2
            historicoComando2 = 'inicio'
    elif distance_x3 != (0, 0) and distance_x4 == (0, 0)and historicoComando2 != 'r' and distance_x1 != (0, 0) and distance_x2 == (0, 0):
        if time.time() - last_command_time2 >= delay_seconds:
            ser.write(b'r')  
            print("Enviando 'r' para ESP32")
            last_command_time2 = current_time2
            historicoComando2 = 'r'
    elif distance_x3 == (0, 0) and distance_x4 != (0, 0) and historicoComando2 != 'l' and distance_x1 == (0, 0) and distance_x2 != (0, 0):
        if time.time() - last_command_time2 >= delay_seconds:
            ser.write(b'l')  
            print("Enviando 'l' para ESP32")
            last_command_time2 = current_time2
            historicoComando2 = 'l'        

            #-------------------
    if distance_x5 == (0, 0) and distance_x5 == (0, 0) and historicoComando3 =='inicio':
            pass  # Não há interseção em ambas as linhas, nenhum comando é enviado
            
    elif distance_x5 == (0, 0) and distance_x6 == (0, 0) and historicoComando3 =='p' and distance_x1 != (0,0):
        if time.time() - last_command_time3 >= delay_seconds:
            ser.write(b'f')  
            print("Enviando 'f' para ESP32")
            last_command_time3 = current_time3
            historicoComando3 = 'inicio'

    elif distance_x5 == (0, 0) and distance_x6 == (0, 0) and historicoComando3 =='f' and distance_x1 != (0,0):
        if time.time() - last_command_time2>= delay_seconds:
            ser.write(b'p')  
            print("Enviando 'p' para ESP32")
            last_command_time3 = current_time3
            historicoComando3 = 'inicio'
    elif distance_x5 != (0, 0) and distance_x6 == (0, 0) and historicoComando3 != 'p' and distance_x1 != (0, 0)and distance_x3 != (0, 0)  and distance_x2 == (0, 0) and distance_x4 == (0, 0):
        if time.time() - last_command_time3 >= delay_seconds:
            ser.write(b'p')  
            print("Enviando 'p' para ESP32")
            last_command_time3 = current_time3
            historicoComando3 = 'p'
    elif distance_x5 == (0, 0) and distance_x6 != (0, 0) and historicoComando3 != 'f' and distance_x1 == (0, 0) and distance_x3 == (0, 0) and distance_x2 != (0, 0) and distance_x4 != (0, 0):
        if time.time() - last_command_time3 >= delay_seconds:
            ser.write(b'f')  
            print("Enviando 'f' para ESP32")
            last_command_time3 = current_time3
            historicoComando3 = 'f'        


def linhasVerticais(thresh, img):
# Inicialize as posições das linhas
    linha1_x1, linha1_y1, linha1_x2, linha1_y2 = 185, 30, 185, 600
    linha2_x1, linha2_y1, linha2_x2, linha2_y2 = 460, 30, 460, 600
    linha3_x1, linha3_y1, linha3_x2, linha3_y2 = 185, 90, 185, 600
    linha4_x1, linha4_y1, linha4_x2, linha4_y2 = 460, 90, 460, 600
    linha5_x1, linha5_y1, linha5_x2, linha5_y2 = 185, 160, 185, 600
    linha6_x1, linha6_y1, linha6_x2, linha6_y2 = 460, 160, 460, 600

    # Inicialize as listas para armazenar as coordenadas de interseção para cada linha
    intersecoes_linha1 = []
    intersecoes_linha2 = []
    intersecoes_linha3 = []
    intersecoes_linha4 = []
    intersecoes_linha5 = []
    intersecoes_linha6 = []

    line = extract_roi(img)
    cv2.line(line, (linha1_x1, linha1_y1), (linha1_x2, linha1_y2), (0, 0, 0), 20)
    cv2.line(line, (linha2_x1, linha2_y1), (linha2_x2, linha2_y2), (0, 0, 0), 20)
    cv2.line(line, (linha3_x1, linha3_y1), (linha3_x2, linha3_y2), (255, 0, 0), 20)
    cv2.line(line, (linha4_x1, linha4_y1), (linha4_x2, linha4_y2), (255, 0, 0), 20)
    cv2.line(line, (linha5_x1, linha5_y1), (linha5_x2, linha5_y2), (0, 255, 0), 20)
    cv2.line(line, (linha6_x1, linha6_y1), (linha6_x2, linha6_y2), (0, 255, 0), 20)


    # Loop para verificar a primeira linha
    for y in range(min(linha1_y1, linha1_y2), max(linha1_y1, linha1_y2) + 1):
        x = linha1_x1  # A linha 1 é vertical, então o valor x é constante
        if 0 <= y < thresh.shape[0]:  # Verifique se y está dentro dos limites do array
            if thresh[y, x] == 255:  # Verifique se o pixel é branco
                intersecoes_linha1.append((x, y))

    # Loop para verificar a segunda linha
    for y in range(min(linha2_y1, linha2_y2), max(linha2_y1, linha2_y2) + 1):
        x = linha2_x1  # A linha 2 é vertical, então o valor x é constante
        if 0 <= y < thresh.shape[0]:  # Verifique se y está dentro dos limites do array
            if thresh[y, x] == 255:  # Verifique se o pixel é branco
                intersecoes_linha2.append((x, y))

    # Loop para verificar a terceira linha
    for y in range(min(linha3_y1, linha3_y2), max(linha3_y1, linha3_y2) + 1):
        x = linha3_x1  # A linha 3 é vertical, então o valor x é constante
        if 0 <= y < thresh.shape[0]:  # Verifique se y está dentro dos limites do array
            if thresh[y, x] == 255:  # Verifique se o pixel é branco
                intersecoes_linha3.append((x, y))

    # Loop para verificar a quarta linha
    for y in range(min(linha4_y1, linha4_y2), max(linha4_y1, linha4_y2) + 1):
        x = linha4_x1  # A linha 4 é vertical, então o valor x é constante
        if 0 <= y < thresh.shape[0]:  # Verifique se y está dentro dos limites do array
            if thresh[y, x] == 255:  # Verifique se o pixel é branco
                intersecoes_linha4.append((x, y))

    # Loop para verificar a quinta linha
    for y in range(min(linha5_y1, linha5_y2), max(linha5_y1, linha5_y2) + 1):
        x = linha5_x1  # A linha 5 é vertical, então o valor x é constante
        if 0 <= y < thresh.shape[0]:  # Verifique se y está dentro dos limites do array
            if thresh[y, x] == 255:  # Verifique se o pixel é branco
                intersecoes_linha5.append((x, y))

    # Loop para verificar a sexta linha
    for y in range(min(linha6_y1, linha6_y2), max(linha6_y1, linha6_y2) + 1):
        x = linha6_x1  # A linha 6 é vertical, então o valor x é constante
        if 0 <= y < thresh.shape[0]:  # Verifique se y está dentro dos limites do array
            if thresh[y, x] == 255:  # Verifique se o pixel é branco
                intersecoes_linha6.append((x, y))


    if len(intersecoes_linha1) > 0:
        # Se houver interseções na linha 1, calcule a média das coordenadas
        media_x_linha1 = sum([p[0] for p in intersecoes_linha1]) / len(intersecoes_linha1)
        media_y_linha1 = sum([p[1] for p in intersecoes_linha1]) / len(intersecoes_linha1)
    else:
        # Se não houver interseções na linha 1, retorne (0, 0) para essa linha
        media_x_linha1, media_y_linha1 = (0, 0)

    if len(intersecoes_linha2) > 0:
        # Se houver interseções na linha 2, calcule a média das coordenadas
        media_x_linha2 = sum([p[0] for p in intersecoes_linha2]) / len(intersecoes_linha2)
        media_y_linha2 = sum([p[1] for p in intersecoes_linha2]) / len(intersecoes_linha2)
    else:
        # Se não houver interseções na linha 2, retorne (0, 0) para essa linha
        media_x_linha2, media_y_linha2 = (0, 0)

    if len(intersecoes_linha3) > 0:
         # Se houver interseções na linha 3, calcule a média das coordenadas
         media_x_linha3 = sum([p[0] for p in intersecoes_linha3]) / len(intersecoes_linha3)
         media_y_linha3 = sum([p[1] for p in intersecoes_linha3]) / len(intersecoes_linha3)
    else:
        # Se não houver interseções na linha 3, retorne (0, 0) para essa linha
        media_x_linha3, media_y_linha3 = (0, 0)

    if len(intersecoes_linha4) > 0:
         # Se houver interseções na linha 4, calcule a média das coordenadas
         media_x_linha4 = sum([p[0] for p in intersecoes_linha4]) / len(intersecoes_linha4)
         media_y_linha4 = sum([p[1] for p in intersecoes_linha4]) / len(intersecoes_linha4)
    else:
        # Se não houver interseções na linha 4, retorne (0, 0) para essa linha
        media_x_linha4, media_y_linha4 = (0, 0)

    if len(intersecoes_linha5) > 0:
         # Se houver interseções na linha 5, calcule a média das coordenadas
         media_x_linha5 = sum([p[0] for p in intersecoes_linha5]) / len(intersecoes_linha5)
         media_y_linha5 = sum([p[1] for p in intersecoes_linha5]) / len(intersecoes_linha5)
    else:
        # Se não houver interseções na linha 5, retorne (0, 0) para essa linha
        media_x_linha5, media_y_linha5 = (0, 0)

    if len(intersecoes_linha6) > 0:
     # Se houver interseções na linha 5, calcule a média das coordenadas
         media_x_linha6 = sum([p[0] for p in intersecoes_linha6]) / len(intersecoes_linha6)
         media_y_linha6 = sum([p[1] for p in intersecoes_linha6]) / len(intersecoes_linha6)
    else:
        # Se não houver interseções na linha 6, retorne (0, 0) para essa linha
        media_x_linha6, media_y_linha6 = (0, 0)

    return (media_x_linha1, media_y_linha1), (media_x_linha2, media_y_linha2), (media_x_linha3, media_y_linha3), (media_x_linha4, media_y_linha4), (media_x_linha5, media_y_linha5), (media_x_linha6, media_y_linha6)

if __name__ =='__main__':   

    cap = cv2.VideoCapture(0)
    initialTrackBarVals = [0,0,0,100]
    initializeTrackbars(initialTrackBarVals)
    ser = serial.Serial(porta_serial, baud_rate, timeout=1)
    print("Porta serial aberta com sucesso.")
   
    while True:
        success, img = cap.read()
    
        thresh = thresholding(img)
    
        centerPoint = draw_center_point(thresh)
    
        circle1_center = location_center_point(thresh)
        circle2_center = location_car_position(thresh)
    
        distance_x = calculate_distance_x(circle1_center, circle2_center)
        distance_y = calculate_distance_y(circle1_center, circle2_center)

       # enviar_ESP(distance_x, distance_y, ser)
        

        Line = img
        distanciaL1, distanciaL2, distanciaL3, distanciaL4, distanciaL5, distanciaL6 = linhasVerticais(thresh,Line)
        
      


        enviarComandoESP32(distanciaL1,distanciaL2, distanciaL3, distanciaL4,distanciaL5, distanciaL6, ser)

        cv2.putText(img, str(distanciaL1), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, str(distanciaL2), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, str(distanciaL3), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, str(distanciaL4), (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, str(distanciaL5), (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, str(distanciaL6), (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Vid', img)
        cv2.imshow('ROI',thresh)
        cv2.imshow('Lines', Line)
        cv2.waitKey(1)
