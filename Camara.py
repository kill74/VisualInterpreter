#Isto e um teste

import numpy as np
import cv2

#isto ira buscar o video da primeira webcam do computador
cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

#o loop ira comecar se a camara comecou a capturar video
while(True):
    #isto ira ler os frames da camara
    #ira dar return para cada frame
    ret, frame = cap.read()

    #ira ler as cores como BGR
    #cada frame ira ser convertido para hsv
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #ira dar output do frame
    out.write(hsv)


    #o output original ira dar display na janela
    cv2.imshow('Frame Original', frame)

    #ira dar output com o hsv
    cv2.imshow('Frame Processed', hsv)

    #iremos defenir uma tecla para parar o programa
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#fecha a janela
cap.release()

#iremos dar release ao output
out.release()

#Remove toda a memoria que tem alocada no programa
cv2.destroyAllWindows()
