import cv2
import numpy as np

def desenhar_botao(imagem, texto, pos, cor_fundo=(0,0,0), cor_texto=(255,255,255)):
   """
   :param imagem: Janela onde o botao sera desenhado
   :param texto: Texto exibido no botao
   :param pos: posicao e tamanho do botao (x,y: largura,altura)
   :param cor_fundo: Cor de fundo do botao
   :param cor_texto: cor do texto do botao
   """
   x, y, w, h = pos
   cv2.rectangle(imagem, (x, y), (x + w, y + h), cor_fundo, -1)
   fonte = cv2.FONT_HERSHEY_SIMPLEX
   tamanho_texto = cv2.getTextSize(texto, fonte, 0.7, 2)[0]
   text_x = x + (w - tamanho_texto[0]) // 2
   text_y = y + (h - tamanho_texto[1]) // 2
   cv2.putText(imagem, texto, (text_x, text_y), fonte, 0.7, cor_texto,)

# Dimensões da janela e botões
largura, altura = 1000, 800 #tamanho da janela em si
botao_camara = (0, 700, 200, 70)
botao_EfeitoSepia = (250, 700, 200, 70)
botao_cinza = (500, 700, 200, 70)
botao_fechar = (750, 700, 200, 70)

# Iniciar janela
janela = np.zeros((altura, largura, 3), dtype=np.uint8)

# Desenhar botões
desenhar_botao(janela, "Iniciar", botao_camara)
desenhar_botao(janela, "Efeito Sepia", botao_EfeitoSepia)
desenhar_botao(janela, "Converter para Cinza", botao_cinza)
desenhar_botao(janela, "Fechar", botao_fechar)

# Variáveis de estado
cap = None
cinza_ativo = False

def clique_dentro_botao(x, y, botao):
   """
   :param x: coordenada x do clique
   :param y: coordenada y do clique
   :param botao: Posicao e tamanho do botao
   :return: True se o clique foirem dentro do botao
   """
   bx, by, bw, bh = botao
   return bx <= x < bx + bw and by <= y < by + bh

def eventos_rato(event, x, y, flags, param):
   global cap, cinza_ativo
   if event == cv2.EVENT_LBUTTONDOWN:
       if clique_dentro_botao(x, y, botao_camara):
           if cap is None:
               cap = cv2.VideoCapture(0)
               if not cap.isOpened():
                   print("Erro ao abrir a camera.")
                   cap = None
       elif clique_dentro_botao(x, y, botao_cinza):
           cinza_ativo = not cinza_ativo
       elif clique_dentro_botao(x, y, botao_fechar):
           if cap is not None:
               cap.release()
           cv2.destroyAllWindows()
           exit()

# Configurar janela e callback do mouse
cv2.namedWindow('Janela')
cv2.setMouseCallback('Janela', eventos_rato)

# Loop principal
while True:
   cv2.imshow('Janela', janela)
   
   # Capturar e processar frame da câmera
   if cap is not None:
       ret, frame = cap.read()
       if ret:
           if cinza_ativo:
               frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
               frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
           cv2.imshow('Camera', frame)

   # Verificar tecla de saída
   key = cv2.waitKey(1) & 0xFF
   if key == 27:  # para sair do programa basta clicar na tecla ESC
       break # que ira dar break (parar o programa)

# Liberar recursos
if cap is not None:
   cap.release()
cv2.destroyAllWindows()