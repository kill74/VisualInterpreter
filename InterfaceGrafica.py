import cv2
import numpy as np

def desenhar_botao(imagem, texto, pos, cor_fundo=(0, 0, 0), cor_texto=(255, 255, 255)):
    x, y, w, h = pos
    cv2.rectangle(imagem, (x, y), (x + w, y + h), cor_fundo, -1)
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    tamanho_texto = cv2.getTextSize(texto, fonte, 0.7, 2)[0]
    text_x = x + (w - tamanho_texto[0]) // 2
    text_y = y + (h + tamanho_texto[1]) // 2
    cv2.putText(imagem, texto, (text_x, text_y), fonte, 0.7, cor_texto, 2)

# Dimensões da janela e botões
largura, altura = 1250, 900
altura_camara = 800  # Altura reservada para o frame da câmera
botao_camara = (0, 800, 200, 70)
botao_blur = (250, 800, 200, 70)
botao_EfeitoSepia = (500, 800, 200, 70)
botao_cinza = (750, 800, 200, 70)
botao_fechar = (900, 800, 450, 70)

# Variáveis de estado
cap = None
cinza_ativo = False
desfoque_ativo = False
sepia_ativo = False

# Matriz do filtro sépia
sepia = np.array([[0.272, 0.534, 0.131],
                  [0.349, 0.686, 0.168],
                  [0.393, 0.769, 0.189]])

# Identificar Cara
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
#Identificar olhos
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")




def clique_dentro_botao(x, y, botao):
    bx, by, bw, bh = botao
    return bx <= x < bx + bw and by <= y < by + bh

def eventos_rato(event, x, y, flags, param):
    global cap, cinza_ativo, desfoque_ativo, sepia_ativo
    if event == cv2.EVENT_LBUTTONDOWN:
        if clique_dentro_botao(x, y, botao_camara):
            if cap is None:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    print("Erro ao abrir a câmera.")
                    cap = None
        elif clique_dentro_botao(x, y, botao_cinza):
            cinza_ativo = not cinza_ativo
        elif clique_dentro_botao(x, y, botao_blur):
            desfoque_ativo = not desfoque_ativo
        elif clique_dentro_botao(x, y, botao_EfeitoSepia):
            sepia_ativo = not sepia_ativo
        elif clique_dentro_botao(x, y, botao_fechar):
            if cap is not None:
                cap.release()
            cv2.destroyAllWindows()
            exit()

# Configurar janela e callback do rato
cv2.namedWindow('Janela')
cv2.setMouseCallback('Janela', eventos_rato)

# Loop para o programa funcionar e para os filtros funcionarem também
while True:
    # Criar a janela principal (área de câmera + área de botões)
    janela = np.zeros((altura, largura, 3), dtype=np.uint8)

    # Capturar e processar frame da câmera
    if cap is not None:
        ret, frame = cap.read()
        if ret:
            # Redimensionar o frame para caber na área da câmera
            frame = cv2.resize(frame, (largura, altura_camara))
            if cinza_ativo:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            if desfoque_ativo:
                frame = cv2.GaussianBlur(frame, (15, 15), 0) #ira meter a camara com desfoque 
            if sepia_ativo:
                frame = cv2.transform(frame, sepia)
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            # Inserir o frame processado na área de câmera da janela
            janela[0:altura_camara, 0:largura] = frame

    # Desenhar botões na janela
    desenhar_botao(janela, "Iniciar", botao_camara)
    desenhar_botao(janela, "Blur", botao_blur)
    desenhar_botao(janela, "Efeito Sepia", botao_EfeitoSepia)
    desenhar_botao(janela, "Cinza", botao_cinza)
    desenhar_botao(janela, "Fechar", botao_fechar)

    # Exibir a janela principal
    cv2.imshow('Janela', janela)

    # Verificar tecla de saída
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Pressionar ESC para sair
        break

if cap is not None:
    cap.release()
cv2.destroyAllWindows()
