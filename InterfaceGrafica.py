import cv2
import numpy as np

def desenhar_botao(imagem, texto, pos, cor_fundo=(0, 0, 0), cor_texto=(255, 255, 255)): 
    """
    Função para desenhar um botão na imagem
    
    :param imagem: Imagem onde o botão será desenhado
    :param texto: Texto do botão
    :param pos: Posição e dimensões do botão (x, y, largura, altura)
    :param cor_fundo: Cor de fundo do botão
    :param cor_texto: Cor do texto do botão
    """
    x, y, w, h = pos
    # Desenha o retângulo do botão
    cv2.rectangle(imagem, (x, y), (x + w, y + h), cor_fundo, -1)

    # Configura a fonte
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    
    # Calcula a posição central do texto
    tamanho_texto = cv2.getTextSize(texto, fonte, 0.7, 2)[0]
    text_x = x + (w - tamanho_texto[0]) // 2
    text_y = y + (h + tamanho_texto[1]) // 2
    
    # Desenha o texto no botão
    cv2.putText(imagem, texto, (text_x, text_y), fonte, 0.7, cor_texto, 2)

# Dimensões da janela e área dos botões
largura, altura = 1250, 900
altura_camara = 800  # Altura reservada para o frame da câmara

# Definição da posição dos botões
botao_camara = (0, 800, 200, 70)
botao_blur = (250, 800, 200, 70)
botao_EfeitoSepia = (500, 800, 200, 70)
botao_cinza = (750, 800, 200, 70)
botao_detetar = (1000, 800, 200, 70)
botao_fechar = (1200, 800, 50, 70)

# Variáveis de estado
cap = None
cinza_ativo = False
desfoque_ativo = False
sepia_ativo = False
detetar_ativo = False

# Matriz do filtro sépia
sepia = np.array([[0.272, 0.534, 0.131],
                  [0.349, 0.686, 0.168],
                  [0.393, 0.769, 0.189]])

# Carregamento dos classificadores para deteção de caras e olhos
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def clique_dentro_botao(x, y, botao):
    """
    Verifica se o clique do rato está dentro dos limites de um botão
    
    :param x: Coordenada x do clique
    :param y: Coordenada y do clique
    :param botao: Dimensões do botão
    :return: Verdadeiro se o clique estiver dentro do botão, Falso caso contrário
    """
    bx, by, bw, bh = botao
    return bx <= x < bx + bw and by <= y < by + bh

def detetar_caras_olhos(frame):
    """
    Deteta caras e olhos no frame
    
    :param frame: Imagem de entrada
    :return: Frame com caras e olhos marcados
    """
    # Converte para escala de cinzentos para deteção
    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Deteta caras
    caras = face_cascade.detectMultiScale(cinza, 1.3, 5)
    
    # Desenha retângulos nas caras detetadas
    for (x, y, w, h) in caras:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Na região da cara, deteta olhos
        cara_cinza = cinza[y:y+h, x:x+w]
        cara_colorida = frame[y:y+h, x:x+w]
        
        # Deteta olhos na região da cara
        olhos = eye_cascade.detectMultiScale(cara_cinza)
        
        # Desenha retângulos nos olhos detetados
        for (ex, ey, ew, eh) in olhos:
            cv2.rectangle(cara_colorida, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    
    return frame

def eventos_rato(event, x, y, flags, param):
    """
    Trata eventos do rato
    
    :param event: Tipo de evento do rato
    :param x: Coordenada x do rato
    :param y: Coordenada y do rato
    """
    global cap, cinza_ativo, desfoque_ativo, sepia_ativo, detetar_ativo
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Botão para iniciar câmara
        if clique_dentro_botao(x, y, botao_camara):
            if cap is None:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    print("Erro ao abrir a câmara.")
                    cap = None
        
        # Botões de filtros e deteção
        elif clique_dentro_botao(x, y, botao_cinza):
            cinza_ativo = not cinza_ativo
        elif clique_dentro_botao(x, y, botao_blur):
            desfoque_ativo = not desfoque_ativo
        elif clique_dentro_botao(x, y, botao_EfeitoSepia):
            sepia_ativo = not sepia_ativo
        elif clique_dentro_botao(x, y, botao_detetar):
            detetar_ativo = not detetar_ativo
        
        # Botão para fechar
        elif clique_dentro_botao(x, y, botao_fechar):
            if cap is not None:
                cap.release()
                cv2.destroyAllWindows()
            exit()

# Configurar janela e callback do rato
cv2.namedWindow('Janela')
cv2.setMouseCallback('Janela', eventos_rato)

# Loop principal do programa
while True:
    # Criar a janela principal (área de câmara + área de botões)
    janela = np.zeros((altura, largura, 3), dtype=np.uint8)
    
    # Capturar e processar frame da câmara
    if cap is not None:
        ret, frame = cap.read()
        if ret:
            # Redimensionar o frame para caber na área da câmara
            frame = cv2.resize(frame, (largura, altura_camara))
            
            # Aplicar filtros conforme selecionado
            if cinza_ativo:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
            if desfoque_ativo:
                frame = cv2.GaussianBlur(frame, (15, 15), 0)
            
            if sepia_ativo:
                frame = cv2.transform(frame, sepia)
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            
            # Deteção de caras e olhos
            if detetar_ativo:
                frame = detetar_caras_olhos(frame)
            
            # Inserir o frame processado na área de câmara da janela
            janela[0:altura_camara, 0:largura] = frame
    
    # Desenhar botões na janela
    desenhar_botao(janela, "Iniciar", botao_camara) #ira começar a câmara
    desenhar_botao(janela, "Blur", botao_blur) # ira meter a câmara
    desenhar_botao(janela, "Efeito Sepia", botao_EfeitoSepia) #ira meter o efeito sepia na câmara
    desenhar_botao(janela, "Cinza", botao_cinza) # ira meter a câmara com o efeito de preto e branco
    desenhar_botao(janela, "Detetar", botao_detetar) # ira detetar a cara e os olhos 
    desenhar_botao(janela, "X", botao_fechar) #poderemos carregar no esc on no X para fechar a janela 
    
    # Exibir a janela principal
    cv2.imshow('Janela', janela)
    
    # Verificar tecla de saída
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Pressionar ESC para sair
        break

# Libertar recursos
if cap is not None:
    cap.release()
cv2.destroyAllWindows()