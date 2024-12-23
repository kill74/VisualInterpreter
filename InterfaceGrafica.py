import cv2
import numpy as np

def desenhar_botao(imagem, texto, pos, cor_fundo=(0, 0, 0), cor_texto=(255, 255, 255)): 
    """
    Função para desenhar um botão na imagem.
    
    :param imagem: Imagem onde o botão será desenhado.
    :param texto: Texto a ser exibido no botão.
    :param pos: Posição e dimensões do botão (x, y, largura, altura).
    :param cor_fundo: Cor de fundo do botão.
    :param cor_texto: Cor do texto do botão.
    """
    x, y, w, h = pos
    # Desenha o retângulo que representa o botão
    cv2.rectangle(imagem, (x, y), (x + w, y + h), cor_fundo, -1)

    # Configura a fonte para o texto do botão
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    
    # Calcula a posição do texto para centralizá-lo no botão
    tamanho_texto = cv2.getTextSize(texto, fonte, 0.7, 2)[0]
    text_x = x + (w - tamanho_texto[0]) // 2
    text_y = y + (h + tamanho_texto[1]) // 2
    
    # Desenha o texto no botão
    cv2.putText(imagem, texto, (text_x, text_y), fonte, 0.7, cor_texto, 2)

# Dimensões da janela principal e área reservada para os botões
largura, altura = 1500, 900  # Aumentar a largura para caber todos os botões
altura_camara = 800  # A altura da área reservada para o vídeo da câmara

# Definição da posição e tamanho dos botões (diminuindo a largura para caber todos)
botao_camara = (0, 800, 180, 70)
botao_blur = (200, 800, 180, 70)
botao_EfeitoSepia = (400, 800, 180, 70)
botao_cinza = (600, 800, 180, 70)
botao_detetar = (800, 800, 180, 70)
botao_segmentar = (1000, 800, 180, 70)  # Novo botão para segmentação de objetos
botao_fechar = (1180, 800, 70, 70)  # Botão para fechar a aplicação

# Variáveis de estado para controlar quais efeitos estão ativados
cap = None
cinza_ativo = False
desfoque_ativo = False
sepia_ativo = False
detetar_ativo = False
segmentar_ativo = False  # Variável para controlar a segmentação de objetos

# Matriz do filtro sépia para alterar as cores
sepia = np.array([[0.272, 0.534, 0.131],
                  [0.349, 0.686, 0.168],
                  [0.393, 0.769, 0.189]])

# Carregamento dos classificadores para deteção de caras e olhos
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def clique_dentro_botao(x, y, botao):
    """
    Verifica se o clique do rato está dentro dos limites de um botão.
    
    :param x: Coordenada x do clique.
    :param y: Coordenada y do clique.
    :param botao: Dimensões do botão (x, y, largura, altura).
    :return: Verdadeiro se o clique estiver dentro do botão, Falso caso contrário.
    """
    bx, by, bw, bh = botao
    return bx <= x < bx + bw and by <= y < by + bh

def detetar_caras_olhos(frame):
    """
    Função para detetar caras e olhos no frame da câmara.
    
    :param frame: Imagem da câmara.
    :return: Imagem com as caras e olhos marcados.
    """
    # Converte a imagem para escala de cinza para melhorar a deteção
    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Deteta caras na imagem
    caras = face_cascade.detectMultiScale(cinza, 1.3, 5)
    
    # Desenha retângulos nas caras detetadas
    for (x, y, w, h) in caras:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Na região da cara, deteta os olhos
        cara_cinza = cinza[y:y+h, x:x+w]
        cara_colorida = frame[y:y+h, x:x+w]
        
        # Deteta olhos na região da cara
        olhos = eye_cascade.detectMultiScale(cara_cinza)
        
        # Desenha retângulos nos olhos detetados
        for (ex, ey, ew, eh) in olhos:
            cv2.rectangle(cara_colorida, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    
    return frame

def segmentar_e_contar(frame):
    """
    Função para segmentar e contar objetos na imagem utilizando técnicas de pré-processamento.
    
    :param frame: Imagem da câmara.
    :return: Imagem com os contornos dos objetos desenhados e a contagem exibida.
    """
    # Converte a imagem para escala de cinza
    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Aplica uma suavização para reduzir o ruído na imagem
    suavizado = cv2.GaussianBlur(cinza, (5, 5), 0)
    
    # Binariza a imagem com um limiar fixo
    _, binarizado = cv2.threshold(suavizado, 100, 255, cv2.THRESH_BINARY)
    
    # Operações morfológicas para melhorar a segmentação
    kernel = np.ones((5, 5), np.uint8)
    binarizado = cv2.dilate(binarizado, kernel, iterations=2)
    binarizado = cv2.erode(binarizado, kernel, iterations=1)
    
    # Deteta os contornos dos objetos
    contornos, _ = cv2.findContours(binarizado, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Desenha os contornos e conta os objetos
    objetos_contados = 0
    for contorno in contornos:
        if cv2.contourArea(contorno) > 500:  # Filtro para ignorar contornos pequenos
            objetos_contados += 1
            cv2.drawContours(frame, [contorno], -1, (0, 255, 0), 3)
    
    # Exibe a contagem de objetos na imagem
    cv2.putText(frame, f"Objetos: {objetos_contados}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return frame

def eventos_rato(event, x, y, flags, param):
    """
    Função que trata os eventos do rato.
    
    :param event: Tipo de evento do rato (clicar, mover, etc.).
    :param x: Coordenada x do clique do rato.
    :param y: Coordenada y do clique do rato.
    """
    global cap, cinza_ativo, desfoque_ativo, sepia_ativo, detetar_ativo, segmentar_ativo
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Botão para iniciar a câmara
        if clique_dentro_botao(x, y, botao_camara):
            if cap is None:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    print("Erro ao abrir a câmara.")
                    cap = None
        
        # Botões para ativar os efeitos ou funcionalidades
        elif clique_dentro_botao(x, y, botao_cinza):
            cinza_ativo = not cinza_ativo
        elif clique_dentro_botao(x, y, botao_blur):
            desfoque_ativo = not desfoque_ativo
        elif clique_dentro_botao(x, y, botao_EfeitoSepia):
            sepia_ativo = not sepia_ativo
        elif clique_dentro_botao(x, y, botao_detetar):
            detetar_ativo = not detetar_ativo
        elif clique_dentro_botao(x, y, botao_segmentar):
            segmentar_ativo = not segmentar_ativo
        
        # Botão para fechar a aplicação
        elif clique_dentro_botao(x, y, botao_fechar):
            if cap is not None:
                cap.release()
                cv2.destroyAllWindows()
            exit()

# Configurar a janela e a callback do rato
cv2.namedWindow('Janela')
cv2.setMouseCallback('Janela', eventos_rato)

# Loop principal do programa
while True:
    # Criar a janela principal (com a área da câmara e a área dos botões)
    janela = np.zeros((altura, largura, 3), dtype=np.uint8)
    
    # Capturar e processar o frame da câmara
    if cap is not None:
        ret, frame = cap.read()
        if ret:
            # Redimensionar o frame da câmara para caber na área da câmara
            frame = cv2.resize(frame, (largura, altura_camara))
            
            # Aplicar os efeitos ativados
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
            
            # Segmentação e contagem de objetos
            if segmentar_ativo:
                frame = segmentar_e_contar(frame)
            
            # Inserir o frame processado na janela
            janela[0:altura_camara, 0:largura] = frame
    
    # Desenhar todos os botões na janela
    desenhar_botao(janela, "Iniciar", botao_camara)
    desenhar_botao(janela, "Blur", botao_blur)
    desenhar_botao(janela, "Efeito Sepia", botao_EfeitoSepia)
    desenhar_botao(janela, "Cinza", botao_cinza)
    desenhar_botao(janela, "Detetar", botao_detetar)
    desenhar_botao(janela, "Segmentar", botao_segmentar)  # Novo botão de segmentação
    desenhar_botao(janela, "X", botao_fechar)
    
    # Exibir a janela principal
    cv2.imshow('Janela', janela)
    
    # Verificar se o utilizador pressionou a tecla ESC para sair
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Pressionar ESC para sair
        break

# Libertar os recursos da câmara e fechar as janelas
if cap is not None:
    cap.release()
cv2.destroyAllWindows()
