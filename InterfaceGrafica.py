"""
feito com ajuda (https://docs.opencv.org/3.4/d8/dfe/classcv_1_1VideoCapture.html)
                (https://docs.opencv.org/3.4/examples.html)
                (https://docs.opencv.org/3.4/annotated.html)
"""
import cv2
import numpy as np

def desenhar_botao (imagem, texto, pos, cor_fundo=(50,50,50),cor_texto=(255,255,255)):
    """
    Desenhar um botao com texto na interface
    :param imagem:  Janela onde o botao sera desenhado
    :param texto: Texto exibido no botao
    :param pos: posicao e tamanho do botao (x,y: largura,altura)
    :param cor_fundo: Cor de fundo do botao
    :param cor_texto: cor do texto do botao
    """
    x, y, w, h = pos
    #ira desenhar o retangulo do botao
    cv2.rectangle(imagem, (x, y), (x + w, y + h), cor_fundo, -1)
    #centralizar o texto do butao no meio
    #defenir fonte do texto
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    #defenir o tamanho do texto
    tamanho_texto = cv2.getTextSize(texto, fonte, 0.7,2)[0]
    text_x = x + (w - tamanho_texto[0]) // 2
    text_y = y + (h - tamanho_texto[1]) // 2
    cv2.putText(imagem, texto, (text_x, text_y), fonte, 0.7, cor_texto,2)

    #verificar se o click foir dentro do botao
    def clique_dentro_botao(x, y, botao):
        """
        Verificar se o clique foi dentro do botao
        :param x: coordenada x do clique
        :param y: coordenada y do clique
        :param botao: Posicao e tamanho do botao
        :return: True se o clique foirem dentro do botao
        """
        bx, by, bw, bh = botao
        return bx <= x < bx + bw and by <= y < by + bh

    #Dimensoes da janela e botoes
    largura, altura = 800, 600
    botao_camara = (50, 500, 200, 50) #botao "iniciar a camara"
    botao_cinza = (300, 500, 200, 50) #botao "converter para cinzento"
    botao_fechar = (550, 500, 200, 50) # botao "Fechar"]

    #Inicia a janela inicial
    janela = np.zeros((altura, largura,3), dtype=np.uint8)

    #Desenhar botoes na janela inicial
    desenhar_botao(janela, "Iniciar Câmara", botao_camara)
    desenhar_botao(janela, "Converter para Cinzento", botao_cinza)
    desenhar_botao(janela, "Fechar", botao_fechar)

    cap = None #Variavel para o video
    cinza_ativo = False #estado do filtro de cinza

    #Funcao para os movimentos do rato
    def enventos_rato (event, x, y, flags, param):
        global  cap, cinza_ativo
        if event == cv2.EVENT_LBUTTONDOWN: #se carregar no botao esquerdo do rato
            if clique_dentro_botao(x, y, botao_camara):
                if cap is None: #iniciar a camara
                    cap = cv2.VideoCapture(0)
                    if not cap.isOpened():   #da return a true ou false se o video esta a ser capturado
                        print ("DEU ERRO KKKKKKKKKKKKK.")
                        cap = None
            elif clique_dentro_botao(x, y, botao_fechar):
                #ativar ou desativar o filtro da camara
                cinza_ativo = not cinza_ativo
            elif clique_dentro_botao(x, y, botao_fechar):
                if cap is not None: #se ainda estiver a capturar
                    cap.release() #ira para a captura
                cv2.destroyAllWindows()#e ira destruir as janelas
                exit() # e sair da aplicacao