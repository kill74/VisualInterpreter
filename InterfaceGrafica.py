import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import numpy as np

# Carregar os classificadores para deteção de faces e olhos
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Definir a matriz de filtro sepia
sepia_filter = np.array([[0.272, 0.534, 0.131],
                         [0.349, 0.686, 0.168],
                         [0.393, 0.769, 0.189]])

# Variáveis de estado para os efeitos
cinza_ativo = False
desfoque_ativo = False
sepia_ativo = False
segmentacao_ativa = False

# Função para aplicar o filtro de sepia
def aplicar_sepia(frame):
    return cv2.transform(frame, sepia_filter)

# Função para aplicar o efeito de desfoque
def aplicar_desfoque(frame):
    return cv2.GaussianBlur(frame, (15, 15), 0)

# Função para converter a imagem para preto e branco
def aplicar_cinza(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Função para segmentação e contagem de objetos
def segmentar_e_contar(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Aplicar operações morfológicas para melhorar a segmentação
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Encontrar os contornos dos objetos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Contar o número de objetos detectados
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filtrar pequenos objetos
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 3)
    
    return frame, len(contours)

# Função para capturar o vídeo e processar as imagens
def capturar_video():
    global cap
    if cap is None:
        cap = cv2.VideoCapture(0)
    
    ret, frame = cap.read()
    if ret:
        # Aplicar os efeitos conforme ativados
        if cinza_ativo:
            frame = aplicar_cinza(frame)
        if desfoque_ativo:
            frame = aplicar_desfoque(frame)
        if sepia_ativo:
            frame = aplicar_sepia(frame)
        if segmentacao_ativa:
            frame, num_objetos = segmentar_e_contar(frame)
            cv2.putText(frame, f"Objetos Detectados: {num_objetos}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Converter a imagem BGR para RGB para exibir no Tkinter
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = ImageTk.PhotoImage(img)

        # Atualizar o widget de imagem
        painel_video.imgtk = img
        painel_video.configure(image=img)
    
    # Atualizar a imagem a cada 10ms
    painel_video.after(10, capturar_video)

# Função para fechar a janela e liberar a câmera
def fechar_janela():
    global cap
    if cap is not None:
        cap.release()
    root.quit()

# Funções para ativar/desativar os efeitos
def ativar_cinza():
    global cinza_ativo
    cinza_ativo = not cinza_ativo

def ativar_desfoque():
    global desfoque_ativo
    desfoque_ativo = not desfoque_ativo

def ativar_sepia():
    global sepia_ativo
    sepia_ativo = not sepia_ativo

def ativar_segmentacao():
    global segmentacao_ativa
    segmentacao_ativa = not segmentacao_ativa

# Criando a janela principal
root = tk.Tk()
root.title("Detecção de Face e Efeitos")

# Estilo do tema
style = ttk.Style()
style.configure("TButton",
                padding=6,
                relief="flat",
                background="#3e8e41",
                foreground="white",
                font=("Arial", 12))

# Layout da janela
root.geometry("1000x700")
root.configure(bg="#333333")

# Adicionando o painel para exibição do vídeo
painel_video = tk.Label(root)
painel_video.pack(pady=20)

# Adicionando botões
btn_iniciar = ttk.Button(root, text="Iniciar Detecção", command=capturar_video)
btn_iniciar.pack(pady=10)

btn_cinza = ttk.Button(root, text="Preto e Branco", command=ativar_cinza)
btn_cinza.pack(pady=10)

btn_desfoque = ttk.Button(root, text="Desfoque", command=ativar_desfoque)
btn_desfoque.pack(pady=10)

btn_sepia = ttk.Button(root, text="Efeito Sepia", command=ativar_sepia)
btn_sepia.pack(pady=10)

btn_segmentacao = ttk.Button(root, text="Segmentação", command=ativar_segmentacao)
btn_segmentacao.pack(pady=10)

btn_fechar = ttk.Button(root, text="Fechar", command=fechar_janela)
btn_fechar.pack(pady=10)

# Variáveis globais
cap = None

# Iniciar o loop da interface gráfica
root.mainloop()
