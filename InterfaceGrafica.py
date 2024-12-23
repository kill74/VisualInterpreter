import tkinter as tk
from tkinter import ttk, filedialog
import cv2
from PIL import Image, ImageTk
import numpy as np
import os

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
deteccao_ativa = False

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

# Função para detectar faces e olhos
def detectar_olhos_e_cara(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    return frame

# Função para capturar o vídeo e processar as imagens
def capturar_video():
    global cap, frame_com_efeitos
    if cap is None:
        cap = cv2.VideoCapture(0)
    
    ret, frame = cap.read()
    if ret:
        # Aplicar os efeitos conforme ativados
        frame_com_efeitos = frame.copy()
        if cinza_ativo:
            frame_com_efeitos = aplicar_cinza(frame_com_efeitos)
        if desfoque_ativo:
            frame_com_efeitos = aplicar_desfoque(frame_com_efeitos)
        if sepia_ativo:
            frame_com_efeitos = aplicar_sepia(frame_com_efeitos)
        if segmentacao_ativa:
            frame_com_efeitos, num_objetos = segmentar_e_contar(frame_com_efeitos)
            cv2.putText(frame_com_efeitos, f"Objetos Detectados: {num_objetos}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        if deteccao_ativa:
            frame_com_efeitos = detectar_olhos_e_cara(frame_com_efeitos)
        
        # Converter a imagem BGR para RGB para exibir no Tkinter
        frame_rgb = cv2.cvtColor(frame_com_efeitos, cv2.COLOR_BGR2RGB)
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

def ativar_deteccao():
    global deteccao_ativa
    deteccao_ativa = not deteccao_ativa

# Função para salvar a imagem
def salvar_imagem():
    global frame_com_efeitos
    if frame_com_efeitos is not None:
        desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop') if os.name == 'nt' else os.path.join(os.path.expanduser("~"), 'Desktop')
        caminho = os.path.join(desktop, "imagem_salva.png")
        cv2.imwrite(caminho, frame_com_efeitos)
        print(f"Imagem salva com sucesso em: {caminho}")

# Função para iniciar a câmera e mostrar os botões de efeitos
def iniciar_camera():
    # Esconder o botão de "Iniciar Câmera" e mostrar os botões de efeitos
    btn_iniciar_camera.pack_forget()
    frame_botoes.pack(pady=10)
    
    # Iniciar a captura de vídeo
    capturar_video()

# Criando a janela principal
root = tk.Tk()
root.title("Detecção de Face e Efeitos")

# Layout da janela
root.geometry("1000x700")
root.configure(bg="#333333")

# Adicionando o painel para exibição do vídeo
painel_video = tk.Label(root)
painel_video.pack(pady=20)

# Adicionando o botão "Iniciar Câmera"
btn_iniciar_camera = ttk.Button(root, text="Iniciar Câmera", command=iniciar_camera)
btn_iniciar_camera.pack(pady=10)

# Frame para organizar os botões de efeitos
frame_botoes = tk.Frame(root, bg="#333333")
frame_botoes.pack(pady=10)

# Botões de efeitos
btn_cinza = ttk.Button(frame_botoes, text="Preto e Branco", command=ativar_cinza)
btn_sepia = ttk.Button(frame_botoes, text="Efeito Sepia", command=ativar_sepia)
btn_desfoque = ttk.Button(frame_botoes, text="Desfoque", command=ativar_desfoque)
btn_segmentacao = ttk.Button(frame_botoes, text="Segmentação", command=ativar_segmentacao)
btn_deteccao = ttk.Button(frame_botoes, text="Identificar Olhos e Cara", command=ativar_deteccao)
btn_salvar = ttk.Button(frame_botoes, text="Salvar Imagem", command=salvar_imagem)

# Adicionar os botões ao frame
btn_cinza.grid(row=0, column=0, padx=5, pady=5)
btn_sepia.grid(row=0, column=1, padx=5, pady=5)
btn_desfoque.grid(row=0, column=2, padx=5, pady=5)
btn_segmentacao.grid(row=0, column=3, padx=5, pady=5)
btn_deteccao.grid(row=0, column=4, padx=5, pady=5)
btn_salvar.grid(row=0, column=5, padx=5, pady=5)

# Variáveis globais
cap = None
frame_com_efeitos = None

# Iniciar o loop da interface gráfica
root.mainloop()
