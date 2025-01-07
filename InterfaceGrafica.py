import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                           QHBoxLayout, QPushButton, QLabel, QSlider, QFileDialog)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap

class ProcessadorImagem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Processador de Imagem OpenCV")
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
            }
            QLabel {
                color: white;
                font-size: 12px;
            }
            QPushButton {
                background-color: #0d6efd;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #0b5ed7;
            }
            QSlider {
                background-color: transparent;
            }
            QSlider::groove:horizontal {
                height: 4px;
                background: #4a4a4a;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #0d6efd;
                width: 16px;
                height: 16px;
                margin: -6px 0;
                border-radius: 8px;
            }
        """)

        # Inicialização das variáveis
        self.capture = cv2.VideoCapture(0)
        self.current_frame = None
        self.processed_frame = None
        self.is_processing = True
        self.current_filter = "original"
        self.detect_objects = False  # Nova variável para controle de detecção

        # Configuração dos parâmetros de processamento
        self.blur_value = 1
        self.brightness = 0
        self.contrast = 1
        self.canny_low = 100
        self.canny_high = 200

        # Parâmetros para detecção de objetos
        self.min_area = 5000  # Área mínima para considerar um objeto
        self.max_area = 50000  # Área máxima para considerar um objeto

        # Configuração da interface
        self.setup_ui()

        # Timer para atualização do frame
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 30ms = ~33 FPS

    def setup_ui(self):
        """Configura a interface gráfica principal"""
        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)

        # Área de exibição do vídeo
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("border: 2px solid #4a4a4a;")
        layout.addWidget(self.video_label)

        # Painel de controles
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_widget.setFixedWidth(200)

        # Botões de filtros
        self.add_filter_button("Original", "original", controls_layout)
        self.add_filter_button("Escala de Cinza", "gray", controls_layout)
        self.add_filter_button("Detecção de Bordas", "canny", controls_layout)
        self.add_filter_button("Desfoque", "blur", controls_layout)
        self.add_filter_button("Sépia", "sepia", controls_layout)

        # Botão para detecção de objetos
        self.detect_button = QPushButton("Detectar Objetos")
        self.detect_button.setCheckable(True)  # Faz o botão alternável
        self.detect_button.clicked.connect(self.toggle_detection)
        controls_layout.addWidget(self.detect_button)

        # Sliders de controle
        self.add_slider("Desfoque", 1, 21, self.blur_value, controls_layout)
        self.add_slider("Brilho", -100, 100, self.brightness, controls_layout)
        self.add_slider("Contraste", 0, 200, self.contrast, controls_layout)

        # Botões de ação
        self.add_action_button("Capturar Imagem", self.capture_image, controls_layout)
        self.add_action_button("Salvar Imagem", self.save_image, controls_layout)
        self.add_action_button("Contar Objetos", self.count_objects, controls_layout)

        layout.addWidget(controls_widget)

    def toggle_detection(self):
        """Alterna a detecção de objetos"""
        self.detect_objects = self.detect_button.isChecked()

    def detect_common_objects(self, frame):
        """Detecta objetos comuns como canetas e borrachas"""
        # Converte para HSV para melhor segmentação
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Cria uma máscara para objetos escuros (possíveis canetas)
        lower_dark = np.array([0, 0, 0])
        upper_dark = np.array([180, 255, 50])
        mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)

        # Cria uma máscara para objetos claros (possíveis borrachas)
        lower_light = np.array([0, 0, 200])
        upper_light = np.array([180, 30, 255])
        mask_light = cv2.inRange(hsv, lower_light, upper_light)

        # Combina as máscaras
        mask = cv2.bitwise_or(mask_dark, mask_light)

        # Aplica operações morfológicas para remover ruído
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Encontra contornos
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        result = frame.copy()

        # Analisa cada contorno
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_area < area < self.max_area:
                # Calcula características do objeto
                perimeter = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)

                # Calcula a razão de aspecto
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w)/h

                # Identifica o objeto baseado em suas características
                if len(approx) > 6 and aspect_ratio > 3:
                    object_type = "Caneta"
                    color = (0, 255, 0)  # Verde para canetas
                else:
                    object_type = "Borracha"
                    color = (0, 0, 255)  # Vermelho para borrachas

                # Desenha o retângulo e o nome do objeto
                cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
                cv2.putText(result, object_type, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return result

    def process_frame(self):
        """Processa o frame atual com o filtro selecionado"""
        if self.current_frame is None:
            return

        self.processed_frame = self.current_frame.copy()

        # Primeiro aplica os filtros normais
        if self.current_filter == "gray":
            self.processed_frame = cv2.cvtColor(self.processed_frame, cv2.COLOR_BGR2GRAY)
            self.processed_frame = cv2.cvtColor(self.processed_frame, cv2.COLOR_GRAY2BGR)
        elif self.current_filter == "canny":
            gray = cv2.cvtColor(self.processed_frame, cv2.COLOR_BGR2GRAY)
            self.processed_frame = cv2.Canny(gray, self.canny_low, self.canny_high)
            self.processed_frame = cv2.cvtColor(self.processed_frame, cv2.COLOR_GRAY2BGR)
        elif self.current_filter == "blur":
            k_size = self.blur_value * 2 + 1
            self.processed_frame = cv2.GaussianBlur(self.processed_frame, (k_size, k_size), 0)
        elif self.current_filter == "sepia":
            self.processed_frame = self.apply_sepia()

        # Depois aplica a detecção de objetos se estiver ativada
        if self.detect_objects:
            self.processed_frame = self.detect_common_objects(self.processed_frame)

    # [Resto do código permanece o mesmo...]

    def closeEvent(self, event):
        """Método chamado quando a janela é fechada"""
        self.capture.release()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ProcessadorImagem()
    window.show()
    sys.exit(app.exec())
