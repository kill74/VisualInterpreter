import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QSlider,
    QFileDialog,
    QCheckBox,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
import os


class ProcessadorImagem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trabalho AI - Processador de Imagens")
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #2b2b2b;
            }
            QLabel {
                color: white;
                font-size: 12px;
            }
            QPushButton {
                background-color: #800000;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #0b5ed7;
            }
            QCheckBox {
                color: white;
                font-size: 12px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
            QCheckBox::indicator:unchecked {
                border: 2px solid #4a4a4a;
                background: #2b2b2b;
            }
            QCheckBox::indicator:checked {
                background: #800000;
                border: 2px solid #0d6efd;
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
                background: #800000;
                width: 16px;
                height: 16px;
                margin: -6px 0;
                border-radius: 8px;
            }
        """
        )

        # Carrega os classificadores para detecção facial
        cascade_path = "/usr/share/opencv4/haarcascades/"  # Path for Arch Linux
        self.face_cascade = cv2.CascadeClassifier(
            os.path.join(cascade_path, "haarcascade_frontalface_default.xml")
        )
        self.eye_cascade = cv2.CascadeClassifier(
            os.path.join(cascade_path, "haarcascade_eye.xml")
        )

        # Inicialização das variáveis
        self.capture = cv2.VideoCapture(0)
        self.current_frame = None
        self.processed_frame = None
        self.is_processing = True

        # Dicionário para controlar filtros ativos
        self.active_filters = {
            "gray": False,
            "canny": False,
            "blur": False,
            "sepia": False,
            "face_detection": False,
        }

        # Configuração dos parâmetros de processamento
        self.blur_value = 1
        self.brightness = 0
        self.contrast = 1
        self.canny_low = 100
        self.canny_high = 200

        # Configuração da interface
        self.setup_ui()

        # Timer para atualização do frame
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def add_filter_checkbox(self, text, filter_name, layout):
        """Adiciona uma checkbox para o filtro ao layout"""
        checkbox = QCheckBox(text)
        checkbox.stateChanged.connect(lambda state: self.toggle_filter(filter_name, state))
        layout.addWidget(checkbox)
        return checkbox

    def toggle_filter(self, filter_name, state):
        """Alterna o estado do filtro"""
        self.active_filters[filter_name] = bool(state)

    def setup_ui(self):
        """Configura a interface gráfica principal"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)

        # Área de exibição do vídeo
        self.video_label = QLabel()
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setStyleSheet("border: 2px solid #4a4a4a;")
        layout.addWidget(self.video_label)

        # Painel de controles
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_widget.setFixedWidth(400)

        # Título da seção de filtros
        filters_label = QLabel("Filtros Disponíveis:")
        filters_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        controls_layout.addWidget(filters_label)

        # Checkboxes para filtros
        self.add_filter_checkbox("Escala de Cinza", "gray", controls_layout)
        self.add_filter_checkbox("Detecção de Bordas", "canny", controls_layout)
        self.add_filter_checkbox("Desfoque", "blur", controls_layout)
        self.add_filter_checkbox("Sépia", "sepia", controls_layout)
        self.add_filter_checkbox("Detectar Face/Olhos", "face_detection", controls_layout)

        # Sliders de controle
        self.blur_slider = self.add_slider("Desfoque", 1, 21, self.blur_value, controls_layout)
        self.brightness_slider = self.add_slider("Brilho", -100, 100, self.brightness, controls_layout)
        self.contrast_slider = self.add_slider("Contraste", 0, 200, self.contrast, controls_layout)

        # Botões de ação
        self.add_action_button("Capturar Imagem", self.capture_image, controls_layout)
        self.add_action_button("Salvar Imagem", self.save_image, controls_layout)

        layout.addWidget(controls_widget)

    def detect_faces(self, frame):
        """Detecta faces e olhos na imagem"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Desenha retângulo ao redor da face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Região de interesse para os olhos
            roi_gray = gray[y : y + h, x : x + w]
            roi_color = frame[y : y + h, x : x + w]

            # Detecta olhos na região da face
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        return frame

    def process_frame(self):
        """Processa o frame atual com os filtros selecionados"""
        if self.current_frame is None:
            return

        # Começa com o frame original
        self.processed_frame = self.current_frame.copy()

        # Aplica cada filtro ativo em sequência
        if self.active_filters["gray"]:
            self.processed_frame = cv2.cvtColor(self.processed_frame, cv2.COLOR_BGR2GRAY)
            self.processed_frame = cv2.cvtColor(self.processed_frame, cv2.COLOR_GRAY2BGR)

        if self.active_filters["blur"]:
            k_size = self.blur_slider.value() * 2 + 1
            self.processed_frame = cv2.GaussianBlur(self.processed_frame, (k_size, k_size), 0)

        if self.active_filters["canny"]:
            gray = cv2.cvtColor(self.processed_frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, self.canny_low, self.canny_high)
            self.processed_frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        if self.active_filters["sepia"]:
            self.processed_frame = self.apply_sepia()

        if self.active_filters["face_detection"]:
            self.processed_frame = self.detect_faces(self.processed_frame)

    def apply_sepia(self):
        """Aplica o efeito sépia à imagem"""
        kernel = np.array(
            [
                [0.272, 0.534, 0.131],
                [0.349, 0.686, 0.168],
                [0.393, 0.769, 0.189],
            ]
        )
        return cv2.transform(self.processed_frame, kernel)

    def update_frame(self):
        """Atualiza o frame do vídeo"""
        ret, frame = self.capture.read()
        if ret:
            self.current_frame = cv2.flip(frame, 1)
            self.process_frame()
            self.display_frame()

    def display_frame(self):
        """Exibe o frame processado na interface"""
        if self.processed_frame is not None:
            frame = cv2.cvtColor(self.processed_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
                self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio
            ))

    def add_slider(self, name, min_val, max_val, default_val, layout):
        """Adiciona um slider de controle ao layout"""
        label = QLabel(name)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(default_val)
        layout.addWidget(label)
        layout.addWidget(slider)
        return slider

    def add_action_button(self, text, function, layout):
        """Adiciona um botão de ação ao layout"""
        button = QPushButton(text)
        button.clicked.connect(function)
        layout.addWidget(button)

    def capture_image(self):
        """Captura o frame atual"""
        if self.processed_frame is not None:
            self.captured_frame = self.processed_frame.copy()

    def save_image(self):
        """Salva a imagem processada"""
        if self.processed_frame is not None:
            filename, _ = QFileDialog.getSaveFileName(
                self, "Salvar Imagem", "", "Imagens (*.png *.jpg *.jpeg)"
            )
            if filename:
                cv2.imwrite(filename, self.processed_frame)

    def closeEvent(self, event):
        """Método chamado quando a janela é fechada"""
        self.capture.release()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ProcessadorImagem()
    window.show()
    sys.exit(app.exec())
