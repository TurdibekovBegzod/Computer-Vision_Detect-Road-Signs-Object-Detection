from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO

from PyQt5.QtCore import QObject, QPoint, QSize, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QIcon, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from detect import GROUP_MEANINGS_UZ
from sign_definitions import SIGN_DEFINITIONS


GROUP_COLORS = {
    "I": ("#F59E0B", "#FFF7E6"),
    "II": ("#2563EB", "#EFF6FF"),
    "III": ("#DC2626", "#FEF2F2"),
    "IV": ("#16A34A", "#F0FDF4"),
    "V": ("#0891B2", "#ECFEFF"),
    "VI": ("#7C3AED", "#F5F3FF"),
    "VII": ("#475569", "#F8FAFC"),
}


def resource_path(relative_path: str) -> Path:
    base_path = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base_path / relative_path


class DetectionWorker(QObject):
    finished = pyqtSignal(str, list)
    failed = pyqtSignal(str)

    def __init__(self, model: YOLO, image_path: str, confidence: float) -> None:
        super().__init__()
        self.model = model
        self.image_path = image_path
        self.confidence = confidence

    def run(self) -> None:
        try:
            results = self.model.predict(
                source=self.image_path,
                conf=self.confidence,
                imgsz=640,
                save=False,
                verbose=False,
            )
            detections = []
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    class_id = int(box.cls[0].item())
                    class_name = self.model.names[class_id]
                    group = class_name.split("-", 1)[0]
                    x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
                    detections.append(
                        {
                            "image_path": self.image_path,
                            "class_id": class_id,
                            "class_name": class_name,
                            "group": group,
                            "type": GROUP_MEANINGS_UZ.get(group, "Noma'lum guruh"),
                            "confidence": float(box.conf[0].item()),
                            "box": (x1, y1, x2, y2),
                        }
                    )
            self.finished.emit(self.image_path, detections)
        except Exception as error:
            self.failed.emit(str(error))


class ImagePanel(QLabel):
    clicked = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self._pixmap: QPixmap | None = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(680, 390)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setText("Rasm yuklash")
        self.setStyleSheet(
            """
            QLabel {
                border: 2px dashed #94A3B8;
                border-radius: 8px;
                background: #F8FAFC;
                color: #334155;
                font-size: 24px;
                font-weight: 700;
            }
            """
        )

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            self.clicked.emit()

    def set_display_pixmap(self, pixmap: QPixmap) -> None:
        self._pixmap = pixmap
        self._refresh_scaled_pixmap()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._refresh_scaled_pixmap()

    def _refresh_scaled_pixmap(self) -> None:
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.setPixmap(scaled)


class DragScrollArea(QScrollArea):
    def __init__(self) -> None:
        super().__init__()
        self._dragging = False
        self._last_pos = QPoint()
        self.setCursor(Qt.OpenHandCursor)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self._last_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._dragging:
            delta = event.pos() - self._last_pos
            self._last_pos = event.pos()
            scrollbar = self.verticalScrollBar()
            scrollbar.setValue(scrollbar.value() - delta.y())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton and self._dragging:
            self._dragging = False
            self.setCursor(Qt.OpenHandCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        if self._dragging:
            self._dragging = False
            self.setCursor(Qt.OpenHandCursor)
        super().leaveEvent(event)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self.widget() is not None:
            self.widget().setFixedWidth(self.viewport().width())


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Yo'l belgilarini aniqlash")
        self.resize(1120, 780)

        self.model_path = resource_path("best.pt")
        self.model = YOLO(str(self.model_path))
        self.definitions = SIGN_DEFINITIONS
        self.camera_turn_on_icon = QIcon(str(resource_path("images/camera_turn_on.png")))
        self.camera_turn_off_icon = QIcon(str(resource_path("images/camera_turn_off.png")))
        self.picture_icon = QIcon(str(resource_path("images/picture.png")))
        self.thread: QThread | None = None
        self.worker: DetectionWorker | None = None
        self.current_image_path: str | None = None
        self.camera_capture = None
        self.camera_running = False
        self.camera_frame_index = 0
        self.camera_detection_interval = 30
        self.current_camera_frame_pixmap: QPixmap | None = None
        self.camera_last_detections: list[dict] = []
        self.camera_timer = QTimer(self)
        self.camera_timer.timeout.connect(self.process_camera_frame)

        self.image_panel = ImagePanel()
        self.image_panel.clicked.connect(self.choose_image)
        self.reset_image_panel()

        status_text = "Model tayyor. Rasm yuklang."
        if self.definitions:
            status_text = f"Model tayyor. {len(self.definitions)} ta definition yuklandi."
        self.status_label = QLabel(status_text)
        self.status_label.setStyleSheet("color: #475569; font-size: 14px;")

        self.confidence_input = QDoubleSpinBox()
        self.confidence_input.setRange(0.05, 0.95)
        self.confidence_input.setSingleStep(0.05)
        self.confidence_input.setValue(0.25)
        self.confidence_input.setDecimals(2)
        self.confidence_input.setFixedWidth(88)

        info_button = QPushButton("Belgilar haqida")
        info_button.clicked.connect(self.show_sign_info)
        info_button.setCursor(Qt.PointingHandCursor)
        info_button.setStyleSheet(
            """
            QPushButton {
                background: #F8FAFC;
                color: #0F172A;
                border: 1px solid #CBD5E1;
                border-radius: 7px;
                padding: 10px 16px;
                font-weight: 700;
            }
            QPushButton:hover { background: #E2E8F0; }
            """
        )

        camera_button = QPushButton("Kamerani yoqish")
        camera_button.setIcon(self.camera_turn_on_icon)
        camera_button.setIconSize(QSize(20, 20))
        camera_button.clicked.connect(self.toggle_camera)
        camera_button.setCursor(Qt.PointingHandCursor)
        camera_button.setStyleSheet(
            """
            QPushButton {
                background: #0F766E;
                color: white;
                border: none;
                border-radius: 7px;
                padding: 10px 16px;
                font-weight: 700;
            }
            QPushButton:hover { background: #115E59; }
            QPushButton:disabled { background: #94A3B8; }
            """
        )
        self.camera_button = camera_button

        recalculate_button = QPushButton("Qayta hisoblash")
        recalculate_button.clicked.connect(self.recalculate_current_image)
        recalculate_button.setCursor(Qt.PointingHandCursor)
        recalculate_button.setEnabled(False)
        recalculate_button.setStyleSheet(
            """
            QPushButton {
                background: #F8FAFC;
                color: #0F172A;
                border: 1px solid #CBD5E1;
                border-radius: 7px;
                padding: 10px 16px;
                font-weight: 700;
            }
            QPushButton:hover { background: #E2E8F0; }
            QPushButton:disabled {
                color: #94A3B8;
                background: #F1F5F9;
            }
            """
        )
        self.recalculate_button = recalculate_button

        upload_button = QPushButton("Rasm yuklash")
        upload_button.setIcon(self.picture_icon)
        upload_button.setIconSize(QSize(20, 20))
        upload_button.clicked.connect(self.choose_image)
        upload_button.setCursor(Qt.PointingHandCursor)
        upload_button.setStyleSheet(
            """
            QPushButton {
                background: #2563EB;
                color: white;
                border: none;
                border-radius: 7px;
                padding: 10px 18px;
                font-weight: 700;
            }
            QPushButton:hover { background: #1D4ED8; }
            QPushButton:disabled { background: #94A3B8; }
            """
        )
        self.upload_button = upload_button

        self.summary_layout = QVBoxLayout()
        self.summary_layout.setSpacing(8)
        self.summary_layout.setAlignment(Qt.AlignTop)

        self.detections_layout = QVBoxLayout()
        self.detections_layout.setSpacing(10)
        self.detections_layout.setAlignment(Qt.AlignTop)

        result_content = QWidget()
        result_content.setLayout(self.detections_layout)
        result_content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        result_scroll = DragScrollArea()
        result_scroll.setWidgetResizable(True)
        result_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        result_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        result_scroll.setWidget(result_content)
        result_scroll.setMinimumHeight(130)
        result_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        result_scroll.setFrameShape(QFrame.NoFrame)
        result_scroll.setStyleSheet(
            """
            QScrollArea {
                background: white;
                border: 1px solid #E2E8F0;
                border-radius: 8px;
            }
            QScrollBar:vertical {
                background: #F8FAFC;
                width: 12px;
                margin: 4px 2px 4px 2px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: #CBD5E1;
                border-radius: 6px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background: #94A3B8;
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0;
            }
            """
        )
        self.result_scroll = result_scroll

        lower_panel = QWidget()
        lower_layout = QVBoxLayout()
        lower_layout.setContentsMargins(0, 0, 0, 0)
        lower_layout.setSpacing(10)
        lower_layout.addWidget(self.status_label)
        lower_layout.addLayout(self.summary_layout)
        lower_layout.addWidget(result_scroll)
        lower_panel.setLayout(lower_layout)
        lower_panel.setMinimumWidth(300)
        self.image_panel.setMinimumWidth(420)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.image_panel)
        splitter.addWidget(lower_panel)
        splitter.setChildrenCollapsible(False)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([720, 380])
        splitter.setHandleWidth(10)
        splitter.setStyleSheet(
            """
            QSplitter::handle:horizontal {
                background: #E2E8F0;
                border-left: 1px solid #CBD5E1;
                border-right: 1px solid #CBD5E1;
                margin: 0 3px;
                border-radius: 4px;
            }
            QSplitter::handle:horizontal:hover {
                background: #94A3B8;
            }
            """
        )
        self.splitter = splitter

        header = QHBoxLayout()
        title = QLabel("Yo'l belgilari detektori")
        title.setStyleSheet("font-size: 24px; font-weight: 800; color: #0F172A;")
        header.addWidget(title)
        header.addStretch()
        header.addWidget(QLabel("Confidence:"))
        header.addWidget(self.confidence_input)
        header.addWidget(info_button)
        header.addWidget(camera_button)
        header.addWidget(recalculate_button)
        header.addWidget(upload_button)

        root = QVBoxLayout()
        root.setContentsMargins(22, 18, 22, 18)
        root.setSpacing(14)
        root.addLayout(header)
        root.addWidget(splitter, stretch=1)

        container = QWidget()
        container.setLayout(root)
        container.setStyleSheet("background: white;")
        self.setCentralWidget(container)
        self._set_empty_results()

    def reset_image_panel(self) -> None:
        self.image_panel.clear()
        self.image_panel._pixmap = None
        self.image_panel.setText("Rasm yuklash")

    def show_sign_info(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Belgilar haqida")
        dialog.resize(620, 520)

        layout = QVBoxLayout()
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(10)

        title = QLabel("Yo'l belgilarining turlari")
        title.setStyleSheet("font-size: 22px; font-weight: 800; color: #0F172A; border: none;")
        layout.addWidget(title)

        subtitle = QLabel("Natijalar pastki panelda shu turlarga mos rang bilan ajratib ko'rsatiladi.")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("font-size: 14px; color: #475569; border: none;")
        layout.addWidget(subtitle)

        class_counts = Counter(name.split("-", 1)[0] for name in self.model.names.values())
        for group in ["I", "II", "III", "IV", "V", "VI", "VII"]:
            layout.addWidget(self._info_row(group, class_counts.get(group, 0)))

        close_button = QPushButton("Yopish")
        close_button.clicked.connect(dialog.accept)
        close_button.setCursor(Qt.PointingHandCursor)
        close_button.setStyleSheet(
            """
            QPushButton {
                background: #111827;
                color: white;
                border: none;
                border-radius: 7px;
                padding: 10px 18px;
                font-weight: 700;
            }
            QPushButton:hover { background: #1F2937; }
            """
        )
        layout.addWidget(close_button, alignment=Qt.AlignRight)

        dialog.setLayout(layout)
        dialog.exec_()

    def _info_row(self, group: str, count: int) -> QFrame:
        border, background = GROUP_COLORS.get(group, ("#111827", "#F8FAFC"))
        row = QFrame()
        row.setStyleSheet(
            f"""
            QFrame {{
                background: {background};
                border: 2px solid {border};
                border-radius: 8px;
            }}
            """
        )

        color_box = QLabel()
        color_box.setFixedSize(34, 34)
        color_box.setStyleSheet(f"background: {border}; border-radius: 6px; border: none;")

        code_label = QLabel(group)
        code_label.setFixedWidth(44)
        code_label.setAlignment(Qt.AlignCenter)
        code_label.setStyleSheet(
            f"color: {border}; border: none; font-size: 18px; font-weight: 900;"
        )

        text = QLabel(f"{GROUP_MEANINGS_UZ.get(group, group)} | modelda {count} ta kod")
        text.setWordWrap(True)
        text.setStyleSheet("color: #0F172A; border: none; font-size: 14px; font-weight: 700;")

        row_layout = QHBoxLayout()
        row_layout.setContentsMargins(12, 10, 12, 10)
        row_layout.setSpacing(12)
        row_layout.addWidget(color_box)
        row_layout.addWidget(code_label)
        row_layout.addWidget(text, stretch=1)
        row.setLayout(row_layout)
        return row

    def choose_image(self) -> None:
        image_path, _ = QFileDialog.getOpenFileName(
            self,
            "Rasm tanlang",
            str(Path.cwd()),
            "Images (*.jpg *.jpeg *.png *.bmp *.webp)",
        )
        if image_path:
            self.stop_camera(update_status=False, clear_preview=False)
            self.detect_image(image_path)

    def toggle_camera(self) -> None:
        if self.camera_running:
            self.stop_camera()
            return

        capture = cv2.VideoCapture(0)
        if not capture.isOpened():
            self.status_label.setText("Kamera ochilmadi. Kamera ulanganini tekshiring.")
            capture.release()
            return

        self.camera_capture = capture
        self.camera_running = True
        self.camera_frame_index = 0
        self.current_image_path = None
        self.current_camera_frame_pixmap = None
        self.camera_last_detections = []
        self.camera_button.setText("Kamerani o'chirish")
        self.camera_button.setIcon(self.camera_turn_off_icon)
        self.camera_button.setIconSize(QSize(20, 20))
        self.camera_button.setStyleSheet(
            """
            QPushButton {
                background: #DC2626;
                color: white;
                border: none;
                border-radius: 7px;
                padding: 10px 16px;
                font-weight: 700;
            }
            QPushButton:hover { background: #B91C1C; }
            """
        )
        self.upload_button.setEnabled(False)
        self.recalculate_button.setEnabled(False)
        self.status_label.setText("?? Kamera ishlayapti...")
        self._set_empty_results("Kamera obyektlarni qidiryapti...")
        self.camera_timer.start(30)

    def stop_camera(self, update_status: bool = True, clear_preview: bool = True) -> None:
        if self.camera_timer.isActive():
            self.camera_timer.stop()
        if self.camera_capture is not None:
            self.camera_capture.release()
            self.camera_capture = None
        was_running = self.camera_running
        self.camera_running = False
        self.current_camera_frame_pixmap = None
        self.camera_last_detections = []
        if clear_preview and was_running:
            self.reset_image_panel()
            self._set_empty_results()
        self.camera_button.setText("Kamerani yoqish")
        self.camera_button.setIcon(self.camera_turn_on_icon)
        self.camera_button.setIconSize(QSize(20, 20))
        self.camera_button.setStyleSheet(
            """
            QPushButton {
                background: #0F766E;
                color: white;
                border: none;
                border-radius: 7px;
                padding: 10px 16px;
                font-weight: 700;
            }
            QPushButton:hover { background: #115E59; }
            QPushButton:disabled { background: #94A3B8; }
            """
        )
        self.upload_button.setEnabled(True)
        self.recalculate_button.setEnabled(self.current_image_path is not None)
        if update_status and was_running:
            self.status_label.setText("Kamera o'chirildi.")

    def process_camera_frame(self) -> None:
        if self.camera_capture is None:
            self.stop_camera()
            return

        ok, frame = self.camera_capture.read()
        if not ok:
            self.status_label.setText("Kameradan kadr olinmadi.")
            self.stop_camera(update_status=False)
            return

        self.camera_frame_index += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pixmap = QPixmap.fromImage(self._qimage_from_rgb_frame(rgb_frame))
        self.current_camera_frame_pixmap = frame_pixmap

        if self.camera_frame_index % self.camera_detection_interval != 0:
            display_pixmap = self._draw_detections_on_pixmap(frame_pixmap, self.camera_last_detections) if self.camera_last_detections else frame_pixmap
            self.image_panel.set_display_pixmap(display_pixmap)
            return

        results = self.model.predict(
            source=frame,
            conf=self.confidence_input.value(),
            imgsz=640,
            save=False,
            verbose=False,
        )
        detections = self._detections_from_results(results, image_path=None)
        self.camera_last_detections = detections
        annotated = self._draw_detections_on_pixmap(frame_pixmap, detections)
        self.image_panel.set_display_pixmap(annotated)
        self._render_results(detections)
        self.status_label.setText(f"?? Kamera: topilgan belgilar {len(detections)} ta")

    def recalculate_current_image(self) -> None:
        if self.current_image_path is None:
            self.status_label.setText("Avval rasm yuklang.")
            return
        self.detect_image(self.current_image_path)

    def detect_image(self, image_path: str) -> None:
        self.stop_camera(update_status=False, clear_preview=False)
        self.current_image_path = image_path
        self.upload_button.setEnabled(False)
        self.recalculate_button.setEnabled(False)
        self.status_label.setText("Aniqlanmoqda...")
        self._set_empty_results("Natija tayyorlanmoqda")

        pixmap = QPixmap(image_path)
        self.image_panel.set_display_pixmap(pixmap)

        self.thread = QThread()
        self.worker = DetectionWorker(self.model, image_path, self.confidence_input.value())
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_detection_finished)
        self.worker.failed.connect(self.on_detection_failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def on_detection_finished(self, image_path: str, detections: list[dict]) -> None:
        pixmap = self._draw_detections(image_path, detections)
        self.image_panel.set_display_pixmap(pixmap)
        self._render_results(detections)
        self.status_label.setText(f"Topilgan belgilar: {len(detections)} ta")
        self.upload_button.setEnabled(True)
        self.recalculate_button.setEnabled(True)

    def on_detection_failed(self, message: str) -> None:
        self.status_label.setText(f"Xatolik: {message}")
        self.upload_button.setEnabled(True)
        self.recalculate_button.setEnabled(self.current_image_path is not None)

    def _qimage_from_rgb_frame(self, rgb_frame) -> QImage:
        height, width, channels = rgb_frame.shape
        bytes_per_line = channels * width
        return QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format_RGB888).copy()

    def _detections_from_results(self, results, image_path: str | None) -> list[dict]:
        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                class_name = self.model.names[class_id]
                group = class_name.split("-", 1)[0]
                x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
                detections.append(
                    {
                        "image_path": image_path,
                        "class_id": class_id,
                        "class_name": class_name,
                        "group": group,
                        "type": GROUP_MEANINGS_UZ.get(group, "Noma'lum guruh"),
                        "confidence": float(box.conf[0].item()),
                        "box": (x1, y1, x2, y2),
                    }
                )
        return detections

    def _draw_detections(self, image_path: str, detections: list[dict]) -> QPixmap:
        image = QImage(image_path).convertToFormat(QImage.Format_ARGB32)
        pixmap = QPixmap.fromImage(image)
        return self._draw_detections_on_pixmap(pixmap, detections)

    def _draw_detections_on_pixmap(self, source_pixmap: QPixmap, detections: list[dict]) -> QPixmap:
        pixmap = QPixmap(source_pixmap)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        for detection in detections:
            color_hex, _ = GROUP_COLORS.get(detection["group"], ("#111827", "#F8FAFC"))
            color = QColor(color_hex)
            x1, y1, x2, y2 = detection["box"]
            pen = QPen(color, max(3, int(pixmap.width() / 220)))
            painter.setPen(pen)
            painter.drawRect(int(x1), int(y1), int(x2 - x1), int(y2 - y1))

            label = f"{detection['class_name']} {detection['confidence']:.2f}"
            font = QFont("Arial", max(10, int(pixmap.width() / 70)), QFont.Bold)
            painter.setFont(font)
            metrics = painter.fontMetrics()
            label_width = metrics.horizontalAdvance(label) + 14
            label_height = metrics.height() + 8
            label_y = max(0, int(y1) - label_height)
            painter.fillRect(int(x1), label_y, label_width, label_height, color)
            painter.setPen(QColor("white"))
            painter.drawText(int(x1) + 7, label_y + label_height - 7, label)

        painter.end()
        return pixmap

    def _render_results(self, detections: list[dict]) -> None:
        self._clear_layout(self.summary_layout)
        self._clear_layout(self.detections_layout)

        if not detections:
            self._set_empty_results("Belgi topilmadi")
            return

        grouped_codes: dict[str, list[str]] = defaultdict(list)
        for detection in detections:
            grouped_codes[detection["group"]].append(detection["class_name"])

        for group, count in Counter(detection["group"] for detection in detections).items():
            codes = ", ".join(sorted(set(grouped_codes[group])))
            text = f"{GROUP_MEANINGS_UZ.get(group, group)}: {count} ta | kodlar: {codes}"
            self.summary_layout.addWidget(self._chip(text, group, bold=True))

        for detection in detections:
            definition = self.definitions.get(
                detection["class_name"],
                "Bu kod uchun Word faylidan definition topilmadi.",
            )
            self.detections_layout.addWidget(self._result_card(detection, definition))
        self.detections_layout.addStretch()

    def _set_empty_results(self, text: str = "Pastda topilgan belgilar ranglar bilan chiqadi") -> None:
        self._clear_layout(self.summary_layout)
        self._clear_layout(self.detections_layout)
        empty = QLabel(text)
        empty.setStyleSheet("color: #64748B; font-size: 14px; padding: 10px 2px;")
        self.detections_layout.addWidget(empty)
        self.detections_layout.addStretch()

    def _chip(self, text: str, group: str, bold: bool = False) -> QLabel:
        border, background = GROUP_COLORS.get(group, ("#111827", "#F8FAFC"))
        chip = QLabel(text)
        chip.setWordWrap(True)
        chip.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        chip.setStyleSheet(
            f"""
            QLabel {{
                background: {background};
                color: #0F172A;
                border: 2px solid {border};
                border-radius: 7px;
                padding: 8px 10px;
                font-size: 13px;
                font-weight: {700 if bold else 600};
            }}
            """
        )
        return chip

    def _result_card(self, detection: dict, definition: str) -> QFrame:
        border, background = GROUP_COLORS.get(detection["group"], ("#111827", "#F8FAFC"))

        card = QFrame()
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        card.setStyleSheet(
            f"""
            QFrame {{
                background: {background};
                border: 2px solid {border};
                border-radius: 8px;
            }}
            """
        )

        thumbnail = QLabel()
        thumbnail.setFixedSize(92, 92)
        thumbnail.setAlignment(Qt.AlignCenter)
        thumbnail.setStyleSheet(
            f"""
            QLabel {{
                background: white;
                border: 3px solid {border};
                border-radius: 7px;
            }}
            """
        )
        thumb_pixmap = self._crop_thumbnail(detection)
        if thumb_pixmap is not None:
            thumbnail.setPixmap(
                thumb_pixmap.scaled(82, 82, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        else:
            thumbnail.setText(detection["class_name"])

        code_label = QLabel(
            f"{detection['class_name']} | {detection['type']} | "
            f"{detection['confidence'] * 100:.1f}%"
        )
        code_label.setWordWrap(True)
        code_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        code_label.setStyleSheet(
            f"color: {border}; border: none; font-size: 15px; font-weight: 800;"
        )

        definition_label = QLabel(definition)
        definition_label.setWordWrap(True)
        definition_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        definition_label.setStyleSheet(
            "color: #0F172A; border: none; font-size: 14px; line-height: 1.3;"
        )

        text_layout = QVBoxLayout()
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(6)
        text_layout.addWidget(code_label)
        text_layout.addWidget(definition_label)

        layout = QHBoxLayout()
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(12)
        layout.addWidget(thumbnail)
        layout.addLayout(text_layout, stretch=1)
        card.setLayout(layout)
        return card

    def _crop_thumbnail(self, detection: dict) -> QPixmap | None:
        if detection.get("image_path"):
            pixmap = QPixmap(detection["image_path"])
        else:
            pixmap = QPixmap(self.current_camera_frame_pixmap) if self.current_camera_frame_pixmap else QPixmap()
        if pixmap.isNull():
            return None
        x1, y1, x2, y2 = detection["box"]
        padding = max(4, int(min(pixmap.width(), pixmap.height()) * 0.015))
        x = max(0, int(x1) - padding)
        y = max(0, int(y1) - padding)
        width = min(pixmap.width() - x, int(x2 - x1) + padding * 2)
        height = min(pixmap.height() - y, int(y2 - y1) + padding * 2)
        if width <= 0 or height <= 0:
            return None
        return pixmap.copy(x, y, width, height)

    def _clear_layout(self, layout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()


    def closeEvent(self, event) -> None:  # noqa: N802
        self.stop_camera(update_status=False, clear_preview=False)
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
