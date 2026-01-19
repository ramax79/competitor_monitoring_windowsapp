import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTabWidget,
    QTextEdit,
    QPushButton,
    QLabel,
    QFileDialog,
    QLineEdit,
    QScrollArea,
    QFrame,
    QMessageBox,
    QGroupBox,
    QListWidget,
    QListWidgetItem,
)

# Конфигурация
# Можно переопределить через переменную окружения: API_BASE=http://your-server:8000
API_BASE = os.getenv("API_BASE", "http://localhost:8000")
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 МБ


class AnalysisWorker(QThread):
    """Воркер для асинхронного выполнения запросов к API"""
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, request_type: str, data: dict):
        super().__init__()
        self.request_type = request_type
        self.data = data

    def run(self):
        try:
            if self.request_type == "text":
                self.progress.emit("Отправка текста на анализ...")
                response = requests.post(
                    f"{API_BASE}/analyze_text",
                    json={"text": self.data["text"]},
                    timeout=60,
                )
                response.raise_for_status()
                self.finished.emit({"type": "text", "data": response.json()})

            elif self.request_type == "image":
                self.progress.emit("Загрузка изображения...")
                with open(self.data["file_path"], "rb") as f:
                    files = {"file": (self.data["filename"], f, self.data["content_type"])}
                    response = requests.post(
                        f"{API_BASE}/analyze_image",
                        files=files,
                        timeout=120,
                    )
                response.raise_for_status()
                self.finished.emit({"type": "image", "data": response.json()})

            elif self.request_type == "parse":
                self.progress.emit("Парсинг URL через Selenium...")
                response = requests.post(
                    f"{API_BASE}/parse_demo",
                    json={"url": self.data["url"]},
                    timeout=120,
                )
                response.raise_for_status()
                result = response.json()
                self.finished.emit({"type": "parse", "data": result})

        except requests.exceptions.RequestException as e:
            error_msg = f"Ошибка запроса: {str(e)}"
            if hasattr(e.response, "json"):
                try:
                    error_detail = e.response.json().get("detail", str(e))
                    error_msg = f"Ошибка: {error_detail}"
                except:
                    pass
            self.error.emit(error_msg)
        except Exception as e:
            self.error.emit(f"Неожиданная ошибка: {str(e)}")


class ExpandableSection(QFrame):
    """Раскрывающаяся секция для результатов"""
    def __init__(self, title: str, content_widget: QWidget, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Shape.StyledPanel)
        self.setStyleSheet("""
            QFrame {
                border: 1px solid rgba(31, 41, 55, 0.9);
                border-radius: 8px;
                background-color: #020617;
                margin: 4px;
            }
        """)
        
        self.is_expanded = True
        self.content_widget = content_widget
        
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 8, 10, 8)
        
        # Заголовок (кликабельный)
        header = QPushButton(title)
        header.setStyleSheet("""
            QPushButton {
                text-align: left;
                font-weight: bold;
                font-size: 11px;
                text-transform: uppercase;
                letter-spacing: 1px;
                color: #9ca3af;
                border: none;
                padding: 4px;
                background: transparent;
            }
            QPushButton:hover {
                color: #38bdf8;
            }
        """)
        header.clicked.connect(self.toggle)
        layout.addWidget(header)
        
        # Контент
        self.content_container = QWidget()
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.addWidget(content_widget)
        self.content_container.setLayout(content_layout)
        layout.addWidget(self.content_container)
        
        self.setLayout(layout)
        self.update_display()
    
    def toggle(self):
        self.is_expanded = not self.is_expanded
        self.update_display()
    
    def update_display(self):
        self.content_container.setVisible(self.is_expanded)
        # Обновляем текст кнопки с индикатором
        button = self.findChild(QPushButton)
        if button:
            indicator = "▼" if self.is_expanded else "▶"
            text = button.text().replace("▼", "").replace("▶", "").strip()
            button.setText(f"{indicator} {text}")


class CompetitorMonitorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.worker: Optional[AnalysisWorker] = None
        self.selected_image_path: Optional[str] = None
        self.init_ui()
        self.check_backend_connection()
        self.load_history()
    
    def check_backend_connection(self):
        """Проверка подключения к backend при запуске"""
        try:
            response = requests.get(f"{API_BASE}/", timeout=5)
            if response.status_code == 200:
                self.status_label.setText("Подключено к backend")
                self.status_label.setStyleSheet("font-size: 10px; color: #4ade80; margin-top: 8px;")
            else:
                self.status_label.setText("Backend недоступен")
                self.status_label.setStyleSheet("font-size: 10px; color: #f97373; margin-top: 8px;")
        except Exception as e:
            self.status_label.setText(f"Ошибка подключения: {str(e)[:50]}")
            self.status_label.setStyleSheet("font-size: 10px; color: #f97373; margin-top: 8px;")
            QMessageBox.warning(
                self,
                "Предупреждение",
                f"Не удалось подключиться к backend на {API_BASE}\n\n"
                "Убедитесь, что backend запущен:\n"
                "uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000"
            )

    def init_ui(self):
        self.setWindowTitle("Мониторинг конкурентов — MVP Ассистент")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0f172a;
            }
            QWidget {
                color: #e5e7eb;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QTabWidget::pane {
                border: 1px solid rgba(31, 41, 55, 0.9);
                background: #020617;
            }
            QTabBar::tab {
                background: rgba(15, 23, 42, 0.9);
                color: #9ca3af;
                padding: 8px 16px;
                border: 1px solid rgba(31, 41, 55, 0.9);
                border-bottom: none;
            }
            QTabBar::tab:selected {
                background: #0369a1;
                color: #ecfeff;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #0369a1, stop:1 #0ea5e9);
                color: #ecfeff;
                border: none;
                border-radius: 20px;
                padding: 8px 20px;
                font-weight: bold;
                font-size: 11px;
                text-transform: uppercase;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #0284c7, stop:1 #0ea5e9);
            }
            QPushButton:disabled {
                background: #374151;
                color: #6b7280;
            }
            QTextEdit, QLineEdit {
                background: #020617;
                border: 1px solid rgba(31, 41, 55, 0.9);
                border-radius: 8px;
                padding: 8px;
                color: #e5e7eb;
            }
            QTextEdit:focus, QLineEdit:focus {
                border: 1px solid rgba(56, 189, 248, 0.8);
            }
            QListWidget {
                background: #020617;
                border: 1px solid rgba(31, 41, 55, 0.9);
                border-radius: 8px;
                color: #e5e7eb;
            }
            QListWidgetItem {
                padding: 6px;
                border-bottom: 1px solid rgba(31, 41, 55, 0.5);
            }
            QListWidgetItem:hover {
                background: rgba(56, 189, 248, 0.1);
            }
            QScrollArea {
                border: none;
                background: transparent;
            }
        """)

        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # Левая панель (ввод)
        left_panel = QWidget()
        left_panel.setMaximumWidth(500)
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)

        # Заголовок
        title = QLabel("Мониторинг конкурентов")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #38bdf8; margin-bottom: 4px;")
        subtitle = QLabel("Аналитика конкурентных текстов и креативов на базе GPT-4o-mini")
        subtitle.setStyleSheet("font-size: 10px; color: #9ca3af; margin-bottom: 16px;")
        left_layout.addWidget(title)
        left_layout.addWidget(subtitle)

        # Вкладки
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid rgba(31, 41, 55, 0.9);
                background: #020617;
                border-radius: 8px;
            }
        """)

        # Вкладка 1: Текст
        text_tab = QWidget()
        text_layout = QVBoxLayout()
        text_label = QLabel("Текст конкурента:")
        text_label.setStyleSheet("font-size: 10px; color: #9ca3af; margin-bottom: 4px;")
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("Вставьте текст конкурента — заголовок, оффер, блоки лендинга...")
        self.text_input.setMinimumHeight(150)
        text_layout.addWidget(text_label)
        text_layout.addWidget(self.text_input)
        text_tab.setLayout(text_layout)
        self.tabs.addTab(text_tab, "Текст конкурента")

        # Вкладка 2: Изображение
        image_tab = QWidget()
        image_layout = QVBoxLayout()
        image_label = QLabel("Изображение конкурента:")
        image_label.setStyleSheet("font-size: 10px; color: #9ca3af; margin-bottom: 4px;")
        self.image_path_label = QLabel("Файл не выбран")
        self.image_path_label.setStyleSheet("font-size: 10px; color: #6b7280; padding: 8px; border: 1px dashed rgba(31, 41, 55, 0.9); border-radius: 8px;")
        self.image_path_label.setWordWrap(True)
        self.image_path_label.setMinimumHeight(60)
        select_image_btn = QPushButton("Выбрать изображение")
        select_image_btn.clicked.connect(self.select_image)
        image_layout.addWidget(image_label)
        image_layout.addWidget(self.image_path_label)
        image_layout.addWidget(select_image_btn)
        image_tab.setLayout(image_layout)
        self.tabs.addTab(image_tab, "Изображение")

        # Вкладка 3: URL
        url_tab = QWidget()
        url_layout = QVBoxLayout()
        url_label = QLabel("URL конкурента:")
        url_label.setStyleSheet("font-size: 10px; color: #9ca3af; margin-bottom: 4px;")
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("https://пример-конкурента.ru")
        url_layout.addWidget(url_label)
        url_layout.addWidget(self.url_input)
        url_tab.setLayout(url_layout)
        self.tabs.addTab(url_tab, "URL конкурента")

        left_layout.addWidget(self.tabs)

        # Кнопка анализа
        self.analyze_btn = QPushButton("Проанализировать")
        self.analyze_btn.clicked.connect(self.analyze)
        self.analyze_btn.setMinimumHeight(40)
        left_layout.addWidget(self.analyze_btn)

        # Статус
        self.status_label = QLabel("Готов к анализу")
        self.status_label.setStyleSheet("font-size: 10px; color: #9ca3af; margin-top: 8px;")
        left_layout.addWidget(self.status_label)

        left_layout.addStretch()
        main_layout.addWidget(left_panel)

        # Правая панель (результаты)
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)

        results_title = QLabel("Структурированная аналитика")
        results_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #38bdf8; margin-bottom: 4px;")
        results_subtitle = QLabel("Сильные / слабые стороны, УТП и рекомендации")
        results_subtitle.setStyleSheet("font-size: 10px; color: #9ca3af; margin-bottom: 12px;")
        right_layout.addWidget(results_title)
        right_layout.addWidget(results_subtitle)

        # Область результатов
        self.results_scroll = QScrollArea()
        self.results_scroll.setWidgetResizable(True)
        self.results_widget = QWidget()
        self.results_layout = QVBoxLayout()
        self.results_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.results_widget.setLayout(self.results_layout)
        self.results_scroll.setWidget(self.results_widget)
        right_layout.addWidget(self.results_scroll)

        # История
        history_group = QGroupBox("Последние запросы")
        history_group.setStyleSheet("""
            QGroupBox {
                font-size: 11px;
                font-weight: bold;
                color: #9ca3af;
                border: 1px solid rgba(31, 41, 55, 0.9);
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
        """)
        history_layout = QVBoxLayout()
        self.history_list = QListWidget()
        self.history_list.setMaximumHeight(150)
        self.history_list.itemClicked.connect(self.load_history_item)
        refresh_history_btn = QPushButton("Обновить историю")
        refresh_history_btn.setMaximumHeight(30)
        refresh_history_btn.clicked.connect(self.load_history)
        history_layout.addWidget(self.history_list)
        history_layout.addWidget(refresh_history_btn)
        history_group.setLayout(history_layout)
        right_layout.addWidget(history_group)

        main_layout.addWidget(right_panel, stretch=2)

    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите изображение",
            "",
            "Images (*.png *.jpg *.jpeg *.gif *.bmp);;All Files (*)"
        )
        if file_path:
            file_size = Path(file_path).stat().st_size
            if file_size > MAX_FILE_SIZE:
                QMessageBox.warning(
                    self,
                    "Ошибка",
                    f"Файл слишком большой ({file_size / 1024 / 1024:.2f} МБ). Максимальный размер: 5 МБ."
                )
                return
            self.image_path_label.setText(f"Выбрано: {Path(file_path).name} ({(file_size / 1024 / 1024):.2f} МБ)")
            self.image_path_label.setStyleSheet("font-size: 10px; color: #4ade80; padding: 8px; border: 1px solid rgba(34, 197, 94, 0.5); border-radius: 8px;")
            self.selected_image_path = file_path

    def analyze(self):
        current_tab = self.tabs.currentIndex()
        
        if current_tab == 0:  # Текст
            text = self.text_input.toPlainText().strip()
            if not text:
                QMessageBox.warning(self, "Ошибка", "Введите текст конкурента.")
                return
            self.start_analysis("text", {"text": text})

        elif current_tab == 1:  # Изображение
            if not self.selected_image_path:
                QMessageBox.warning(self, "Ошибка", "Выберите изображение для анализа.")
                return
            file_path = Path(self.selected_image_path)
            content_type = f"image/{file_path.suffix[1:].lower()}" if file_path.suffix else "image/png"
            self.start_analysis("image", {
                "file_path": str(file_path),
                "filename": file_path.name,
                "content_type": content_type,
            })

        elif current_tab == 2:  # URL
            url = self.url_input.text().strip()
            if not url:
                QMessageBox.warning(self, "Ошибка", "Введите URL конкурента.")
                return
            if not url.startswith(("http://", "https://")):
                url = "https://" + url
            self.start_analysis("parse", {"url": url})

    def start_analysis(self, request_type: str, data: dict):
        self.analyze_btn.setEnabled(False)
        self.status_label.setText("Анализ выполняется...")
        self.status_label.setStyleSheet("font-size: 10px; color: #38bdf8; margin-top: 8px;")
        
        # Очистка предыдущих результатов
        self.clear_results()
        
        self.worker = AnalysisWorker(request_type, data)
        self.worker.finished.connect(self.on_analysis_finished)
        self.worker.error.connect(self.on_analysis_error)
        self.worker.progress.connect(self.status_label.setText)
        self.worker.start()

    def on_analysis_finished(self, result: dict):
        self.analyze_btn.setEnabled(True)
        self.status_label.setText("Анализ завершён")
        self.status_label.setStyleSheet("font-size: 10px; color: #4ade80; margin-top: 8px;")
        
        result_type = result["type"]
        data = result["data"]
        
        if result_type == "text":
            self.display_text_results(data)
        elif result_type == "image":
            self.display_image_results(data)
        elif result_type == "parse":
            self.display_parse_results(data)
        
        self.load_history()

    def on_analysis_error(self, error_msg: str):
        self.analyze_btn.setEnabled(True)
        self.status_label.setText("Ошибка анализа")
        self.status_label.setStyleSheet("font-size: 10px; color: #f97373; margin-top: 8px;")
        QMessageBox.critical(self, "Ошибка", error_msg)

    def clear_results(self):
        while self.results_layout.count():
            child = self.results_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def display_text_results(self, data: dict):
        sections = [
            ("Сильные стороны", data.get("strengths", [])),
            ("Слабые стороны", data.get("weaknesses", [])),
            ("Уникальные предложения", data.get("unique_offers", [])),
            ("Рекомендации по стратегии", data.get("recommendations", [])),
        ]
        
        for title, items in sections:
            if items:
                content = QTextEdit()
                content.setReadOnly(True)
                content.setStyleSheet("background: transparent; border: none; font-size: 11px;")
                content_text = "\n".join(f"• {item}" for item in items)
                content.setPlainText(content_text)
                content.setMaximumHeight(200)
                
                section = ExpandableSection(title, content)
                self.results_layout.addWidget(section)

    def display_image_results(self, data: dict):
        # Описание
        if data.get("description"):
            desc = QTextEdit()
            desc.setReadOnly(True)
            desc.setStyleSheet("background: transparent; border: none; font-size: 11px;")
            desc.setPlainText(data["description"])
            desc.setMaximumHeight(150)
            section = ExpandableSection("Описание изображения", desc)
            self.results_layout.addWidget(section)
        
        # Маркетинговые инсайты
        if data.get("marketing_insights"):
            insights = QTextEdit()
            insights.setReadOnly(True)
            insights.setStyleSheet("background: transparent; border: none; font-size: 11px;")
            insights.setPlainText("\n".join(f"• {item}" for item in data["marketing_insights"]))
            insights.setMaximumHeight(200)
            section = ExpandableSection("Маркетинговые инсайты", insights)
            self.results_layout.addWidget(section)
        
        # Оценка визуального стиля
        if data.get("visual_style_evaluation"):
            style = QTextEdit()
            style.setReadOnly(True)
            style.setStyleSheet("background: transparent; border: none; font-size: 11px;")
            style.setPlainText("\n".join(f"• {item}" for item in data["visual_style_evaluation"]))
            style.setMaximumHeight(200)
            section = ExpandableSection("Оценка визуального стиля", style)
            self.results_layout.addWidget(section)
        
        # Оценка дизайна
        if "design_score" in data:
            score_label = QLabel(f"Текущая оценка визуального стиля: {data['design_score']:.1f} из 10")
            score_label.setStyleSheet("font-size: 12px; color: #38bdf8; padding: 8px;")
            section = ExpandableSection("Оценка дизайна", score_label)
            self.results_layout.addWidget(section)
        
        # Потенциал анимации
        if data.get("animation_potential"):
            anim = QTextEdit()
            anim.setReadOnly(True)
            anim.setStyleSheet("background: transparent; border: none; font-size: 11px;")
            anim.setPlainText("\n".join(f"• {item}" for item in data["animation_potential"]))
            anim.setMaximumHeight(200)
            section = ExpandableSection("Потенциал анимации", anim)
            self.results_layout.addWidget(section)

    def display_parse_results(self, data: dict):
        # Извлечённый контент
        if data.get("title") or data.get("h1") or data.get("first_paragraph"):
            extracted = QTextEdit()
            extracted.setReadOnly(True)
            extracted.setStyleSheet("background: transparent; border: none; font-size: 11px;")
            parts = []
            if data.get("title"):
                parts.append(f"TITLE: {data['title']}")
            if data.get("h1"):
                parts.append(f"H1: {data['h1']}")
            if data.get("first_paragraph"):
                parts.append(f"FIRST PARAGRAPH: {data['first_paragraph']}")
            extracted.setPlainText("\n\n".join(parts))
            extracted.setMaximumHeight(150)
            section = ExpandableSection("Извлечённый контент", extracted)
            self.results_layout.addWidget(section)
        
        # Анализ
        if data.get("analysis"):
            self.display_text_results(data["analysis"])

    def load_history(self):
        try:
            response = requests.get(f"{API_BASE}/history", timeout=10)
            response.raise_for_status()
            data = response.json()
            items = data.get("items", [])
            
            self.history_list.clear()
            for item in reversed(items):  # Новые сверху
                type_label = {"text": "TEXT", "image": "IMAGE", "parse_demo": "PARSE"}.get(item["type"], "UNKNOWN")
                preview = ""
                if item.get("payload"):
                    payload = item["payload"]
                    preview = payload.get("input_preview") or payload.get("url") or payload.get("filename") or ""
                
                timestamp = item.get("timestamp", "")
                try:
                    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    time_str = dt.strftime("%d.%m %H:%M")
                except:
                    time_str = timestamp[:16] if timestamp else ""
                
                list_item = QListWidgetItem(f"[{type_label}] {preview[:50]}... ({time_str})")
                list_item.setData(Qt.ItemDataRole.UserRole, item)
                self.history_list.addItem(list_item)
        except Exception as e:
            print(f"Ошибка загрузки истории: {e}")

    def load_history_item(self, item: QListWidgetItem):
        history_item = item.data(Qt.ItemDataRole.UserRole)
        if not history_item:
            return
        
        item_type = history_item.get("type")
        payload = history_item.get("payload", {})
        analysis = payload.get("analysis", {})
        
        self.clear_results()
        
        if item_type == "parse_demo":
            # Показываем извлечённый контент
            if payload.get("extracted"):
                extracted = QTextEdit()
                extracted.setReadOnly(True)
                extracted.setStyleSheet("background: transparent; border: none; font-size: 11px;")
                parts = []
                if payload["extracted"].get("title"):
                    parts.append(f"TITLE: {payload['extracted']['title']}")
                if payload["extracted"].get("h1"):
                    parts.append(f"H1: {payload['extracted']['h1']}")
                if payload["extracted"].get("first_paragraph"):
                    parts.append(f"FIRST PARAGRAPH: {payload['extracted']['first_paragraph']}")
                if parts:
                    extracted.setPlainText("\n\n".join(parts))
                    extracted.setMaximumHeight(150)
                    section = ExpandableSection("Извлечённый контент", extracted)
                    self.results_layout.addWidget(section)
            self.display_text_results(analysis)
        elif item_type == "image":
            self.display_image_results(analysis)
        else:
            self.display_text_results(analysis)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = CompetitorMonitorApp()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
