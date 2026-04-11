import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QTextEdit
from spikeNet import detspike
from sleep import sleep

class FileUploader(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Edf File Uploader')
        self.setGeometry(100, 100, 800, 900)

        self.layout = QVBoxLayout()

        self.upload_button = QPushButton('load Edf File for spike')
        self.upload_button.clicked.connect(self.open_file)

        self.upload_button_sleep = QPushButton('load Edf File for sleep')
        self.upload_button_sleep.clicked.connect(self.open_file_sleep)

        # 初始化 text_edit 并添加到布局中
        self.text_edit = QTextEdit(self)
        self.text_edit.setReadOnly(True)
        self.layout.addWidget(self.text_edit)

        self.layout.addWidget(self.upload_button)
        self.layout.addWidget(self.upload_button_sleep)

        self.setLayout(self.layout)

    def open_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Edf File", "", "Edf Files (*.edf)", options=options)
        if file_path:
            print(f"Selected file: {file_path}")
            self.process_file(file_path)

    def open_file_sleep(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Edf File", "", "Edf Files (*.edf)", options=options)
        if file_path:
            print(f"Selected file: {file_path}")
            self.process_file_sleep(file_path) 

    def process_file(self, file_path):
        # 处理文件的逻辑
        try:
            print(f"Processing file: {file_path}")
            output_text = detspike(file_path)
            self.text_edit.append(output_text)
        except Exception as e:
            error_message = f"Error while processing file with detspike: {str(e)}"
            print(error_message)  # 打印到控制台
            self.text_edit.append(error_message)  # 显示在界面

    def process_file_sleep(self, file_path):
        # 处理文件的逻辑
        try:
            print(f"Processing file: {file_path}")
            output_text = sleep(file_path)
            self.text_edit.append(output_text)
        except Exception as e:
            error_message = f"Error while processing file with sleep: {str(e)}"
            print(error_message)  # 打印到控制台
            self.text_edit.append(error_message)  # 显示在界面


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FileUploader()
    window.show()
    sys.exit(app.exec_())