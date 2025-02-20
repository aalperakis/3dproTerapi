from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, QScrollArea,
                             QGridLayout, QTextEdit)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from Analyzer import PostureAnalyzer
import os
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

def plot_angles(result_img: Image, angles, position):
    height = result_img.size[1]
    draw = ImageDraw.Draw(result_img)
    
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", height // 70)
    except:
        font = ImageFont.load_default()

    for item in angles:
        x, y = item["coord"]
        label = item["name"]
        angle = item["angle"]

        label_with_angle = label + " : " + str(angle) + "°"

        if position != "back":
            if "Sol" in label:
                x_, y_ = x + (x * 1/5), y - (y * 1/20)
            else:
                x_, y_ = x - (x * 1/2), y - (y * 1/20)
        else:
            if "Sol" in label:
                x_, y_ = x - (x * 1/2), y - (y * 1/20)
            else:
                x_, y_ = x + (x * 1/5), y - (y * 1/20)

        bbox = draw.textbbox((0, 0), label_with_angle, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        padding = height // 180
        corner_radius = height // 170
        background_color = (255, 255, 255)
        border_color = (0, 128, 0)

        width = height // 450
        if position == "front":
            if "Sol" in label:
                draw.line([(x, y), (x_, y_)], fill=(255, 255, 255), width=width)
            else:
                x_ = x_ - text_width//2
                draw.line([(x, y), (x_ + text_width, y_)], fill=(255, 255, 255), width=width)
            label_with_angle = " ".join(label_with_angle.split(" ")[1:])
        elif position == "back":
            if "Sol" in label:
                x_ = x_ - text_width//2
                draw.line([(x, y), (x_ + text_width, y_)], fill=(255, 255, 255), width=width)
            else:
                draw.line([(x, y), (x_, y_)], fill=(255, 255, 255), width=width)
            label_with_angle = " ".join(label_with_angle.split(" ")[1:])
        else:
            if "Sol" in label:
                draw.line([(x, y), (x_, y_)], fill=(255, 255, 255), width=width)
            else:
                x_ = x_ - text_width//2
                draw.line([(x, y), (x_ + text_width, y_)], fill=(255, 255, 255), width=width)

            if "Kafa" in label:
                label_with_angle = " ".join(label_with_angle.split(" ")[1:])

        draw.rounded_rectangle(
            [x_ - padding, y_ - text_height - padding,
             x_ + text_width + padding, y_ + padding],
            radius=corner_radius,
            fill=background_color,
            outline=border_color,
            width=height // 800
        )
        draw.text((x_, y_ - text_height), label_with_angle, font=font, fill=(0, 0, 0))

    return result_img

class PostureAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Duruş Analizi")
        self.setGeometry(100, 100, 1000, 800)
        
        # Ana widget ve scroll area
        scroll = QScrollArea()
        self.setCentralWidget(scroll)
        scroll.setWidgetResizable(True)
        
        # İçerik widget'ı
        content_widget = QWidget()
        scroll.setWidget(content_widget)
        main_layout = QVBoxLayout(content_widget)
        base_path = os.path.dirname(os.path.abspath(__file__))
        logo_path = os.path.join(base_path, 'static', 'logo.jpg')
        # Logo
        logo_label = QLabel()
        logo_pixmap = QPixmap(logo_path)
        logo_label.setPixmap(logo_pixmap.scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        logo_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(logo_label)
        
        # Görüntü seçme butonları
        btn_layout = QHBoxLayout()
        self.load_buttons = {}
        gorunum_isimleri = {
            'front': 'Ön Görüntü',
            'back': 'Arka Görüntü',
            'left': 'Sol Yan Görüntü',
            'right': 'Sağ Yan Görüntü'
        }
        for view, isim in gorunum_isimleri.items():
            btn = QPushButton(f'{isim} Yükle')
            btn.clicked.connect(lambda checked, v=view: self.load_image(v))
            btn_layout.addWidget(btn)
            self.load_buttons[view] = btn
        main_layout.addLayout(btn_layout)
        
        # Analiz butonu
        self.analyze_btn = QPushButton('Analiz Et')
        self.analyze_btn.clicked.connect(self.analyze_images)
        self.analyze_btn.setEnabled(False)
        main_layout.addWidget(self.analyze_btn)
        
        # Grid layout for images (2x2)
        image_widget = QWidget()
        self.image_layout = QGridLayout(image_widget)
        main_layout.addWidget(image_widget)
        
        # Store loaded images and their labels
        self.images = {}
        self.image_labels = {}
        positions = {
            'front': (0, 0),
            'back': (0, 1),
            'left': (1, 0),
            'right': (1, 1)
        }
        
        for view, pos in positions.items():
            label = QLabel()
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumSize(400, 600)
            self.image_labels[view] = label
            self.image_layout.addWidget(label, *pos)
        
        # Text area for results
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMinimumHeight(150)
        main_layout.addWidget(self.results_text)
        
        # Save PDF button
        self.save_pdf_btn = QPushButton('Analiz PDF kaydet')
        self.save_pdf_btn.clicked.connect(self.save_pdf)
        self.save_pdf_btn.setEnabled(False)
        main_layout.addWidget(self.save_pdf_btn)
        
        self.analyzer = PostureAnalyzer()
        self.analysis_results = {}
        
    def load_image(self, view):
        file_name, _ = QFileDialog.getOpenFileName(self, f'Select {view} image', '', 
                                                 'Image Files (*.bmp)')
        if file_name:
            pixmap = QPixmap(file_name)
            scaled_pixmap = pixmap.scaled(400, 600, Qt.KeepAspectRatio)
            self.image_labels[view].setPixmap(scaled_pixmap)
            self.images[view] = file_name
            
            if len(self.images) == 4:
                self.analyze_btn.setEnabled(True)
    
    def analyze_images(self):
        self.results_text.clear()
        self.analysis_results.clear()
        
        gorunum_isimleri = {
            'front': 'ÖN',
            'back': 'ARKA',
            'left': 'SOL YAN',
            'right': 'SAĞ YAN'
        }
        
        for view, image_path in self.images.items():
            try:
                img = Image.open(image_path).convert("RGB")
                img_np = np.array(img)
                
                result_img, angles = self.analyzer.analyze(image_path, img_np, view)
                result_img = plot_angles(result_img, angles, view)
                
                self.analysis_results[view] = {
                    'image': result_img,
                    'angles': angles
                }
                
                result_array = np.array(result_img)
                height, width, channel = result_array.shape
                bytes_per_line = 3 * width
                q_img = QImage(result_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                scaled_pixmap = pixmap.scaled(400, 600, Qt.KeepAspectRatio)
                self.image_labels[view].setPixmap(scaled_pixmap)
                
                # Sonuçları Türkçe göster
                self.results_text.append(f"\n{gorunum_isimleri[view]} görüntü sonuçları:")
                for angle_data in angles:
                    self.results_text.append(f"  {angle_data['name']}: {angle_data['angle']}°")
                
            except Exception as e:
                self.results_text.append(f"{gorunum_isimleri[view]} görüntü işlenirken hata oluştu: {str(e)}")
        
        self.save_pdf_btn.setEnabled(True)
    
    def save_pdf(self):
        try:
            # Türkçe karakterler için font tanımlama
            pdfmetrics.registerFont(TTFont('Arial-Turkish', 'arial.ttf'))
            
            zaman_damgasi = datetime.now().strftime("%Y%m%d_%H%M%S")
            varsayilan_isim = f'duruş_analizi_{zaman_damgasi}.pdf'
            
            # Kullanıcıya kaydetme yerini sor
            pdf_yolu, _ = QFileDialog.getSaveFileName(
                self,
                'PDF Kaydet',
                varsayilan_isim,
                'PDF Dosyaları (*.pdf)'
            )
            
            if not pdf_yolu:  # Kullanıcı iptal ettiyse
                return
            
            c = canvas.Canvas(pdf_yolu, pagesize=A4)
            genislik, yukseklik = A4  # A4: (595.27, 841.89) points
            
            base_path = os.path.dirname(os.path.abspath(__file__))
            logo_path = os.path.join(base_path, 'static', 'logo.jpg')
            # Logo'yu üst köşeye ekle
            logo_genislik = 100
            logo_yukseklik = 100
            c.drawImage(logo_path, 50, yukseklik - 110, width=logo_genislik, height=logo_yukseklik, preserveAspectRatio=True)
            
            # Başlık metni - logo'nun yanına
            c.setFont("Arial-Turkish", 9)
            baslik_metni = "3D Pro Terapi; Yapay zeka desteği ile omurga ve ayak analizi yaparak,"
            c.drawString(160, yukseklik - 40, baslik_metni)
            alt_baslik = "posturunu iyileştirmeye ve ardından kişiye özel egzersiz planları sunarak"
            c.drawString(160, yukseklik - 55, alt_baslik)
            son_metin = "fiziksel sağlığını desteklemeye yardımcı olan yenilikçi bir teknoloji çözümüdür."
            c.drawString(160, yukseklik - 70, son_metin)

            # Web sitesini kalın yazı tipiyle yaz
            c.setFont("Arial-Turkish", 9)  # Bold font yerine normal font kullanıyoruz
            web_sitesi = "www.3dproterapi.com.tr"
            c.drawString(160, yukseklik - 85, web_sitesi)
            
            # Görüntüleri 2x2 grid şeklinde yerleştir
            goruntu_genislik = 250
            goruntu_yukseklik = 350
            kenar_bosluk_x = 50
            kenar_bosluk_y = 100  # Üst kenar boşluğu logo için
            bosluk = 5
            
            konumlar = {
                'front': (kenar_bosluk_x, yukseklik - kenar_bosluk_y - goruntu_yukseklik),
                'back': (kenar_bosluk_x + goruntu_genislik + bosluk, yukseklik - kenar_bosluk_y - goruntu_yukseklik),
                'left': (kenar_bosluk_x, yukseklik - kenar_bosluk_y - 2*goruntu_yukseklik - bosluk),
                'right': (kenar_bosluk_x + goruntu_genislik + bosluk, yukseklik - kenar_bosluk_y - 2*goruntu_yukseklik - bosluk)
            }
            
            # Görüntüleri yerleştir
            for gorunum, veri in self.analysis_results.items():
                gecici_goruntu_yolu = f'gecici_{gorunum}.jpg'
                veri['image'].save(gecici_goruntu_yolu)
                x, y = konumlar[gorunum]
                c.drawImage(gecici_goruntu_yolu, x, y, width=goruntu_genislik, height=goruntu_yukseklik, preserveAspectRatio=True)
                os.remove(gecici_goruntu_yolu)
            
            c.save()
            self.results_text.append(f"\nAnaliz kaydedildi: {pdf_yolu}")
            
        except Exception as e:
            self.results_text.append(f"PDF kaydedilirken hata oluştu: {str(e)}")

def main():
    app = QApplication(sys.argv)
    window = PostureAnalysisApp()
    window.show()
    app.exec_()
    app.quit()  
    exit_code = app.exec_()  # Uygulama kapanana kadar çalıştır
    del window  # Pencereyi bellekten temizle
    sys.exit(exit_code)
if __name__ == '__main__':
    main()