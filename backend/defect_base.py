import datetime
import pickle
import sqlite3
import cv2
from backend import defects
from reportlab.pdfgen import canvas
from reportlab.graphics import renderPDF
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.utils import ImageReader
from svglib.svglib import svg2rlg
import io


def add_text_to_pdf_center(canvas, text, y):
    width, height = A4
    text_width = stringWidth(text, fontName='GOST', fontSize=14)
    pdf_text_object = canvas.beginText((width - text_width) / 2.0, y)
    pdf_text_object.textOut(text)


def add_text_to_pdf_left(canvas, text, x, y):
    pdf_text_object = canvas.beginText(x, y)
    pdf_text_object.textOut(text)


def footer(canvas, date, width, page):
    canvas.drawCentredString(width / 2, 40, '{}.{}.{}'.format(date.day,
                                                              date.month,
                                                              date.year))
    canvas.drawCentredString(width / 2, 60, 'ООО СК-"Роботикс"')
    canvas.drawCentredString(width / 2, 20, 'стр. {}'.format(page))
    canvas.line(width / 2 - 150, 53, width / 2 + 150, 53)


class DefectsBase:
    def __init__(self, path_to_base='defects_base.db'):
        try:
            self.db_connection =sqlite3.connect(path_to_base)
            self.cursor =self.db_connection.cursor()
            sql_req = 'CREATE TABLE IF NOT EXISTS airplanes (id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT DEFAULT 0,' \
                      'name TEXT NOT NULL,' \
                      'serial TEXT NOT NULL,' \
                      'comment TEXT DEFAULT "",' \
                      'UNIQUE (name, serial) ON CONFLICT  IGNORE);'
            self.cursor.execute(sql_req)
            sql_req = 'CREATE TABLE IF NOT EXISTS defects (id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT DEFAULT 0,' \
                      'airplane_name TEXT NOT NULL,' \
                      'air_plane_serial TEXT NOT NULL,' \
                      'date TIMESTAMP NOT NULL,' \
                      'defect_data BLOB,' \
                      'comment TEXT DEFAULT "");'
            self.cursor.execute(sql_req)
            self.db_connection.commit()
            self.cursor.close()
        except sqlite3.Error as e:
            print('Error opening/creating database', e)

    def add(self, aircraft_defects_list: defects.defect_detector):
        self.cursor = self.db_connection.cursor()
        sql_req = 'INSERT OR IGNORE INTO airplanes (name, serial) VALUES (?,?);'
        self.cursor.execute(sql_req, (aircraft_defects_list.name, aircraft_defects_list.serial_num))

        date_now = datetime.datetime.now()
        for defect in aircraft_defects_list.defects:
            sql_req = 'INSERT INTO defects (airplane_name, air_plane_serial, date, defect_data, comment) ' \
                      'VALUES (?, ?, ?, ?, ?);'
            self.cursor.execute(sql_req, (aircraft_defects_list.name,
                                          aircraft_defects_list.serial_num,
                                          aircraft_defects_list.date,
                                          pickle.dumps(defect),
                                          ''))
        self.db_connection.commit()
        self.cursor.close()

    def get(self, aircraft_name, aircraft_serial):
        self.cursor = self.db_connection.cursor()
        sql_req = 'SELECT MAX(date) FROM defects WHERE airplane_name=? AND air_plane_serial=?;'
        self.cursor.execute(sql_req, (aircraft_name, aircraft_serial))
        date, = self.cursor.fetchone()

        sql_req = 'SELECT defect_data FROM defects WHERE airplane_name=? AND air_plane_serial=? AND date=?;'
        self.cursor.execute(sql_req, (aircraft_name, aircraft_serial, date))
        defects_resp = self.cursor.fetchall()
        date = datetime.datetime.strptime(date.split('.')[0], '%Y-%m-%d %H:%M:%S')

        air_craft = defects.AirCraftDefectsList(aircraft_serial, aircraft_name)
        air_craft.date = date
        for defect_resp in defects_resp:
            defect_pickled, = defect_resp
            defect = pickle.loads(defect_pickled)
            air_craft.defects.append(defect)

        return air_craft

    def report(self, aircraft_name, aircraft_serial):
        air_craft = self.get(aircraft_name, aircraft_serial)
        pdfmetrics.registerFont(TTFont('GOST', 'backend\\GOSTtypeB.ttf'))

        pdf_canvas = canvas.Canvas('air_craft_{}_serial_{}.pdf'.format(aircraft_name, aircraft_serial), pagesize=A4)
        width, height = A4
        pdf_canvas.setLineWidth(3)
        pdf_canvas.setFont('GOST', size=14)

        report_day = air_craft.date
        page_num = 1
        footer(pdf_canvas, report_day, width, page_num)

        pdf_canvas.drawCentredString(width / 2, 755, 'Отчет по ВС {}'.format(air_craft.name))
        pdf_canvas.drawCentredString(width / 2, 740, 'серийный номер {}'.format(air_craft.serial_num))
        pdf_canvas.line(width / 2 - 240, 732, width / 2 + 240, 732)
        pdf_canvas.line(width / 2 - 230, 730, width / 2 + 230, 730)

        total_defects = [len(defect.types) for defect in air_craft.defects]
        count_defects = sum(total_defects)

        pdf_canvas.drawString(75, 650, 'Обнаружено дефектов: {} на {} кадрах'.format(count_defects, len(total_defects)))
        pdf_canvas.showPage()

        for defect in air_craft.defects:
            pdf_canvas.setFont('GOST', size=14)
            cv2.imwrite('temp.jpg', cv2.resize(defect.image, (int(width)-100, int(3*height/4))))
            pic = ImageReader('temp.jpg')
            pdf_canvas.drawImage(pic, 40, 100)
            page_num += 1
            footer(pdf_canvas, report_day, width, page_num)
            pdf_canvas.showPage()
        pdf_canvas.save()

    def close(self):
        self.db_connection.close()
