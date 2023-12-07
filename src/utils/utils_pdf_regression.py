from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, PageBreak, Image, Paragraph, Spacer, TableStyle
import math 

from reportlab.lib.pagesizes import letter
import os

class ManagePDF_R():    
    
    def __init__(self, path, imagespath, erase_image=True) -> None:
        self.elements = []
        self.imagespath = imagespath
        self.total_pages = self.calculate_num_pages(len(self.imagespath))
        self.delete_image = erase_image
        # pdf configuration
        self.styles = getSampleStyleSheet()
        self.doc = SimpleDocTemplate(path, rightMargin=0, leftMargin=0, topMargin=0, bottomMargin=0, pagesize=letter)
        self.page_width, self.page_height = letter
        # self.paragrath_style = ParagraphStyle()
        
    def calculate_num_pages(self, len_list):
        return math.ceil(len_list / 2) # Four images per page

    def createPageHeader(self):
        self.elements.append(Spacer(1, 8)) #* original - Spacer(1, 10)
        # self.elements.append(Image('car_logo.png', 100, 25)) #* logo - if necessary
        self.elements.append(Paragraph("Relatory/Results in this Epoch", self.styles['Title']))
        self.elements.append(Spacer(1, 8))

    def insert_element(self, element):
        if ".png" in element:
            # Add the image to the PDF document
            image_width = int(self.page_width * 0.8)
            image_height = int(self.page_height * 0.65)
            image = Image(element, width=image_width, height=image_height)
            self.elements.append(image)
            
            if self.delete_image:
                os.remove(element) # delete image from source
        else:
            # Add text to the PDF document
            # style= Paragraph(element, style=self.styles['BodyText']))
            # self.elements.append(Paragraph(element, style=self.styles['BodyText']))
            # self.set_table(element)
            pass

    def generatePDF(self):
        """AI is creating summary for generatePDF
        """
        for index in range(len(self.imagespath)):
            new_page = (((index + 1) % 2 == 0) and (index != 0))# create a new page each tow elements (one image and one table).
            new_header = (index % 2 == 0) # create a header every new page.
            
            if new_header: self.createPageHeader()
            
            self.insert_element(self.imagespath[index])
            
            if new_page: self.elements.append(PageBreak())
            
        self.doc.build(self.elements)
    
    # def set_table(self, classification_relatory):
    #     """AI is creating summary for set_table

    #     Args:
    #         classification_relatory ([type]): [description]
    #     """
        
    #     tbl = Table(classification_relatory)
    #     tbl.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), '#F5F5F5'),
    #                             ('FONTSIZE', (0, 0), (-1, 0), 8),
    #                             ('GRID', (0, 0), (-1, -1), .5, '#a7a5a5')])) 
    #     self.elements.append(tbl)

    def save_pdf(self):
        """AI is creating summary for save_pdf
        """
        self.generatePDF()