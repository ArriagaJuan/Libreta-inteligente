import os
import datetime
import re
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_CENTER

def convertir_markdown_a_reportlab(texto):
    """
    Convierte texto con formato Markdown a formato compatible con ReportLab.
    Maneja negritas, cursivas, bloques de código, listas y encabezados.
    """
    if not texto:
        return ""
        
    # Convertir bloques de código
    texto = re.sub(r'```(.*?)```', r'<pre>\1</pre>', texto, flags=re.DOTALL)
    
    # Convertir código en línea
    texto = re.sub(r'`([^`]+)`', r'<code>\1</code>', texto)
    
    # Convertir negritas
    texto = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', texto)
    
    # Convertir cursivas
    texto = re.sub(r'\*([^*]+)\*', r'<i>\1</i>', texto)
    
    # Convertir encabezados
    texto = re.sub(r'^# (.*?)$', r'<h1>\1</h1>', texto, flags=re.MULTILINE)
    texto = re.sub(r'^## (.*?)$', r'<h2>\1</h2>', texto, flags=re.MULTILINE)
    texto = re.sub(r'^### (.*?)$', r'<h3>\1</h3>', texto, flags=re.MULTILINE)
    
    # Convertir listas
    texto = re.sub(r'^\* (.*?)$', r'• \1', texto, flags=re.MULTILINE)
    texto = re.sub(r'^- (.*?)$', r'• \1', texto, flags=re.MULTILINE)
    texto = re.sub(r'^(\d+)\. (.*?)$', r'\1. \2', texto, flags=re.MULTILINE)
    
    return texto

def guardar_conversacion_pdf_mejorado(chat_historial, nombre_base="conversacion"):
    """
    Genera un PDF mejorado de la conversación con formato.
    
    Args:
        chat_historial (list): Lista de tuplas (mensaje_usuario, mensaje_asistente)
        nombre_base (str): Prefijo para el nombre del archivo PDF
    
    Returns:
        str: Ruta al archivo PDF creado
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    nombre = f"{nombre_base}_{timestamp}.pdf"
    
    # Configurar el documento
    doc = SimpleDocTemplate(
        nombre,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18
    )
    
    # Definir estilos
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Titulo', fontName='Helvetica-Bold', fontSize=16, alignment=TA_CENTER, spaceAfter=12))
    styles.add(ParagraphStyle(name='Usuario', fontName='Helvetica-Bold', fontSize=11, textColor=colors.blue, spaceAfter=6))
    styles.add(ParagraphStyle(name='Asistente', fontName='Helvetica', fontSize=11, spaceAfter=12))
    styles.add(ParagraphStyle(name='Codigo', fontName='Courier', fontSize=9, backColor=colors.lightgrey, borderWidth=1, borderColor=colors.grey, borderPadding=5))
    
    # Elementos del documento
    elementos = []
    
    # Título
    elementos.append(Paragraph("Conversación con Asistente Educativo", styles['Titulo']))
    elementos.append(Spacer(1, 0.25*inch))
    
    # Añadir mensajes
    for usuario, asistente in chat_historial:
        # Mensaje del usuario
        elementos.append(Paragraph("<b>Usuario:</b>", styles['Usuario']))
        texto_usuario = convertir_markdown_a_reportlab(usuario)
        elementos.append(Paragraph(texto_usuario, styles['Normal']))
        elementos.append(Spacer(1, 0.1*inch))
        
        # Mensaje del asistente
        elementos.append(Paragraph("<b>Asistente:</b>", styles['Asistente']))
        texto_asistente = convertir_markdown_a_reportlab(asistente)
        elementos.append(Paragraph(texto_asistente, styles['Normal']))
        elementos.append(Spacer(1, 0.2*inch))
    
    # Añadir pie de página
    elementos.append(Spacer(1, 0.5*inch))
    fecha_generacion = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    elementos.append(Paragraph(f"<i>Generado: {fecha_generacion}</i>", styles['Normal']))
    
    # Construir el documento
    doc.build(elementos)
    
    return nombre
