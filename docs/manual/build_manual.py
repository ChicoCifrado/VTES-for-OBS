#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Genera el manual de usuario de VTES Card Scanner (PDF) con fpdf2.

Uso:  python3 build_manual.py
Salida: VTES-for-OBS_Manual_v0.1.7.pdf (en esta misma carpeta)

Requiere: fpdf2 y Pillow. Las imágenes se generan a partir del diseño real
del overlay (data/obs-plugins/vtes-card-scanner/overlay-bg.png).
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from fpdf import FPDF
from fpdf.enums import XPos, YPos

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
ASSETS = BASE / "assets"
OUT = BASE / "VTES-for-OBS_Manual_v0.1.7.pdf"

OVERLAY_PNG = ROOT / "data" / "obs-plugins" / "vtes-card-scanner" / "overlay-bg.png"
FONT_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_MONO_B = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"

# ── Paleta (la misma de la web del plugin) ────────────────────────────────
GOLD = (201, 168, 76)
DARKRED = (140, 40, 40)
RED = (122, 31, 31)
NAVY = (26, 26, 46)
NAVY2 = (13, 27, 62)
TXT = (45, 45, 55)
GREY = (120, 120, 130)
LIGHTLINE = (205, 200, 190)
ACCENT_BG = (250, 246, 232)
WHITISH = (235, 235, 240)

VERSION = "v0.1.7"
DATE = "Octubre de 2026"


# ── Assets (se generan desde el diseño real) ──────────────────────────────
def make_assets() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    design = Image.open(OVERLAY_PNG).convert("RGBA")

    # 1) Ankh recortado para la portada (con bordes difuminados para que
    #    se funda con el fondo de la cubierta).
    ankh = design.crop((20, 470, 242, 975))
    fade_w = int(ankh.width * 0.16)
    fade_h = int(ankh.height * 0.14)
    mask = Image.new("L", ankh.size, 255)
    px = mask.load()
    for yy in range(ankh.height):
        for xx in range(ankh.width):
            fx = min(xx, ankh.width - 1 - xx) / fade_w
            fy = min(yy, ankh.height - 1 - yy) / fade_h
            f = min(1.0, fx, fy)
            f = max(0.0, min(1.0, f))
            px[xx, yy] = int(255 * (f * f * (3 - 2 * f)))
    ankh.putalpha(mask)
    # JPEG (compuesto sobre el azul de la portada): mantiene los PDF por
    # debajo de los límites de subida habituales (0,5 MB) sin pérdida
    # visible de calidad.
    ankh_flat = Image.new("RGB", ankh.size, NAVY)
    ankh_flat.paste(ankh, mask=ankh.split()[3])
    ankh_flat.save(ASSETS / "ankh.jpg", quality=88, optimize=True,
                   progressive=True)

    # 2) Diagrama anotado del overlay.
    canvas = Image.new("RGBA", design.size, (14, 14, 22, 255))
    canvas.alpha_composite(design)
    img = canvas.convert("RGB")
    d = ImageDraw.Draw(img, "RGBA")
    f_lab = ImageFont.truetype(FONT_BOLD, 30)
    f_sub = ImageFont.truetype(FONT_REG, 24)
    f_big = ImageFont.truetype(FONT_BOLD, 46)

    def label_box(x0, y0, x1, y1, text, font):
        d.rounded_rectangle((x0, y0, x1, y1), radius=14, fill=(10, 10, 18, 210),
                            outline=(201, 168, 76, 255), width=3)
        d.text(((x0 + x1) / 2, (y0 + y1) / 2), text, font=font, anchor="mm",
               fill=(238, 232, 210), stroke_width=3, stroke_fill=(0, 0, 0))

    def arrow(x0, y0, x1, y1):
        import math
        d.line((x0, y0, x1, y1), fill=(201, 168, 76, 255), width=5)
        ang = math.atan2(y1 - y0, x1 - x0)
        L = 22
        for s in (-0.45, 0.45):
            d.line((x1, y1, x1 - L * math.cos(ang + s), y1 - L * math.sin(ang + s)),
                   fill=(201, 168, 76, 255), width=5)

    # Ventana de cámara (289,41)-(1492,988): borde discontinuo.
    x0, y0, x1, y1 = 289, 41, 1492, 988
    dash, gap = 22, 14
    x = x0
    while x < x1:
        d.line((x, y0, min(x + dash, x1), y0), fill=(201, 168, 76, 255), width=3)
        d.line((x, y1, min(x + dash, x1), y1), fill=(201, 168, 76, 255), width=3)
        x += dash + gap
    y = y0
    while y < y1:
        d.line((x0, y, x0, min(y + dash, y1)), fill=(201, 168, 76, 255), width=3)
        d.line((x1, y, x1, min(y + dash, y1)), fill=(201, 168, 76, 255), width=3)
        y += dash + gap

    d.text((890, 470), "VENTANA DE CÁMARA", font=f_big, anchor="mm",
           fill=(238, 232, 210), stroke_width=4, stroke_fill=(0, 0, 0))
    d.text((890, 522), "tu captura se ajusta aquí, entera y sin recortar",
           font=f_sub, anchor="mm", fill=(205, 196, 170), stroke_width=3,
           stroke_fill=(0, 0, 0))

    label_box(300, 58, 1060, 130, "PANEL DE JUGADORES — nombre y pool", f_lab)
    arrow(300, 96, 252, 106)
    label_box(300, 636, 1090, 708, "ANKH — última carta mostrada", f_lab)
    arrow(300, 672, 246, 704)

    img.save(ASSETS / "overlay_diagram.jpg", quality=88, optimize=True,
             progressive=True)
    print("Assets generados:", ASSETS)


# ── Documento ─────────────────────────────────────────────────────────────
class Manual(FPDF):
    def __init__(self):
        super().__init__(unit="mm", format="A4")
        self.set_margins(18, 20, 18)
        self.set_auto_page_break(auto=True, margin=22)
        fam = "DejaVu"
        self.add_font(fam, "", FONT_REG)
        self.add_font(fam, "B", FONT_BOLD)
        self.add_font("Mono", "", FONT_MONO)
        self.add_font("Mono", "B", FONT_MONO_B)
        self.set_font(fam, "", 10.5)

    # -- infraestructura ---------------------------------------------------
    def footer(self):
        if self.page_no() == 1:
            return
        self.set_draw_color(*GOLD)
        self.set_line_width(0.25)
        self.line(18, 283, 192, 283)
        self.set_y(-22)
        self.set_font("DejaVu", "", 8)
        self.set_text_color(*GREY)
        self.cell(120, 6, f"VTES Card Scanner · Manual de uso {VERSION}",
                  new_x=XPos.RIGHT, new_y=YPos.TOP)
        self.cell(0, 6, f"Página {self.page_no()}", align="R",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def space(self, h):
        self.ln(h)

    def ensure(self, h):
        if self.h - self.b_margin - self.get_y() < h:
            self.add_page()

    def h1(self, title):
        self.ensure(58)
        y = self.get_y()
        self.space(3)
        y = self.get_y()
        self.set_fill_color(*GOLD)
        self.rect(self.l_margin, y + 1.2, 3.2, 10.5, style="F")
        self.set_xy(self.l_margin + 6, y - 1.6)
        self.set_font("DejaVu", "B", 18)
        self.set_text_color(*NAVY)
        self.cell(0, 13, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*LIGHTLINE)
        self.set_line_width(0.4)
        self.line(self.l_margin, self.get_y() - 1.5, 192, self.get_y() - 1.5)
        self.space(4.5)

    def h2(self, title):
        self.ensure(26)
        self.space(1.5)
        self.set_font("DejaVu", "B", 13)
        self.set_text_color(*DARKRED)
        self.cell(0, 8, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.space(1.2)

    def p(self, text, size=10.5, color=TXT, lh=5.15):
        self.set_font("DejaVu", "", size)
        self.set_text_color(*color)
        self.multi_cell(0, lh, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.space(1.4)

    def bullets(self, items, size=10.3):
        for it in items:
            self.ensure(10)
            y = self.get_y()
            self.set_xy(self.l_margin, y)
            self.set_font("DejaVu", "B", 11)
            self.set_text_color(*GOLD)
            self.cell(4.8, 5.0, "•", new_x=XPos.RIGHT, new_y=YPos.TOP)
            self.set_font("DejaVu", "", size)
            self.set_text_color(*TXT)
            self.multi_cell(174 - 4.8, 5.0, it, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.space(0.7)

    def steps(self, items, size=10.3):
        for i, it in enumerate(items, start=1):
            self.ensure(10)
            self.set_xy(self.l_margin, self.get_y())
            self.set_font("DejaVu", "B", 10.5)
            self.set_text_color(*RED)
            self.cell(6.4, 5.0, f"{i}.", new_x=XPos.RIGHT, new_y=YPos.TOP)
            self.set_font("DejaVu", "", size)
            self.set_text_color(*TXT)
            self.multi_cell(174 - 6.4, 5.0, it, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.space(0.7)

    def defitem(self, term, desc, dsize=9.9):
        self.ensure(16)
        self.set_font("DejaVu", "B", 10.3)
        self.set_text_color(*NAVY2)
        self.multi_cell(0, 5.2, term, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        y0 = self.get_y()
        self.set_xy(self.l_margin + 4.5, y0)
        self.set_font("DejaVu", "", dsize)
        self.set_text_color(*TXT)
        self.multi_cell(174 - 4.5, 4.9, desc, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        y1 = self.get_y()
        self.set_draw_color(*LIGHTLINE)
        self.set_line_width(0.35)
        self.line(self.l_margin + 1.6, y0 - 0.5, self.l_margin + 1.6, y1 - 1.6)
        self.space(1.6)

    def callout(self, title, text, accent=GOLD, fill=ACCENT_BG):
        w = 174
        pad = 4.5
        self.set_font("DejaVu", "B", 10.4)
        th = self.multi_cell(w - 2 * pad - 2.5, 5.2, title, dry_run=True, output="HEIGHT")
        self.set_font("DejaVu", "", 9.7)
        bh = self.multi_cell(w - 2 * pad - 2.5, 4.8, text, dry_run=True, output="HEIGHT")
        h = th + bh + 2 * pad + 2.0
        self.ensure(h + 3)
        y = self.get_y()
        self.set_fill_color(*fill)
        self.rect(self.l_margin, y, w, h, style="F")
        self.set_fill_color(*accent)
        self.rect(self.l_margin, y, 2.2, h, style="F")
        self.set_xy(self.l_margin + pad + 1.5, y + pad - 1.4)
        self.set_font("DejaVu", "B", 10.4)
        self.set_text_color(*DARKRED)
        self.multi_cell(w - 2 * pad - 2.5, 5.2, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_xy(self.l_margin + pad + 1.5, self.get_y() - 0.6)
        self.set_font("DejaVu", "", 9.7)
        self.set_text_color(*TXT)
        self.multi_cell(w - 2 * pad - 2.5, 4.8, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_y(y + h)
        self.space(3.2)

    def mono_block(self, lines):
        self.ensure(8 + 5.6 * len(lines))
        y = self.get_y()
        h = 5.6 * len(lines) + 3.2
        self.set_fill_color(244, 242, 236)
        self.rect(self.l_margin, y, 174, h, style="F")
        self.set_fill_color(*GOLD)
        self.rect(self.l_margin, y, 1.6, h, style="F")
        self.set_xy(self.l_margin + 4, y + 1.4)
        self.set_font("Mono", "", 9.2)
        self.set_text_color(60, 55, 45)
        for ln in lines:
            self.cell(0, 5.6, ln, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_y(y + h)
        self.space(3)

    def image_block(self, path, w_mm, caption=None):
        im = Image.open(path)
        h_mm = w_mm * im.height / im.width
        self.ensure(h_mm + (10 if caption else 4))
        x = (210 - w_mm) / 2
        self.image(str(path), x=x, y=self.get_y(), w=w_mm)
        self.set_y(self.get_y() + h_mm + 1.6)
        if caption:
            self.set_font("DejaVu", "", 8.6)
            self.set_text_color(*GREY)
            self.cell(0, 4.6, caption, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.space(3.2)


def cover(pdf: Manual):
    pdf.add_page()
    pdf.set_fill_color(*NAVY)
    pdf.rect(0, 0, 210, 297, style="F")
    pdf.set_draw_color(*GOLD)
    pdf.set_line_width(0.7)
    pdf.rect(10, 10, 190, 277, style="D")

    pdf.set_font("DejaVu", "B", 29)
    pdf.set_text_color(*GOLD)
    pdf.set_xy(0, 36)
    pdf.cell(210, 14, "VTES Card Scanner", align="C",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("DejaVu", "", 15)
    pdf.set_text_color(*WHITISH)
    pdf.cell(210, 10, "Manual de uso", align="C",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_draw_color(*GOLD)
    pdf.set_line_width(0.4)
    pdf.line(85, 66, 125, 66)

    pdf.image(str(ASSETS / "ankh.jpg"), x=(210 - 46) / 2, y=76, w=46)

    pdf.set_font("DejaVu", "", 12)
    pdf.set_text_color(*WHITISH)
    pdf.set_xy(0, 196)
    pdf.cell(210, 8, "Identificación de cartas VTES en directo para OBS Studio",
             align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_fill_color(*RED)
    pdf.rect(66, 214, 78, 12, style="F")
    pdf.set_font("DejaVu", "B", 10.5)
    pdf.set_text_color(*GOLD)
    pdf.set_xy(66, 215.4)
    pdf.cell(78, 9, f"{VERSION} · {DATE}", align="C",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_font("DejaVu", "B", 10)
    pdf.set_text_color(*GOLD)
    pdf.set_xy(0, 262)
    pdf.cell(210, 6, "La Garra Cifrada", align="C",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("DejaVu", "", 8.6)
    pdf.set_text_color(*GREY)
    pdf.cell(210, 5, "github.com/ChicoCifrado/VTES-for-OBS", align="C",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def toc(pdf: Manual):
    pdf.add_page()
    pdf.h1("Contenido")
    items = [
        "1 · ¿Qué es VTES Card Scanner?",
        "2 · Requisitos",
        "3 · Instalación",
        "4 · Primeros pasos",
        "5 · El overlay de emisión (el diseño del ankh)",
        "6 · Ajustes del filtro",
        "7 · La web de control",
        "8 · Modo realizador (check visual)",
        "9 · Estado de partida automático",
        "10 · Solución de problemas",
        "11 · Privacidad y datos",
        "12 · Créditos y licencia",
    ]
    for it in items:
        pdf.set_font("DejaVu", "", 11.5)
        pdf.set_text_color(*TXT)
        pdf.cell(0, 8.2, it, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.space(4)
    pdf.callout(
        "Cómo leer este manual",
        "Las secciones 1–4 te dejan funcionando en unos minutos. Las secciones 5–8 "
        "describen lo que se ve en pantalla y el uso diario (incluido el modo "
        "realizador). La sección 9 explica el estado de partida automático y la 10 "
        "concentra los problemas típicos con su solución.",
    )


def s1_que_es(pdf: Manual):
    pdf.h1("1 · ¿Qué es VTES Card Scanner?")
    pdf.p(
        "VTES Card Scanner es un plugin (filtro) para OBS Studio que identifica en "
        "tiempo real las cartas de Vampire: The Eternal Struggle que aparecen ante "
        "tu cámara, y monta con ellas una retransmisión lista para emitir: la imagen "
        "de la cámara dentro de un marco decorativo, la carta mostrada sobre el ankh "
        "y la información de partida en pantalla."
    )
    pdf.p("En la práctica, te da:")
    pdf.bullets([
        "Identificación automática: detector YOLO OBB + embeddings visuales como "
        "vía principal incluso en cartas pequeñas + clasificador de tipo (14 tipos). "
        "El OCR del nombre solo se usa como apoyo en primeros planos, a partir de "
        "200 px de alto; la decisión final es del embedding.",
        "Overlay de emisión: tu cámara dentro del diseño del ankh, la última carta "
        "aprobada pintada completa sobre el símbolo y las rulings dibujadas sobre "
        "la carta.",
        "Web de control (móvil o PC): buscador de cartas, control del pool, "
        "detecciones del directo, estado de partida y cola de pendientes.",
        "Estado de partida automático: zonas por jugador, cartas giradas/enderezadas, "
        "influencia, jugadores eliminados y registro de eventos.",
        "Modo realizador: las detecciones no se muestran solas; TÚ decides qué se "
        "pinta revisando la cola de pendientes.",
    ])
    pdf.callout(
        "Proyecto fan",
        "VTES Card Scanner es un proyecto de aficionados, sin relación con Paradox "
        "Interactive. Vampire: The Eternal Struggle es una marca registrada de "
        "Paradox Interactive. El plugin se distribuye bajo licencia GNU GPL v2.",
    )


def s2_requisitos(pdf: Manual):
    pdf.h1("2 · Requisitos")
    pdf.bullets([
        "OBS Studio 30 o superior, en Windows 10/11 de 64 bits.",
        "Cámara: una webcam USB. Cuanta mejor óptica y enfoque, mejor funciona "
        "el embedding, que es la vía principal; el OCR solo entra en cartas "
        "grandes. La cámara debe poder encuadrar las cartas a una distancia "
        "cómoda.",
        "GPU: NVIDIA con CUDA (recomendado). Alternativa: cualquier GPU mediante "
        "DirectML. Modo CPU disponible (lento; solo para probar).",
        "Iluminación uniforme sobre la mesa, sin reflejos directos ni cambios "
        "bruscos de luz.",
        "Opcional: un móvil en la misma red WiFi para controlar la partida durante "
        "el directo.",
        "El instalador incluye los modelos de IA, OpenCV, ONNX Runtime y Tesseract "
        "OCR 5.x; no hace falta instalar nada más.",
    ])


def s3_instalacion(pdf: Manual):
    pdf.h1("3 · Instalación")
    pdf.h2("Opción A — Instalador (recomendada)")
    pdf.steps([
        "Descarga VTES-for-OBS.exe desde la página de Releases del proyecto.",
        "Ejecútalo y sigue el asistente: el instalador localiza OBS automáticamente "
        "desde el registro de Windows (por defecto, C:\\Program Files\\obs-studio).",
        "Abre OBS y añade el filtro a tu cámara (sección 4).",
    ])
    pdf.p(
        "Para desinstalar: cierra OBS y usa «Agregar o quitar programas» → "
        "VTES Card Scanner OBS Plugin → Desinstalar (o ejecuta "
        "uninstall-vtes-card-scanner.exe desde la carpeta del plugin)."
    )
    pdf.h2("Opción B — Copia manual (zip)")
    pdf.steps([
        "Descarga el archivo .zip de la release y descomprímelo.",
        "Copia su contenido dentro de tu carpeta de OBS (normalmente "
        "C:\\Program Files\\obs-studio), manteniendo la estructura de carpetas "
        "(obs-plugins\\64bit\\ y los datos del plugin).",
        "Reinicia OBS.",
    ])
    pdf.h2("Compilar desde el código (desarrolladores)")
    pdf.p(
        "Clona el repositorio y usa el script vtes-grimoire.ps1 (build / deploy). "
        "Los modelos de IA no están en el repositorio; consulta el README para "
        "colocarlos en data\\obs-plugins\\vtes-card-scanner\\."
    )
    pdf.callout(
        "Aviso al añadir el filtro",
        "La primera vez que añades el filtro a una fuente, OBS muestra un aviso con "
        "la dirección de la web de control (http://localhost:8080) y consejos "
        "rápidos: activa «Siempre activo» para detección continua y ajusta el "
        "intervalo si quieres aligerar la carga.",
    )


def s4_primeros_pasos(pdf: Manual):
    pdf.h1("4 · Primeros pasos")
    pdf.steps([
        "Añade a tu escena una fuente «Captura de vídeo (Video Capture Device)» con "
        "tu cámara apuntando a la mesa.",
        "Con la fuente seleccionada: botón derecho → Filtros → + → «Detector de "
        "cartas VTES (OBB)».",
        "En las propiedades del filtro elige el dispositivo de inferencia: "
        "GPU (CUDA) si tienes NVIDIA; GPU (DirectML) para cualquier otra GPU; "
        "CPU como último recurso.",
        "Activa «Mostrar cajas» mientras pruebas: verás cada carta detectada con su "
        "caja y su nombre. Después puedes dejarlo activado o desactivarlo; el "
        "overlay de emisión no depende de esa opción.",
        "Pon una carta ante la cámara: debería identificarse en menos de un segundo "
        "y aparecer en la web. Si el modo realizador está desactivado, también se "
        "pintará directamente sobre el ankh.",
        "Abre la web de control con el botón «Abrir búsqueda de cartas» o visitando "
        "http://localhost:8080.",
    ])
    pdf.callout(
        "Consejos de cámara",
        "Encuadra las cartas lo más grandes posible y dentro de las zonas de su "
        "jugador. Evita sombras y reflejos sobre la carta: el embedding usa la "
        "imagen completa y el OCR solo se ejecuta cuando la carta es un primer "
        "plano de al menos 200 px de alto. Si tu mesa queda muy en perspectiva, "
        "sube un poco la cámara.",
    )


def s5_overlay(pdf: Manual):
    pdf.h1("5 · El overlay de emisión (el diseño del ankh)")
    pdf.p(
        "El resultado del filtro es la imagen final para emitir: tu cámara cubre "
        "el hueco del diseño (escala «cover»: llena el hueco sin barras negras y "
        "recorta lo que sobra) y el diseño se dibuja encima con transparencia "
        "real. El zoom (100–300 %) y la posición permiten elegir el encuadre; el "
        "desplazamiento se recorta automáticamente para que la cámara no deje "
        "bordes. Con «Cámara a sangre» la cámara ocupa también las bandas "
        "laterales y el diseño se dibuja encima. El diseño es 3:2 (1536×1024), "
        "así que en una escena 16:9, sin «a sangre», quedan bandas a los lados: "
        "es parte del estilo. El encuadre solo afecta a la composición visual; "
        "la detección y el embedding siguen usando el fotograma completo y el OCR "
        "solo se aplica a cartas grandes."
    )
    pdf.image_block(ASSETS / "overlay_diagram.jpg", 174,
                    "Anatomía del overlay: el diseño real con la cámara en su hueco.")
    pdf.bullets([
        "Panel izquierdo: una fila por jugador (hasta 5) con su nombre y su pool.",
        "Ankh: la última carta puesta en el overlay se pinta completa sobre el "
        "símbolo.",
        "Iconos de carta: debajo de la carta mostrada se dibuja una fila con el "
        "tipo, el clan o credo y las disciplinas o virtudes. Son los símbolos "
        "oficiales de KRCP, se deduplican y se adaptan al ancho de la carta.",
        "Rulings: el texto de reglas de la carta mostrada se dibuja encima de ella; "
        "su altura se ajusta con «Desplazamiento de rulings».",
        "Ventana de cámara: tu captura de vídeo, encajada con «cover» y ajustable "
        "con zoom y posición.",
        "Cámara a sangre: la cámara llena también las bandas laterales; el diseño "
        "se dibuja encima.",
        "Nombres y pool: se dibujan con la fuente elegida (Cinzel, Grenze Gotisch "
        "o clásica).",
        "Carta grande: ocupa media pantalla; pensada para emisiones vistas en móvil "
        "(se activa en las propiedades).",
        "Mostrar zonas: dibuja los rectángulos de las zonas de cada jugador "
        "(minions y cripta). Útil para calibrar; desactívalo para emitir.",
    ])


def s6_ajustes(pdf: Manual):
    pdf.add_page()
    pdf.h1("6 · Ajustes del filtro")
    pdf.p(
        "Todas las opciones están en las propiedades del filtro (botón derecho sobre "
        "la fuente → Filtros → Detector de cartas VTES). Entre paréntesis, su valor "
        "por defecto."
    )
    pdf.h2("General")
    pdf.defitem("Idioma",
                "Español o English. Cambia los textos del plugin y de la web.")
    pdf.defitem("Mostrar cajas (activado)",
                "Dibuja las cajas de detección y los nombres sobre el vídeo. "
                "Imprescindible para calibrar; puedes desactivarlo después.")
    pdf.defitem("Siempre activo (activado)",
                "Mantiene la detección funcionando aunque «Mostrar cajas» esté "
                "desactivado. En emisión, déjalo activado.")
    pdf.defitem("Carta grande (desactivado)",
                "La carta mostrada ocupa media pantalla. Para torneos y directos "
                "vistos en móvil.")
    pdf.defitem("Desplazamiento de rulings (0 %)",
                "Sube o baja el bloque de rulings sobre la carta mostrada (0–100 %). "
                "0 es la posición base.")
    pdf.h2("Fuente y encuadre")
    pdf.defitem("Fuente de los nombres (Cinzel)",
                "Cinzel (romana) es la predeterminada; «Grenze Gotisch (gótica)» "
                "y «Clásica (OpenCV)» son las alternativas. Si una fuente no está "
                "disponible, se usa la clásica. Afecta a los nombres y al pool del "
                "panel.")
    pdf.defitem("Zoom de cámara (100 %)",
                "De 100 a 300 %. 100 % es el mínimo recorte que cubre el hueco; "
                "súbelo para acercar. En «Cámara a sangre» hace falta más de 100 % "
                "para poder mover el encuadre.")
    pdf.defitem("Posición horizontal / vertical de la cámara (0)",
                "Desplaza el encuadre. El plugin recorta el desplazamiento para que "
                "la cámara siga cubriendo el hueco (o el fotograma completo, con "
                "«Cámara a sangre»).")
    pdf.defitem("Cámara a sangre (desactivado)",
                "La cámara ocupa todo el fotograma, incluidas las bandas laterales "
                "del diseño 3:2; el diseño se dibuja encima. Útil para planos más "
                "amplios.")
    pdf.h2("Detección")
    pdf.defitem("Umbral de confianza (0,50)",
                "De 0 a 1. Súbelo si aparecen detecciones falsas; bájalo si le "
                "cuesta detectar cartas poco iluminadas.")
    pdf.defitem("Modo de detección (Modelo IA — ONNX)",
                "«Modelo IA (ONNX)» es el modo normal. «Forma (contorno)» es un "
                "detector de respaldo sin IA para casos extremos.")
    pdf.defitem("Dispositivo de inferencia (GPU CUDA en Windows)",
                "CPU, GPU (DirectML) o GPU (CUDA). CUDA es lo más rápido con "
                "NVIDIA; DirectML vale para cualquier GPU; CPU solo para probar.")
    pdf.defitem("Intervalo de detección (0 s)",
                "Tiempo mínimo entre detecciones, de 0 a 5 segundos. 0 = continuo. "
                "Súbelo (0,5–1 s) si el PC va justo o usas CPU.")
    pdf.defitem("Clasificador de tipo (activado)",
                "Red de visión que reconoce los 14 tipos de carta (Action, Master, "
                "Combat…). Mejora el filtrado y la información de la web.")
    pdf.defitem("OCR — leer nombres con Tesseract (activado)",
                "Lee el nombre impreso y lo compara con la base de cartas. Solo "
                "se ejecuta cuando la carta mide al menos 200 px de alto en el "
                "fotograma y el embedding no la ha resuelto con seguridad; en "
                "cartas pequeñas no se usa. Si el embedding duda entre dos cartas, "
                "la decisión pasa a la cola de Pendientes.")
    pdf.callout(
        "Cartas pequeñas",
        "El detector y el embedding están pensados para cartas reales de tamaño "
        "reducido. Cuanto más pequeña es la carta, más "
        "importante es el enfoque y la luz; el OCR no se fuerza por debajo de "
        "200 px. Si aparecen dos candidatas, elige en Pendientes en lugar de "
        "confiar en una lectura de texto.",
    )
    pdf.defitem("Escalado NIS (desactivado, requiere CUDA)",
                "Realce del texto pequeño antes del OCR. Mejora la lectura a costa "
                "de algo más de GPU.")
    pdf.defitem("Clasificador CUDA Oval/Rect (activado)",
                "Distingue cartas de otros rectángulos (fundas, teléfonos, la "
                "alfombrilla): menos falsos positivos. Requiere CUDA.")
    pdf.h2("Modo realizador")
    pdf.defitem("«Modo realizador: las detecciones no se muestran solas…» (activado)",
                "Con él activo, cada detección va a la cola de Pendientes y solo se "
                "pinta lo que tú toques (sección 8). Desactivado, se muestra "
                "automáticamente la última carta identificada (modo clásico).")
    pdf.h2("Jugadores")
    pdf.defitem("Número de jugadores (5)",
                "4 o 5 jugadores. El panel del overlay tiene 5 filas.")
    pdf.defitem("Mostrar zonas de jugadores (activado)",
                "Dibuja los rectángulos de las zonas sobre el vídeo (calibración).")
    pdf.defitem("Grupo «Jugador N»",
                "Nombre, Pool inicial (30) y «En juego». Además, la geometría de sus "
                "dos zonas —minions y cripta— con X/Y/Ancho/Alto en valores "
                "normalizados (0–1 respecto al tamaño de la fuente) y ángulo "
                "(−180° a 180°). Una carta se asigna al jugador cuya zona contiene "
                "su centro.")
    pdf.defitem("Abrir búsqueda de cartas (botón)",
                "Abre la web de control en tu navegador; si el servidor no está "
                "activo, lo arranca.")


def s7_web(pdf: Manual):
    pdf.add_page()
    pdf.h1("7 · La web de control")
    pdf.p(
        "El plugin arranca un servidor web en el puerto 8080 al añadir el filtro. "
        "Desde ahí controlas todo durante la partida, desde el PC o el móvil."
    )
    pdf.bullets([
        "En el PC: http://localhost:8080",
        "En el móvil (misma red WiFi): http://IP-DEL-PC:8080 — la primera vez "
        "Windows pedirá permitir a OBS en la red privada; acepta el aviso.",
    ])
    pdf.p(
        "La web usa las mismas fuentes oficiales de iconos que el overlay: cada "
        "carta muestra su tipo, clan o credo y disciplinas o virtudes. Las fuentes "
        "se sirven desde /fonts/ del propio plugin; no hace falta conexión."
    )
    pdf.p("La barra superior enlaza todas las páginas: Buscar · Pool · Detecciones · "
          "Partida · Pendientes.")
    pdf.defitem("Buscar (/)",
                "Buscador sobre la base de cartas. Admite nombre (en español, inglés "
                "o francés), type:, clan:, disc: (abreviatura de disciplina), cap: "
                "(capacidad exacta), cap>N, group: y texto libre, combinables. "
                "Muestra hasta 50 resultados con los iconos oficiales de tipo, clan "
                "y disciplinas; al abrir una carta puedes enviarla al overlay con un "
                "clic.")
    pdf.defitem("Pool (/pool)",
                "Control del pool de cada jugador con botones. El pool no se puede "
                "leer por cámara (bleeds, costes), así que se ajusta a mano desde "
                "aquí o desde Partida.")
    pdf.defitem("Detecciones (/detections)",
                "Últimas detecciones del directo con su imagen, nombre, hora y sus "
                "iconos. Un clic en cualquiera la pinta en el ankh al momento.")
    pdf.defitem("Partida (/game)",
                "Estado de la partida: por jugador, el pool, cuántas cartas están "
                "giradas o listas, sus minions detectados y las cartas en cripta. "
                "Abajo, el Registro con los últimos eventos (giros, influencia, "
                "pool, ousted) y su antigüedad.")
    pdf.defitem("Pendientes (/pending, alias /pendientes)",
                "La cola del modo realizador: cada pendiente muestra sus iconos y "
                "aquí apruebas qué se muestra en el overlay (sección 8).")


def s8_director(pdf: Manual):
    pdf.add_page()
    pdf.h1("8 · Modo realizador (check visual)")
    pdf.p(
        "Es la forma recomendada de emitir: el plugin detecta e identifica, pero no "
        "pinta nada por su cuenta. Las detecciones se acumulan en la web y tú "
        "apruebas cuál se muestra. Así nunca sale una detección dudosa en directo… "
        "y de paso cada elección queda como dato para mejorar futuras versiones."
    )
    pdf.h2("Flujo de uso")
    pdf.steps([
        "Con «Modo realizador» activado (lo está por defecto), muestra cartas a la "
        "cámara con normalidad.",
        "Abre Pendientes en el móvil o en el PC: cada carta detectada aparece con su "
        "imagen, su recorte de evidencia y su nivel de confianza. En el overlay no "
        "se pinta nada todavía.",
        "Toca la carta correcta: se pinta al momento sobre el ankh y la cola se "
        "limpia entera (las demás desaparecen).",
        "Si ninguna es aprovechable, pulsa «Descartar todas».",
        "Para vaciar el ankh cuando quieras, pulsa «Quitar del overlay».",
    ])
    pdf.bullets([
        "Dedupe por carta: si la misma carta vuelve a detectarse, se refresca en la "
        "cola en vez de duplicarse.",
        "Si dos cartas quedan casi empatadas (por ejemplo, con una captura borrosa), "
        "la cola muestra las dos candidatas para que elijas la correcta; el plugin "
        "no pinta ninguna de ellas automáticamente.",
        "La cola guarda hasta 24 pendientes; las que no vuelven a verse caducan a "
        "los 120 segundos.",
        "La vista se comparte entre dispositivos: el móvil y el PC ven la misma "
        "cola; elige desde donde te resulte más cómodo.",
        "Con el modo desactivado vuelves al comportamiento clásico: se muestra "
        "automáticamente la última carta identificada.",
    ])
    pdf.h2("Datos de entrenamiento")
    pdf.p(
        "Cada elección del modo realizador genera datos en tu propio equipo (nunca "
        "salen a internet). En la carpeta de configuración del plugin encontrarás "
        "el historial y los recortes de cada decisión:"
    )
    pdf.mono_block([
        "%AppData%\\obs-studio\\plugin_config\\vtes-card-scanner\\feedback\\",
        "  events.jsonl      historial de elecciones y descartes",
        "  events\\<id>\\      imagen elegida + candidatas de cada evento",
    ])
    pdf.p(
        "Puedes borrar esa carpeta cuando quieras; si la conservas, ayuda a mejorar "
        "el reconocimiento en versiones futuras del plugin."
    )


def s9_partida(pdf: Manual):
    pdf.h1("9 · Estado de partida automático")
    pdf.p(
        "Con las zonas configuradas, el plugin deduce la situación de la mesa a "
        "partir de lo que ve."
    )
    pdf.h2("Zonas de jugador")
    pdf.bullets([
        "Cada jugador tiene dos rectángulos: minions (arriba por defecto) y cripta "
        "(abajo), ambos con ángulo ajustable.",
        "Ajusta X/Y/Ancho/Alto como valores de 0 a 1 respecto al tamaño de tu "
        "fuente; con «Mostrar zonas» activado los verás dibujados en pantalla.",
        "Una carta pertenece al jugador cuya zona contiene su centro; si no cae en "
        "ninguna, queda «fuera» y no cuenta para el estado.",
    ])
    pdf.h2("Zonas automáticas (cuatro cartas boca abajo)")
    pdf.p(
        "Activa «Detectar zonas automáticamente (4 cartas boca abajo)» en las "
        "propiedades del filtro. En la zona de cada jugador coloca sus cuatro "
        "cartas de cripta boca abajo, en fila, juntas y con fundas del mismo "
        "diseño. El detector no intenta leer su cara: agrupa los cuatro dorsos "
        "por forma, tamaño, color y alineación."
    )
    pdf.bullets([
        "La cripta se crea alrededor de la fila; la zona de minions ocupa la "
        "mitad superior hacia el centro de la mesa, con la misma anchura que la "
        "fila.",
        "Las zonas se asignan por orden de asiento: empezando por abajo y "
        "siguiendo el sentido horario en una mesa circular, o de izquierda a "
        "derecha / de arriba abajo en una fila. Comprueba el resultado en /game "
        "con «Zonas automáticas: n/total».",
        "Mientras falten grupos se usan las zonas manuales como respaldo. Al "
        "desactivar la opción vuelven las zonas manuales.",
        "Las zonas automáticas se recalculan al activar la opción o cambiar el "
        "número de jugadores; no se guardan en la configuración del filtro.",
    ])
    pdf.callout(
        "Si no detecta una fila",
        "Comprueba que se ven los cuatro bordes, que las cartas no son demasiado "
        "pequeñas y que hay contraste con la mesa. Evita reflejos y fundas "
        "transparentes. Como respaldo, ajusta las zonas manuales.",
    )
    pdf.h2("Cartas giradas (lock)")
    pdf.bullets([
        "Solo se evalúan las cartas en la zona de minions: en la cripta no se giran.",
        "El giro se mide respecto al ángulo de la zona: ≈90° = girada; vuelta a su "
        "posición = lista. La evaluación es continua: una carta puede girarse o "
        "enderezarse en cualquier momento de la partida.",
        "El estado se ve en /game (giradas/listas por jugador) y en el registro.",
    ])
    pdf.h2("Influencia (cripta → minions)")
    pdf.p(
        "Cuando una carta identificada pasa de la zona de cripta a la de minions "
        "(la influencia que se completa en la mesa real), el Registro muestra "
        "«influencia completada» y se descuenta del pool la capacidad del "
        "vampiro, que es el coste de una influencia completa. La ficha del "
        "jugador muestra cuántas influencias lleva y el pool gastado."
    )
    pdf.bullets([
        "Para cobrar el coste la carta debe estar identificada por el embedding; "
        "si aún no tiene identidad, el paso a minions no descuenta pool.",
        "La capacidad de un vampiro está entre 1 y 11 y es el coste de una "
        "influencia completa. El plugin no puede leer los contadores de sangre "
        "de la carta boca abajo, así que descuenta la capacidad cuando la carta "
        "aparece en minions, no transferencia a transferencia.",
        "Si la carta vuelve a la cripta y permanece allí unos segundos, el "
        "detector se rearma para una nueva influencia. Un parpadeo breve de zona "
        "no duplica el cobro.",
        "El pool también se puede ajustar a mano desde la web para bleeds y "
        "otros efectos.",
    ])
    pdf.h2("Pool y eliminación")
    pdf.bullets([
        "El pool se controla desde la web: el plugin no puede leer los contadores "
        "de sangre ni el pool a través de la cámara.",
        "La influencia completada descuenta automáticamente la capacidad del "
        "vampiro; el resto de cambios se ajustan a mano.",
        "Si un pool llega a 0, el jugador se marca como eliminado (ousted), con "
        "etiqueta y aspecto atenuado. Si lo subes de nuevo (por ejemplo, tras un "
        "toque accidental), se reactiva solo.",
    ])
    pdf.h2("Registro")
    pdf.p(
        "En /game se listan los últimos eventos con su antigüedad: giros y "
        "enderezados, pasos de cripta a minions, cambios de pool y "
        "eliminaciones/reactivaciones."
    )


def s10_problemas(pdf: Manual):
    pdf.add_page()
    pdf.h1("10 · Solución de problemas")
    pdf.defitem("No detecta ninguna carta",
                "Comprueba en el Registro de OBS (Herramientas → Registro de "
                "errores) que los modelos se cargan al iniciar el filtro. Ilumina "
                "mejor la mesa y llena más el encuadre con la carta. Si hace falta, "
                "baja el umbral de confianza a 0,35–0,45. Confirma que «Siempre "
                "activo» está marcado.")
    pdf.defitem("Falsos positivos (detecta cosas que no son cartas)",
                "Sube el umbral de confianza; activa «Clasificador CUDA Oval/Rect» y "
                "el «Escalado NIS» si tienes NVIDIA; sube ligeramente el intervalo "
                "de detección.")
    pdf.defitem("Va lento o el PC sufre",
                "Cambia de CPU a GPU (DirectML o CUDA); sube el intervalo de "
                "detección a 0,5–1 s; desactiva el escalado NIS si lo tenías activo.")
    pdf.defitem("La web no abre en el PC",
                "El servidor arranca cuando el filtro está añadido a alguna fuente: "
                "compruébalo. Si otro programa ocupa el puerto 8080, el Registro de "
                "OBS mostrará un aviso.")
    pdf.defitem("El móvil no conecta con la web",
                "Usa la IP del PC (no «localhost») y asegúrate de estar en la misma "
                "red WiFi. La primera vez, permite a OBS en el Firewall de Windows "
                "para redes privadas.")
    pdf.defitem("En modo realizador no se muestra nada en el ankh",
                "Es lo esperado: no se pinta nada automáticamente. Aprueba las "
                "cartas desde la página Pendientes.")
    pdf.defitem("Toco una pendiente y no se pinta nada",
                "Esa detección se quedó sin imagen asociada. Vuelve a mostrar la "
                "carta ante la cámara y usa la nueva pendiente, o píntala desde "
                "Detecciones.")
    pdf.defitem("El giro (lock) se detecta al revés",
                "Revisa el ángulo de la zona de minions de ese jugador en las "
                "propiedades: debe coincidir con la orientación real de la mesa.")
    pdf.defitem("El OCR confunde nombres de cartas",
                "Mejora la luz y evita reflejos sobre la carta. La identificación "
                "por embedding es la vía principal y corrige la mayoría de los "
                "casos; el OCR solo se usa en cartas de al menos 200 px de alto.")
    pdf.defitem("La carta es pequeña y no se identifica",
                "Acércala o mejora el enfoque: el embedding necesita que la carta "
                "se vea nítida. Por debajo de 200 px no se ejecuta OCR; si el "
                "embedding devuelve dos candidatas, aparecen en Pendientes para "
                "elegir con un clic.")
    pdf.defitem("¿Dónde consulto los errores?",
                "Herramientas → Registro de errores de OBS. Las marcas [Game], "
                "[Director], [WebServer] y [OCR] siguen el flujo del plugin.")


def s11_privacidad(pdf: Manual):
    pdf.h1("11 · Privacidad y datos")
    pdf.p(
        "Todo el procesamiento es local: la cámara, las detecciones y la web "
        "funcionan en tu equipo, y el plugin no envía nada a internet. Los únicos "
        "datos que genera son su configuración y el feedback del modo realizador, "
        "en:"
    )
    pdf.mono_block([
        "%AppData%\\obs-studio\\plugin_config\\vtes-card-scanner\\",
        "    (configuración, base de cartas copiada y carpeta feedback\\ opcional)",
    ])
    pdf.p("Puedes borrar esa carpeta cuando quieras.")


def s12_creditos(pdf: Manual):
    pdf.h1("12 · Créditos y licencia")
    pdf.bullets([
        "Proyecto: VTES Card Scanner — La Garra Cifrada.",
        "Repositorio e incidencias: github.com/ChicoCifrado/VTES-for-OBS",
        "Licencia: GNU GPL v2.0 (ver el archivo LICENSE del repositorio).",
        "Vampire: The Eternal Struggle es una marca de Paradox Interactive. Este "
        "proyecto es contenido no oficial creado por aficionados.",
    ])
    pdf.p(f"Manual correspondiente a {VERSION} ({DATE.lower()}).", size=9.5,
          color=GREY)


def main():
    make_assets()
    pdf = Manual()
    pdf.set_title("VTES Card Scanner — Manual de uso")
    pdf.set_author("La Garra Cifrada")
    pdf.set_creator("build_manual.py (fpdf2)")
    cover(pdf)
    toc(pdf)
    s1_que_es(pdf)
    s2_requisitos(pdf)
    s3_instalacion(pdf)
    s4_primeros_pasos(pdf)
    s5_overlay(pdf)
    s6_ajustes(pdf)
    s7_web(pdf)
    s8_director(pdf)
    s9_partida(pdf)
    s10_problemas(pdf)
    s11_privacidad(pdf)
    s12_creditos(pdf)
    pdf.output(str(OUT))
    print(f"PDF generado: {OUT} ({pdf.pages_count} páginas)")


if __name__ == "__main__":
    main()
