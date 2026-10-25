#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Genera la versión en inglés del manual de VTES Card Scanner (PDF).

Reutiliza la infraestructura de build_manual.py (clase Manual, paleta,
assets) y solo redefine los textos en inglés.

Uso:  python3 build_manual_en.py
Salida: VTES-for-OBS_Manual_v0.1.7_EN.pdf (en esta misma carpeta)
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from fpdf.enums import XPos, YPos

import build_manual as base

BASE = Path(__file__).resolve().parent
ASSETS = base.ASSETS
OUT = BASE / "VTES-for-OBS_Manual_v0.1.7_EN.pdf"

VERSION = base.VERSION
DATE = "October 2026"


# ── Assets en inglés (diagrama anotado) ───────────────────────────────────
def make_assets_en() -> None:
    if not (ASSETS / "ankh.jpg").exists():
        base.make_assets()  # también crea el ankh recortado
    ASSETS.mkdir(parents=True, exist_ok=True)
    design = Image.open(base.OVERLAY_PNG).convert("RGBA")

    canvas = Image.new("RGBA", design.size, (14, 14, 22, 255))
    canvas.alpha_composite(design)
    img = canvas.convert("RGB")
    d = ImageDraw.Draw(img, "RGBA")
    f_lab = ImageFont.truetype(base.FONT_BOLD, 30)
    f_sub = ImageFont.truetype(base.FONT_REG, 24)
    f_big = ImageFont.truetype(base.FONT_BOLD, 46)

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

    d.text((890, 470), "CAMERA WINDOW", font=f_big, anchor="mm",
           fill=(238, 232, 210), stroke_width=4, stroke_fill=(0, 0, 0))
    d.text((890, 522), "your feed fits here, whole and uncropped",
           font=f_sub, anchor="mm", fill=(205, 196, 170), stroke_width=3,
           stroke_fill=(0, 0, 0))

    label_box(300, 58, 1000, 130, "PLAYER PANEL — name and pool", f_lab)
    arrow(300, 96, 252, 106)
    label_box(300, 636, 1010, 708, "ANKH — last shown card", f_lab)
    arrow(300, 672, 246, 704)

    img.save(ASSETS / "overlay_diagram_en.jpg", quality=88, optimize=True,
             progressive=True)
    print("Assets EN generados:", ASSETS / "overlay_diagram_en.jpg")


# ── Documento (inglés) ────────────────────────────────────────────────────
class ManualEN(base.Manual):
    def footer(self):
        if self.page_no() == 1:
            return
        self.set_draw_color(*base.GOLD)
        self.set_line_width(0.25)
        self.line(18, 283, 192, 283)
        self.set_y(-22)
        self.set_font("DejaVu", "", 8)
        self.set_text_color(*base.GREY)
        self.cell(120, 6, f"VTES Card Scanner · User manual {VERSION}",
                  new_x=XPos.RIGHT, new_y=YPos.TOP)
        self.cell(0, 6, f"Page {self.page_no()}", align="R",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def cover(pdf: ManualEN):
    pdf.add_page()
    pdf.set_fill_color(*base.NAVY)
    pdf.rect(0, 0, 210, 297, style="F")
    pdf.set_draw_color(*base.GOLD)
    pdf.set_line_width(0.7)
    pdf.rect(10, 10, 190, 277, style="D")

    pdf.set_font("DejaVu", "B", 29)
    pdf.set_text_color(*base.GOLD)
    pdf.set_xy(0, 36)
    pdf.cell(210, 14, "VTES Card Scanner", align="C",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("DejaVu", "", 15)
    pdf.set_text_color(*base.WHITISH)
    pdf.cell(210, 10, "User manual", align="C",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_draw_color(*base.GOLD)
    pdf.set_line_width(0.4)
    pdf.line(85, 66, 125, 66)

    pdf.image(str(ASSETS / "ankh.jpg"), x=(210 - 46) / 2, y=76, w=46)

    pdf.set_font("DejaVu", "", 12)
    pdf.set_text_color(*base.WHITISH)
    pdf.set_xy(0, 196)
    pdf.cell(210, 8, "Real-time VTES card identification for OBS Studio",
             align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_fill_color(*base.RED)
    pdf.rect(66, 214, 78, 12, style="F")
    pdf.set_font("DejaVu", "B", 10.5)
    pdf.set_text_color(*base.GOLD)
    pdf.set_xy(66, 215.4)
    pdf.cell(78, 9, f"{VERSION} · {DATE}", align="C",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_font("DejaVu", "B", 10)
    pdf.set_text_color(*base.GOLD)
    pdf.set_xy(0, 262)
    pdf.cell(210, 6, "La Garra Cifrada", align="C",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("DejaVu", "", 8.6)
    pdf.set_text_color(*base.GREY)
    pdf.cell(210, 5, "github.com/ChicoCifrado/VTES-for-OBS", align="C",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def toc(pdf: ManualEN):
    pdf.add_page()
    pdf.h1("Contents")
    items = [
        "1 · What is VTES Card Scanner?",
        "2 · Requirements",
        "3 · Installation",
        "4 · Getting started",
        "5 · The broadcast overlay (the ankh design)",
        "6 · Filter settings",
        "7 · The control web app",
        "8 · Director mode (visual check)",
        "9 · Automatic game state",
        "10 · Troubleshooting",
        "11 · Privacy and data",
        "12 · Credits and license",
    ]
    for it in items:
        pdf.set_font("DejaVu", "", 11.5)
        pdf.set_text_color(*base.TXT)
        pdf.cell(0, 8.2, it, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.space(4)
    pdf.callout(
        "How to read this manual",
        "Sections 1–4 will get you running in a few minutes. Sections 5–8 "
        "describe what you see on screen and everyday use (including director "
        "mode). Section 9 explains the automatic game state and section 10 "
        "gathers the typical problems with their solutions.",
    )


def s1_what(pdf: ManualEN):
    pdf.h1("1 · What is VTES Card Scanner?")
    pdf.p(
        "VTES Card Scanner is a plugin (filter) for OBS Studio that identifies "
        "Vampire: The Eternal Struggle cards in real time as they appear in front "
        "of your camera, and builds a broadcast-ready stream with them: your "
        "camera inside a decorative frame, the shown card on the ankh and game "
        "information on screen."
    )
    pdf.p("In practice, it gives you:")
    pdf.bullets([
        "Automatic identification: YOLO OBB detector + visual embeddings as the "
        "main path even on small cards + type classifier (14 card types). OCR on "
        "the name is only a backup for close-ups from 200 px high; the final "
        "decision comes from the embedding.",
        "Broadcast overlay: your camera inside the ankh design, the last approved "
        "card shown whole over the symbol, and the rulings drawn on top of the "
        "card.",
        "Control web app (phone or PC): card search, pool control, live "
        "detections, game state and the pending queue.",
        "Automatic game state: per-player zones, locked/unlocked cards, "
        "influence, eliminated players and an event log.",
        "Director mode: detections are not shown automatically; YOU decide what "
        "gets drawn by reviewing the pending queue.",
    ])
    pdf.callout(
        "Fan project",
        "VTES Card Scanner is a community project, not related to Paradox "
        "Interactive. Vampire: The Eternal Struggle is a trademark of Paradox "
        "Interactive. The plugin is distributed under the GNU GPL v2 license.",
    )


def s2_requirements(pdf: ManualEN):
    pdf.h1("2 · Requirements")
    pdf.bullets([
        "OBS Studio 30 or newer, on Windows 10/11 64-bit.",
        "Camera: a USB webcam. The better the optics and focus, the better the "
        "embedding works; it is the main path, while OCR only runs on large "
        "cards. The camera must be able to frame the cards at a comfortable "
        "distance.",
        "GPU: NVIDIA with CUDA (recommended). Alternative: any GPU via DirectML. "
        "CPU mode available (slow; testing only).",
        "Even lighting over the table, with no direct glare or abrupt light "
        "changes.",
        "Optional: a phone on the same WiFi network to control the game during "
        "the stream.",
        "The installer bundles the AI models, OpenCV, ONNX Runtime and Tesseract "
        "OCR 5.x; no extra installs needed.",
    ])


def s3_installation(pdf: ManualEN):
    pdf.h1("3 · Installation")
    pdf.h2("Option A — Installer (recommended)")
    pdf.steps([
        "Download VTES-for-OBS.exe from the project's Releases page.",
        "Run it and follow the wizard: the installer locates OBS automatically "
        "from the Windows registry (default: C:\\Program Files\\obs-studio).",
        "Open OBS and add the filter to your camera (section 4).",
    ])
    pdf.p(
        "To uninstall: close OBS and use «Add or remove programs» → VTES Card "
        "Scanner OBS Plugin → Uninstall (or run uninstall-vtes-card-scanner.exe "
        "from the plugin folder)."
    )
    pdf.h2("Option B — Manual copy (zip)")
    pdf.steps([
        "Download the release .zip and extract it.",
        "Copy its contents into your OBS folder (usually "
        "C:\\Program Files\\obs-studio), keeping the folder structure "
        "(obs-plugins\\64bit\\ plus the plugin data).",
        "Restart OBS.",
    ])
    pdf.h2("Building from source (developers)")
    pdf.p(
        "Clone the repository and use the vtes-grimoire.ps1 script (build / "
        "deploy). The AI models are not in the repository; see the README to "
        "place them in data\\obs-plugins\\vtes-card-scanner\\."
    )
    pdf.callout(
        "Notice when adding the filter",
        "The first time you add the filter to a source, OBS shows a notice with "
        "the control web address (http://localhost:8080) and quick tips: enable "
        "«Always active» for continuous detection and adjust the interval if you "
        "want to lighten the load.",
    )


def s4_getting_started(pdf: ManualEN):
    pdf.h1("4 · Getting started")
    pdf.steps([
        "Add a «Video Capture Device» source to your scene, with your camera "
        "pointing at the table.",
        "With the source selected: right-click → Filters → + → «VTES Card "
        "Detector (OBB)».",
        "In the filter properties choose the inference device: GPU (CUDA) if you "
        "have NVIDIA; GPU (DirectML) for any other GPU; CPU as a last resort.",
        "Enable «Show bounding boxes» while testing: each detected card appears "
        "with its box and name. You can leave it on or turn it off later; the "
        "broadcast overlay does not depend on it.",
        "Hold a card in front of the camera: it should be identified in under a "
        "second and appear on the web. If director mode is off, it will also be "
        "drawn straight onto the ankh.",
        "Open the control web with the «Open card search» button or by visiting "
        "http://localhost:8080.",
    ])
    pdf.callout(
        "Camera tips",
        "Frame the cards as large as possible and inside their player's zone. "
        "Avoid shadows and glare on the card: the embedding uses the whole image "
        "and OCR only runs when the card is a close-up at least 200 px high. If "
        "your table is at a steep angle, raise the camera a little.",
    )


def s5_overlay(pdf: ManualEN):
    pdf.h1("5 · The broadcast overlay (the ankh design)")
    pdf.p(
        "The filter's output is the final broadcast image: your camera covers the "
        "decorative design's window («cover» scale: it fills the window with no "
        "black bars, cropping the excess) and the design is drawn on top with real "
        "transparency. Zoom (100–300%) and position let you choose the framing; "
        "the offset is clamped automatically so the camera never leaves edges. "
        "With «Full-bleed camera» the camera also fills the side bands and the "
        "design is drawn on top. The design is 3:2 (1536×1024), so a 16:9 scene "
        "without full-bleed will have bands at the sides: that is part of the "
        "style. Framing only affects the visual composition; detection and the "
        "embedding still use the full frame, and OCR only runs on large cards."
    )
    pdf.image_block(ASSETS / "overlay_diagram_en.jpg", 174,
                    "Overlay anatomy: the real design with the camera in its window.")
    pdf.bullets([
        "Left panel: one row per player (up to 5) with their name and pool.",
        "Ankh: the last card sent to the overlay is shown whole over the symbol.",
        "Card icons: a row under the shown card with its type, clan or creed and "
        "disciplines or virtues. They are the official KRCP symbols, deduplicated "
        "and fitted to the card width.",
        "Rulings: the rules text of the shown card is drawn on top of it; its "
        "height is adjusted with «Rulings scroll».",
        "Camera window: your video feed, cover-fitted and adjustable with zoom and "
        "position.",
        "Full-bleed camera: the camera also fills the side bands; the design is "
        "drawn on top.",
        "Names and pool: drawn with the selected font (Cinzel, Grenze Gotisch or "
        "Classic).",
        "Big card: takes half the screen; meant for streams watched on a phone "
        "(enable it in the properties).",
        "Show zones: draws the rectangles of each player's zones (minions and "
        "crypt). Useful for calibration; turn it off to broadcast.",
    ])


def s6_settings(pdf: ManualEN):
    pdf.add_page()
    pdf.h1("6 · Filter settings")
    pdf.p(
        "All options live in the filter properties (right-click the source → "
        "Filters → VTES Card Detector). Defaults in parentheses."
    )
    pdf.h2("General")
    pdf.defitem("Language",
                "Spanish or English. Changes the plugin and web texts.")
    pdf.defitem("Show bounding boxes (on)",
                "Draws detection boxes and names over the video. Essential for "
                "calibration; you can turn it off afterwards.")
    pdf.defitem("Always active (on)",
                "Keeps detection running even when «Show bounding boxes» is off. "
                "Leave it on while streaming.")
    pdf.defitem("Big card (off)",
                "The shown card takes half the screen. For tournaments and "
                "streams watched on a phone.")
    pdf.defitem("Rulings scroll (0 %)",
                "Moves the rulings block up or down over the shown card (0–100 %). "
                "0 is the base position.")
    pdf.h2("Font and framing")
    pdf.defitem("Player name font (Cinzel)",
                "Cinzel (Roman) is the default; «Grenze Gotisch (gothic)» and "
                "«Classic (OpenCV)» are the alternatives. If a font is missing, the "
                "classic one is used. Affects the panel's names and pool.")
    pdf.defitem("Camera zoom (100%)",
                "100 to 300%. 100% is the minimum crop that covers the window; "
                "raise it to zoom in. In full-bleed mode you need more than 100% to "
                "be able to move the framing.")
    pdf.defitem("Camera horizontal / vertical position (0)",
                "Moves the framing. The plugin clamps the offset so the camera keeps "
                "covering the window (or the whole frame, with full-bleed).")
    pdf.defitem("Full-bleed camera (off)",
                "The camera fills the whole frame, including the side bands of the "
                "3:2 design; the design is drawn on top. Useful for wider shots.")
    pdf.h2("Detection")
    pdf.defitem("Confidence threshold (0.50)",
                "From 0 to 1. Raise it if false detections appear; lower it if "
                "cards are hard to detect in poor light.")
    pdf.defitem("Detection mode (AI Model — ONNX)",
                "«AI Model (ONNX)» is the normal mode. «Shape (contour)» is a "
                "backup detector without AI for extreme cases.")
    pdf.defitem("Inference device (GPU CUDA on Windows)",
                "CPU, GPU (DirectML) or GPU (CUDA). CUDA is fastest with NVIDIA; "
                "DirectML works with any GPU; CPU is testing only.")
    pdf.defitem("Detection interval (0 s)",
                "Minimum time between detections, 0 to 5 seconds. 0 = continuous. "
                "Raise it (0.5–1 s) if the PC struggles or you use CPU.")
    pdf.defitem("Type classifier (on)",
                "Vision network that recognises the 14 card types (Action, "
                "Master, Combat…). Improves filtering and the web info.")
    pdf.defitem("OCR — read card names with Tesseract (on)",
                "Reads the printed name and matches it against the card database. "
                "It only runs when the card is at least 200 px high in the frame "
                "and the embedding is not already confident; it is not used on "
                "small cards. If the embedding is torn between two cards, the "
                "decision goes to the Pending queue.")
    pdf.callout(
        "Small cards",
        "The detector and embedding are designed for real cards at reduced size. "
        "The smaller the card, the more focus and light "
        "matter; OCR is not forced below 200 px. If two candidates appear, choose "
        "in Pending instead of trusting a text read.",
    )
    pdf.defitem("NIS upscaling (off, requires CUDA)",
                "Sharpens small text before OCR. Better reading at some GPU "
                "cost.")
    pdf.defitem("CUDA Oval/Rect classifier (on)",
                "Tells cards apart from other rectangles (sleeves, phones, the "
                "playmat): fewer false positives. Requires CUDA.")
    pdf.h2("Director mode")
    pdf.defitem("«Director mode: detections are not shown automatically…» (on)",
                "When on, every detection goes to the Pending queue and only what "
                "you tap gets drawn (section 8). When off, the last identified "
                "card is shown automatically (classic mode).")
    pdf.h2("Players")
    pdf.defitem("Player count (5)",
                "4 or 5 players. The overlay panel has 5 rows.")
    pdf.defitem("Show player zones (on)",
                "Draws the zone rectangles over the video (calibration).")
    pdf.defitem("«Player N» group",
                "Name, Initial pool (30) and «In game». Plus the geometry of "
                "their two zones —minions and crypt— with X/Y/Width/Height as "
                "normalised values (0–1 relative to the source size) and angle "
                "(−180° to 180°). A card is assigned to the player whose zone "
                "contains its centre.")
    pdf.defitem("Open card search (button)",
                "Opens the control web in your browser; starts the server if it "
                "is not running.")


def s7_web(pdf: ManualEN):
    pdf.add_page()
    pdf.h1("7 · The control web app")
    pdf.p(
        "The plugin starts a web server on port 8080 when the filter is added. "
        "From there you control everything during the game, from the PC or the "
        "phone."
    )
    pdf.bullets([
        "On the PC: http://localhost:8080",
        "On the phone (same WiFi): http://PC-IP:8080 — the first time Windows "
        "will ask to allow OBS on the private network; accept the prompt.",
    ])
    pdf.p(
        "The web uses the same official icon fonts as the overlay: every card "
        "shows its type, clan or creed and disciplines or virtues. The fonts are "
        "served from the plugin's own /fonts/ path; no connection is needed."
    )
    pdf.p("The top bar links every page: Search · Pool · Detections · Game · "
          "Pending.")
    pdf.defitem("Search (/)",
                "Search across the card database. Accepts name (Spanish, English "
                "or French), type:, clan:, disc: (discipline abbreviation), cap: "
                "(exact capacity), cap>N, group: and free text, all combinable. "
                "Shows up to 50 results with the official type, clan and discipline "
                "icons; open a card to send it to the overlay with one click.")
    pdf.defitem("Pool (/pool)",
                "Pool control per player with buttons. The pool cannot be read "
                "from the camera (bleeds, costs), so it is adjusted by hand here "
                "or from Game.")
    pdf.defitem("Detections (/detections)",
                "Latest live detections with image, name, time and icons. One click "
                "on any of them draws it on the ankh instantly.")
    pdf.defitem("Game (/game)",
                "Game state: per player, the pool, how many cards are locked or "
                "unlocked, detected minions and crypt cards. Below, the Log with "
                "recent events (locks, influence, pool, ousted) and their age.")
    pdf.defitem("Pending (/pending, alias /pendientes)",
                "The director mode queue: each pending card shows its icons and "
                "here you approve what gets shown on the overlay (section 8).")


def s8_director(pdf: ManualEN):
    pdf.add_page()
    pdf.h1("8 · Director mode (visual check)")
    pdf.p(
        "It is the recommended way to broadcast: the plugin detects and "
        "identifies, but draws nothing on its own. Detections pile up on the web "
        "and you approve which one is shown. That way a doubtful detection never "
        "goes on stream… and every choice also becomes data to improve future "
        "versions."
    )
    pdf.h2("Workflow")
    pdf.steps([
        "With «Director mode» on (it is by default), show cards to the camera as "
        "usual.",
        "Open Pending on the phone or PC: every detected card appears with its "
        "image, its evidence crop and its confidence level. Nothing is drawn on "
        "the overlay yet.",
        "Tap the right card: it is drawn instantly over the ankh and the whole "
        "queue is cleared (the rest disappear).",
        "If none of them is usable, press «Discard all».",
        "To clear the ankh whenever you want, press «Remove from overlay».",
    ])
    pdf.bullets([
        "Per-card dedupe: if the same card is detected again, it refreshes in "
        "the queue instead of duplicating.",
        "If two cards are nearly tied (for example, in a blurry capture), the "
        "queue shows both candidates so you can pick the right one; the plugin "
        "draws neither of them automatically.",
        "The queue holds up to 24 pending cards; ones not seen again expire "
        "after 120 seconds.",
        "The view is shared between devices: phone and PC see the same queue; "
        "approve from wherever is handiest.",
        "With the mode off you return to the classic behaviour: the last "
        "identified card is shown automatically.",
    ])
    pdf.h2("Training data")
    pdf.p(
        "Every director mode choice generates data on your own machine (it never "
        "leaves your equipment). In the plugin's config folder you will find the "
        "history and the crops of each decision:"
    )
    pdf.mono_block([
        "%AppData%\\obs-studio\\plugin_config\\vtes-card-scanner\\feedback\\",
        "  events.jsonl      history of approvals and discards",
        "  events\\<id>\\      chosen image + candidates of each event",
    ])
    pdf.p(
        "You can delete that folder whenever you want; if you keep it, it helps "
        "improve recognition in future versions of the plugin."
    )


def s9_game_state(pdf: ManualEN):
    pdf.h1("9 · Automatic game state")
    pdf.p(
        "With the zones configured, the plugin works out the table situation "
        "from what it sees."
    )
    pdf.h2("Player zones")
    pdf.bullets([
        "Each player has two rectangles: minions (top by default) and crypt "
        "(bottom), both with adjustable angle.",
        "Adjust X/Y/Width/Height as values from 0 to 1 relative to your source "
        "size; with «Show player zones» enabled you will see them drawn on "
        "screen.",
        "A card belongs to the player whose zone contains its centre; if it "
        "falls in none, it is «outside» and does not count for the state.",
    ])
    pdf.h2("Automatic zones (four face-down cards)")
    pdf.p(
        "Enable «Detect zones automatically (4 face-down cards)» in the filter "
        "properties. In each player's area, place their four crypt cards face "
        "down, in a row, close together and with matching sleeves. The detector "
        "does not try to read their faces: it groups the four backs by shape, "
        "size, colour and alignment."
    )
    pdf.bullets([
        "The crypt is created around the row; the minions zone covers the upper "
        "half toward the centre of the table, with the same width as the row.",
        "Zones are assigned by seating order: starting at the bottom and going "
        "clockwise at a round table, or left to right / top to bottom in a row. "
        "Check the result on /game under «Auto zones: n/total».",
        "While groups are missing, the manual zones are used as a fallback. "
        "Turning the option off restores the manual zones.",
        "Automatic zones are recalculated when the option is enabled or the "
        "player count changes; they are not saved in the filter settings.",
    ])
    pdf.callout(
        "If a row is not detected",
        "Check that all four edges are visible, that the cards are not too small "
        "and that they contrast with the table. Avoid glare and transparent "
        "sleeves. As a fallback, adjust the manual zones.",
    )
    pdf.h2("Locked cards (lock)")
    pdf.bullets([
        "Only cards in the minion zone are evaluated: crypt cards are not "
        "tapped.",
        "The tilt is measured against the zone angle: ≈90° = locked; back in "
        "place = unlocked. Evaluation is continuous: a card can lock or unlock "
        "at any point in the game.",
        "The state shows on /game (locked/unlocked per player) and in the log.",
    ])
    pdf.h2("Influence (crypt → minions)")
    pdf.p(
        "When an identified card moves from the crypt zone to the minions zone "
        "(influence completed on the real table), the Log shows «influence "
        "completed» and the vampire's capacity is deducted from the pool, which "
        "is the cost of a full influence. The player card shows how many "
        "influences they have completed and the pool spent."
    )
    pdf.bullets([
        "The card must be identified by the embedding for the cost to be "
        "charged; if it has no identity yet, moving to minions does not deduct "
        "pool.",
        "Vampire capacity is between 1 and 11 and is the cost of a full "
        "influence. The plugin cannot read the blood counters on a face-down "
        "card, so it deducts the capacity when the card appears in minions, not "
        "transfer by transfer.",
        "If the card returns to the crypt and stays there for a few seconds, the "
        "detector re-arms for another influence. A brief zone flicker does not "
        "charge twice.",
        "The pool can also be adjusted by hand from the web for bleeds and other "
        "effects.",
    ])
    pdf.h2("Pool and elimination")
    pdf.bullets([
        "The pool is controlled from the web: the plugin cannot read blood "
        "counters or pool through the camera.",
        "A completed influence automatically deducts the vampire's capacity; "
        "other changes are adjusted by hand.",
        "If a pool reaches 0, the player is marked as ousted, with a label and a "
        "dimmed look. If you raise it again (say, an accidental tap), it "
        "reactivates on its own.",
    ])
    pdf.h2("Log")
    pdf.p(
        "On /game you get the latest events with their age: locks and unlocks, "
        "crypt-to-minions moves, pool changes and eliminations/reactivations."
    )


def s10_troubleshooting(pdf: ManualEN):
    pdf.add_page()
    pdf.h1("10 · Troubleshooting")
    pdf.defitem("No card is detected",
                "Check the OBS Log (Tools → Log files) that the models load when "
                "the filter starts. Improve the lighting and fill more of the "
                "frame with the card. If needed, lower the confidence threshold "
                "to 0.35–0.45. Confirm «Always active» is checked.")
    pdf.defitem("False positives (it detects things that are not cards)",
                "Raise the confidence threshold; enable the «CUDA Oval/Rect "
                "classifier» and «NIS upscaling» if you have NVIDIA; slightly "
                "raise the detection interval.")
    pdf.defitem("It is slow or the PC suffers",
                "Switch from CPU to GPU (DirectML or CUDA); raise the detection "
                "interval to 0.5–1 s; disable NIS upscaling if it was on.")
    pdf.defitem("The web does not open on the PC",
                "The server starts when the filter is added to some source: "
                "check that. If another program uses port 8080, the OBS Log will "
                "show a warning.")
    pdf.defitem("The phone cannot reach the web",
                "Use the PC's IP (not «localhost») and make sure you are on the "
                "same WiFi. The first time, allow OBS in the Windows Firewall "
                "for private networks.")
    pdf.defitem("In director mode nothing shows on the ankh",
                "That is expected: nothing is drawn automatically. Approve "
                "cards from the Pending page.")
    pdf.defitem("I tap a pending card and nothing is drawn",
                "That detection ended up without an associated image. Show the "
                "card to the camera again and use the new pending entry, or draw "
                "it from Detections.")
    pdf.defitem("Lock detection is inverted",
                "Check that player's minion zone angle in the properties: it "
                "must match the real orientation of the table.")
    pdf.defitem("OCR mixes up card names",
                "Improve the light and avoid glare on the card. Image embedding "
                "is the main path and fixes most doubtful cases; OCR only runs "
                "on cards at least 200 px high.")
    pdf.defitem("A small card is not identified",
                "Move it closer or improve focus: the embedding needs a sharp "
                "card. OCR does not run below 200 px; if the embedding returns "
                "two candidates, they appear in Pending for a one-click choice.")
    pdf.defitem("Where do I check errors?",
                "Tools → Log files in OBS. The [Game], [Director], [WebServer] "
                "and [OCR] tags follow the plugin flow.")


def s11_privacy(pdf: ManualEN):
    pdf.h1("11 · Privacy and data")
    pdf.p(
        "All processing is local: the camera, the detections and the web run on "
        "your machine, and the plugin sends nothing to the internet. The only "
        "data it generates are its configuration and the optional director mode "
        "feedback, in:"
    )
    pdf.mono_block([
        "%AppData%\\obs-studio\\plugin_config\\vtes-card-scanner\\",
        "    (config, copied card database and optional feedback\\ folder)",
    ])
    pdf.p("You can delete that folder whenever you want.")


def s12_credits(pdf: ManualEN):
    pdf.h1("12 · Credits and license")
    pdf.bullets([
        "Project: VTES Card Scanner — La Garra Cifrada.",
        "Repository and issues: github.com/ChicoCifrado/VTES-for-OBS",
        "License: GNU GPL v2.0 (see the LICENSE file in the repository).",
        "Vampire: The Eternal Struggle is a trademark of Paradox Interactive. "
        "This project is unofficial fan-made content.",
    ])
    pdf.p(f"Manual for {VERSION} ({DATE.lower()}).", size=9.5, color=base.GREY)


def main():
    make_assets_en()
    pdf = ManualEN()
    pdf.set_title("VTES Card Scanner — User manual")
    pdf.set_author("La Garra Cifrada")
    pdf.set_creator("build_manual_en.py (fpdf2)")
    cover(pdf)
    toc(pdf)
    s1_what(pdf)
    s2_requirements(pdf)
    s3_installation(pdf)
    s4_getting_started(pdf)
    s5_overlay(pdf)
    s6_settings(pdf)
    s7_web(pdf)
    s8_director(pdf)
    s9_game_state(pdf)
    s10_troubleshooting(pdf)
    s11_privacy(pdf)
    s12_credits(pdf)
    pdf.output(str(OUT))
    print(f"PDF generado: {OUT} ({pdf.pages_count} páginas)")


if __name__ == "__main__":
    main()
