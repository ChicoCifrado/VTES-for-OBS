// Define NOMINMAX before any Windows header (obs-module.h may pull in windows.h)
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#endif

#include <obs-module.h>
#include "plugin-support.h"
#include "vtes_ocr.hpp"
#include "vtes_api_lookup.hpp"
#include "nis_upscaler.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <cctype>
#include <string>
#include <cmath>
#include <cstring>
#include <regex>
#include <tuple>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <filesystem>

// ─── Fuzzy match tuning ───────────────────────────────────────────────────
// Umbral de aceptacion del fuzzy match. Medido sobre 4149 nombres reales
// simulando errores de OCR (0-3 ediciones): con el umbral anterior de 0.30
// se aceptaban entre 0.8% y 11.7% de falsos positivos; a 0.70 los falsos
// positivos con score alto caen a 1-2 de 120 casos por nivel de error.
// No subir por encima de 0.80: los nombres con acentos que OCR confunde
// (Francois/François) puntuan justo en esa franja.
static constexpr float kFuzzyMatchThreshold = 0.70f;

// Longitud maxima razonable de un nombre impreso. 195 de 4149 cartas (4.7%)
// superan 25 caracteres tras quitar lo no imprimible; el maximo real es 44
// ("Malkavian Derangement: Alternate Personality"). El filtro antiguo de 25
// descartaba todas esas cartas antes de intentar el match.
static constexpr int kMaxNameChars = 48;

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

// ─── Debug de imagenes OCR ────────────────────────────────────────────────
// Guarda los primeros recortes de cada sesion para poder inspeccionar que ve
// realmente Tesseract (banda detectada, salida del NIS y binarizado final).
// Antes se escribia con ruta relativa en el CWD, que en
// C:\Program Files\obs-studio\bin\64bit falla en silencio sin admin; ahora
// va a %APPDATA%\obs-studio\plugin_config\vtes-card-scanner\ocr_debug.
static constexpr int kMaxOcrDebugSaves = 20;
static int g_ocr_debug_saves = 0;

static std::string ocrDebugDir()
{
    static std::string dir;
    if (dir.empty()) {
#ifdef _WIN32
        char* appdata = nullptr;
        size_t len = 0;
        if (_dupenv_s(&appdata, &len, "APPDATA") == 0 && appdata && *appdata) {
            dir = std::string(appdata) +
                  "\\obs-studio\\plugin_config\\vtes-card-scanner\\ocr_debug";
            free(appdata);
        } else {
            dir = "ocr_debug";
        }
#else
        dir = "/tmp/vtes_ocr_debug";
#endif
    }
    return dir;
}

static void saveOcrDebugImage(const std::string& file_name, const cv::Mat& img)
{
    if (img.empty()) return;
    const std::string dir = ocrDebugDir();
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    cv::imwrite(dir + "/" + file_name, img);
}

VtesOcrReader::VtesOcrReader() {}

VtesOcrReader::~VtesOcrReader()
{
    if (tess_ && fnDelete) fnDelete(tess_);
#ifdef _WIN32
    if (tess_lib_) FreeLibrary((HMODULE)tess_lib_);
#else
    if (tess_lib_) dlclose(tess_lib_);
#endif
}

bool VtesOcrReader::loadTesseractDLL()
{
    if (tess_lib_) return true;

#ifdef _WIN32
    // Set the DLL directory to the Tesseract install so dependencies resolve
    const char* tess_dirs[] = {
        "C:\\Program Files\\Tesseract-OCR",
        "C:\\Program Files (x86)\\Tesseract-OCR",
        ""
    };
    HMODULE lib = nullptr;
    for (const char* dir : tess_dirs) {
        if (dir[0]) SetDllDirectoryA(dir);
        lib = LoadLibraryA("libtesseract-5.dll");
        if (dir[0]) SetDllDirectoryA(NULL);
        if (lib) {
            obs_log(LOG_INFO, "[OCR] Loaded Tesseract DLL from: %s", dir[0] ? dir : "PATH");
            break;
        }
    }

    if (!lib) {
        // Try alternative DLL names
        const char* alt_names[] = {"tesseract.dll", "libtesseract.dll"};
        for (const char* name : alt_names) {
            lib = LoadLibraryA(name);
            if (lib) {
                obs_log(LOG_INFO, "[OCR] Loaded Tesseract DLL: %s (via PATH)", name);
                break;
            }
        }
    }

    if (!lib) {
        obs_log(LOG_WARNING, "[OCR] Tesseract DLL not found");
        return false;
    }
    tess_lib_ = lib;

    fnCreate = (TessBaseAPI*(*)())GetProcAddress(lib, "TessBaseAPICreate");
    fnDelete = (void(*)(TessBaseAPI*))GetProcAddress(lib, "TessBaseAPIDelete");
    fnInit3 = (int(*)(TessBaseAPI*, const char*, const char*))GetProcAddress(lib, "TessBaseAPIInit3");
    fnSetImage = (void(*)(TessBaseAPI*, const unsigned char*, int, int, int, int))GetProcAddress(lib, "TessBaseAPISetImage");
    fnGetUTF8Text = (char*(*)(TessBaseAPI*))GetProcAddress(lib, "TessBaseAPIGetUTF8Text");
    fnDeleteText = (void(*)(char*))GetProcAddress(lib, "TessDeleteText");
    fnSetVariable = (void(*)(TessBaseAPI*, const char*, const char*))GetProcAddress(lib, "TessBaseAPISetVariable");
    fnClear = (void(*)(TessBaseAPI*))GetProcAddress(lib, "TessBaseAPIClear");
    fnEnd = (void(*)(TessBaseAPI*))GetProcAddress(lib, "TessBaseAPIEnd");
#else
    tess_lib_ = dlopen("libtesseract.so.5", RTLD_NOW | RTLD_LOCAL);
    if (!tess_lib_) tess_lib_ = dlopen("libtesseract.so", RTLD_NOW | RTLD_LOCAL);
    if (!tess_lib_) return false;

    fnCreate = (TessBaseAPI*(*)())dlsym(tess_lib_, "TessBaseAPICreate");
    fnDelete = (void(*)(TessBaseAPI*))dlsym(tess_lib_, "TessBaseAPIDelete");
    fnInit3 = (int(*)(TessBaseAPI*, const char*, const char*))dlsym(tess_lib_, "TessBaseAPIInit3");
    fnSetImage = (void(*)(TessBaseAPI*, const unsigned char*, int, int, int, int))dlsym(tess_lib_, "TessBaseAPISetImage");
    fnGetUTF8Text = (char*(*)(TessBaseAPI*))dlsym(tess_lib_, "TessBaseAPIGetUTF8Text");
    fnDeleteText = (void(*)(char*))dlsym(tess_lib_, "TessDeleteText");
    fnSetVariable = (void(*)(TessBaseAPI*, const char*, const char*))dlsym(tess_lib_, "TessBaseAPISetVariable");
    fnClear = (void(*)(TessBaseAPI*))dlsym(tess_lib_, "TessBaseAPIClear");
    fnEnd = (void(*)(TessBaseAPI*))dlsym(tess_lib_, "TessBaseAPIEnd");
#endif

    if (!fnCreate || !fnDelete || !fnInit3 || !fnSetImage || !fnGetUTF8Text || !fnSetVariable) {
#ifdef _WIN32
        FreeLibrary((HMODULE)tess_lib_);
#else
        dlclose(tess_lib_);
#endif
        tess_lib_ = nullptr;
        fnCreate = nullptr;
        return false;
    }
    return true;
}

bool VtesOcrReader::init(const std::string& tessdata_path,
                         const std::vector<VTESCardNameEntry>& card_names)
{
    card_names_ = card_names;

    if (!loadTesseractDLL()) {
        obs_log(LOG_WARNING, "[OCR] Tesseract DLL not found");
        return false;
    }

    tess_ = fnCreate();
    if (!tess_) {
        obs_log(LOG_WARNING, "[OCR] Failed to create Tesseract API");
        return false;
    }

    // Use "vtes" fine-tuned model if available, otherwise "eng"
    std::string lang = "eng";
    std::string vtes_path = tessdata_path + "/vtes.traineddata";
#ifdef _WIN32
    DWORD attr = GetFileAttributesA(vtes_path.c_str());
    if (attr != INVALID_FILE_ATTRIBUTES && !(attr & FILE_ATTRIBUTE_DIRECTORY)) {
        lang = "vtes";
        obs_log(LOG_INFO, "[OCR] Using fine-tuned VTES model");
    }
#else
    struct stat st;
    if (stat(vtes_path.c_str(), &st) == 0 && st.st_size > 0) {
        lang = "vtes";
        obs_log(LOG_INFO, "[OCR] Using fine-tuned VTES model");
    }
#endif

    int rc = fnInit3(tess_, tessdata_path.c_str(), lang.c_str());
    if (rc != 0) {
        obs_log(LOG_WARNING, "[OCR] Tesseract init failed (rc=%d, path=%s, lang=%s)",
                rc, tessdata_path.c_str(), lang.c_str());
        fnDelete(tess_);
        tess_ = nullptr;
        return false;
    }

    // Whitelist ampliada: latin-1 suplementado + latin extendido A + puntuacion
    // tipografica. Medido contra vtes.json: 110 de 4149 cartas (2.7%) usan
    // caracteres fuera del ASCII antiguo (é, á, ñ, ç, ö, ł, œ, —, ™...).
    // Los escapes son bytes UTF-8 explicitos para no depender del charset de
    // ejecucion del compilador (MSVC en Windows usa la pagina de codigos
    // del sistema, no UTF-8). Tesseract espera el whitelist en UTF-8.
    fnSetVariable(tess_, "tessedit_char_whitelist",
                  "\x41\x42\x43\x44\x45\x46\x47\x48"
                  "\x49\x4A\x4B\x4C\x4D\x4E\x4F\x50"
                  "\x51\x52\x53\x54\x55\x56\x57\x58"
                  "\x59\x5A\x61\x62\x63\x64\x65\x66"
                  "\x67\x68\x69\x6A\x6B\x6C\x6D\x6E"
                  "\x6F\x70\x71\x72\x73\x74\x75\x76"
                  "\x77\x78\x79\x7A\x30\x31\x32\x33"
                  "\x34\x35\x36\x37\x38\x39\x22\x27\x2C"
                  "\x2D\x2E\x21\x3F\x3A\x2F\x26\x28"
                  "\x29\x20\xC3\x80\xC3\x81\xC3\x82\xC3\x83\xC3\x84\xC3\x85"
                  "\xC3\x86\xC3\x87\xC3\x88\xC3\x89\xC3\x8A\xC3\x8B\xC3\x8C\xC3\x8D"
                  "\xC3\x8E\xC3\x8F\xC3\x90\xC3\x91\xC3\x92\xC3\x93\xC3\x94\xC3\x95"
                  "\xC3\x96\xC3\x97\xC3\x98\xC3\x99\xC3\x9A\xC3\x9B\xC3\x9C\xC3\x9D"
                  "\xC3\x9E\xC3\x9F\xC3\xA0\xC3\xA1\xC3\xA2\xC3\xA3\xC3\xA4\xC3\xA5"
                  "\xC3\xA6\xC3\xA7\xC3\xA8\xC3\xA9\xC3\xAA\xC3\xAB\xC3\xAC\xC3\xAD"
                  "\xC3\xAE\xC3\xAF\xC3\xB0\xC3\xB1\xC3\xB2\xC3\xB3\xC3\xB4\xC3\xB5"
                  "\xC3\xB6\xC3\xB7\xC3\xB8\xC3\xB9\xC3\xBA\xC3\xBB\xC3\xBC\xC3\xBD"
                  "\xC3\xBE\xC3\xBF\xC5\x81\xC5\x82\xC5\x92\xC5\x93\xC5\xA0\xC5\xA1"
                  "\xC5\xB9\xC5\xBA\xC5\xBB\xC5\xBC\xC5\xBD\xC5\xBE\xE2\x80\x93\xE2\x80\x94"
                  "\xC5\x84\xC4\x9B"
                  "\xE2\x84\xA2");
    fnSetVariable(tess_, "load_system_dawg", "false");
    fnSetVariable(tess_, "load_freq_dawg", "false");
    fnSetVariable(tess_, "classify_bln_numeric_mode", "0");
    fnSetVariable(tess_, "tessedit_do_invert", "0");

    // PSM 7 (single line): la banda del nombre contiene exactamente una linea
    // de texto; el analisis de pagina por defecto (PSM 3) segmenta mal estas
    // tiras cortas y baja la tasa de acierto.
    fnSetVariable(tess_, "tessedit_pageseg_mode", "7");

    initialized_ = true;
    obs_log(LOG_INFO, "[OCR] Tesseract initialized (%zu card names loaded)",
            card_names_.size());
    return true;
}

bool VtesOcrReader::init_upscaler(int device_id)
{
    upscaler_ = std::make_unique<NISUpscaler>();
    if (!upscaler_->init(device_id)) {
        obs_log(LOG_WARNING, "[OCR] NIS CUDA upscaler init failed (device=%d)", device_id);
        upscaler_.reset();
        return false;
    }
    use_upscaler_ = true;
    obs_log(LOG_INFO, "[OCR] NIS CUDA upscaler initialized (device=%d, scale=%d)",
            device_id, upscaler_->scale());
    return true;
}

bool VtesOcrReader::recognize(const cv::Mat& card_bgr,
                              std::string& out_name,
                              std::string& out_id,
                              float& out_confidence,
                              const std::string& card_type_hint)
{
    out_name.clear();
    out_id.clear();
    out_confidence = 0.0f;

    if (!initialized_ || card_bgr.empty()) return false;

    // Indice de debug para esta lectura (-1 = ya no se guardan mas)
    const int dbg_idx = (g_ocr_debug_saves < kMaxOcrDebugSaves) ? g_ocr_debug_saves++ : -1;

    // Step 1: visually locate the card name region (instead of hardcoded crop)
    cv::Mat name_region = detectNameRegion(card_bgr, card_type_hint);
    if (name_region.empty()) return false;
    if (dbg_idx >= 0)
        saveOcrDebugImage("vtes_ocr_name_" + std::to_string(dbg_idx) + ".png", name_region);

    // Step 2: optional NIS CUDA upscale (Lanczos 4x + adaptive sharpen)
    bool used_nis = false;
    if (use_upscaler_ && upscaler_) {
        cv::Mat upscaled;
        if (upscaler_->upscale(name_region, upscaled)) {
            obs_log(LOG_DEBUG, "[OCR] NIS CUDA upscale: %dx%d -> %dx%d",
                    name_region.cols, name_region.rows,
                    upscaled.cols, upscaled.rows);
            name_region = upscaled;
            used_nis = true;
            if (dbg_idx >= 0)
                saveOcrDebugImage("vtes_ocr_nis_" + std::to_string(dbg_idx) + ".png", name_region);
        }
    }

    cv::Mat processed = preprocessForOcr(name_region, used_nis);
    if (processed.empty()) return false;
    if (dbg_idx >= 0)
        saveOcrDebugImage("vtes_ocr_tess_" + std::to_string(dbg_idx) + ".png", processed);

    fnSetImage(tess_, processed.data, processed.cols, processed.rows, 1, (int)processed.step);

    char* text = fnGetUTF8Text(tess_);
    if (!text) return false;

    std::string ocr_text(text);
    fnDeleteText(text);

    ocr_text.erase(std::remove(ocr_text.begin(), ocr_text.end(), '\n'), ocr_text.end());
    ocr_text.erase(std::remove(ocr_text.begin(), ocr_text.end(), '\r'), ocr_text.end());

    if (ocr_text.empty()) return false;

    obs_log(LOG_INFO, "[OCR] Raw Tesseract: \"%s\" (%zu chars)", ocr_text.c_str(), ocr_text.size());

    // Step 2: validate that the first character is a letter (not a digit,
    // punctuation or stray noise). Acepta cualquier letra mayuscula o
    // minuscula: en la BD hay nombres que empiezan por comilla
    // ("Coach" Tyrone Soros, "Mother" Anja Giovanni) y otros en minuscula.
    auto first_char = std::find_if(ocr_text.begin(), ocr_text.end(),
        [](unsigned char c) { return !std::isspace(c); });
    if (first_char == ocr_text.end() || !std::isalpha((unsigned char)*first_char)) {
        obs_log(LOG_INFO, "[OCR] Rejected: first char '%c' not a letter",
                first_char != ocr_text.end() ? *first_char : '?');
        return false;
    }

    obs_log(LOG_INFO, "[OCR] First char '%c' is a letter, continuing", *first_char);

    // ─── Sanity filter: reject garbage OCR text ─────────────────────────
    {
        int alpha = 0, total = 0;
        char first_real = 0;
        bool repeated = true;
        for (unsigned char c : ocr_text) {
            if (!std::isspace(c)) {
                if (first_real == 0) first_real = (char)c;
                else if (c != (unsigned char)first_real) repeated = false;
                if (std::isalpha(c)) alpha++;
                total++;
            }
        }
        if (total < 3) {
            obs_log(LOG_INFO, "[OCR] Rejected: too few chars (%d)", total);
            return false;
        }
        if (total > kMaxNameChars) {
            obs_log(LOG_INFO, "[OCR] Rejected: too many chars (%d, max %d)", total, kMaxNameChars);
            return false;
        }
        if ((float)alpha / (float)total < 0.40f) {
            obs_log(LOG_INFO, "[OCR] Rejected: low alphabetic ratio %d/%d",
                    alpha, total);
            return false;
        }
        if (repeated && total > 4) {
            obs_log(LOG_INFO, "[OCR] Rejected: repeated single char '%c'",
                    first_real);
            return false;
        }
    }

    {
        CardLookupResult api_result = lookup_card_by_ocr(ocr_text);
        if (!api_result.printed_name.empty()) {
            out_name = api_result.printed_name;
            out_id = std::to_string(api_result.id);
            out_confidence = 0.95f;
            obs_log(LOG_DEBUG, "[OCR] API matched: %s (id=%s)", out_name.c_str(), out_id.c_str());
            return true;
        }
    }

    // ── Fuzzy match contra el indice de nombres ──────────────────────
    auto [matched_id, matched_name, score] = fuzzyMatch(ocr_text);
    obs_log(LOG_INFO, "[OCR] FuzzyMatch best=%s score=%.3f thresh=%.2f",
            matched_name.c_str(), score, kFuzzyMatchThreshold);
    if (score < kFuzzyMatchThreshold) {
        obs_log(LOG_INFO, "[OCR] Rejected: below fuzzy threshold");
        return false;
    }

    out_name = matched_name;
    out_id = matched_id;
    out_confidence = score;
    return true;
}

// ─── Visual name region detection ────────────────────────────────────
// VTES card layout — verificado empiricamente sobre 13 cartas reales
// descargadas de krcg.org (vampiros, masters, actions, combats):
//
//   TODOS los tipos llevan el nombre impreso en el banner superior,
//   aproximadamente en el 4-10% de la altura de la carta. Vampiros
//   incluidos: el nombre va ARRIBA del ovalo, no debajo.
//   (La creencia anterior de que los vampiros la tenian en el 53-60%
//   era incorrecta — esa franja es el arte/texto de la carta.)
//
// Usa la proyeccion horizontal de los bordes Sobel verticales para
// localizar la banda de texto. Las filas de texto producen una
// agrupacion densa de bordes verticales (trazos de letra), mucho mas
// selectiva que la desviacion tipica de pixeles, que tambien responde
// al arte de la carta y a la textura del fondo.
cv::Mat VtesOcrReader::detectNameRegion(const cv::Mat& card_bgr,
                                        const std::string& card_type_hint)
{
    if (card_bgr.empty()) return {};

    int h = card_bgr.rows;
    int w = card_bgr.cols;

    cv::Mat gray;
    if (card_bgr.channels() == 3)
        cv::cvtColor(card_bgr, gray, cv::COLOR_BGR2GRAY);
    else
        gray = card_bgr;

    // CLAHE to normalise lighting before edge detection
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    cv::Mat enhanced;
    clahe->apply(gray, enhanced);

    // Vertical Sobel: strong response at letter vertical strokes
    cv::Mat sobel;
    cv::Sobel(enhanced, sobel, CV_16S, 1, 0, 3);

    // Take absolute magnitude and convert to 8-bit
    cv::Mat abs_edges;
    cv::convertScaleAbs(sobel, abs_edges);

    // Binarise edge magnitude (keep only strong edges)
    cv::Mat edge_binary;
    cv::threshold(abs_edges, edge_binary, 24.0, 255.0, cv::THRESH_BINARY);

    // Expected name zone — ALL card types have the card name at the top-left
    // corner, just below the top border. Search the top 2-12% of the card.
    // (Previous per-type zones for vampires at 53-60% were incorrect — that's
    //  the card art / text area, not the name.)
    int zone_top = h * 2 / 100;
    int zone_bot = h * 12 / 100;
    zone_top = std::max(0, zone_top);
    zone_bot = std::min(h, zone_bot);

    // Horizontal projection of edge pixels within the expected zone
    int zone_h = zone_bot - zone_top;
    std::vector<int> proj(zone_h, 0);
    for (int y = 0; y < zone_h; y++) {
        proj[y] = cv::countNonZero(edge_binary.row(zone_top + y));
    }

    // Smooth the projection (moving average, window = 3)
    std::vector<int> smooth(zone_h, 0);
    for (int y = 0; y < zone_h; y++) {
        int sum = 0, cnt = 0;
        for (int dy = -1; dy <= 1; dy++) {
            int ny = y + dy;
            if (ny >= 0 && ny < zone_h) { sum += proj[ny]; cnt++; }
        }
        smooth[y] = sum / cnt;
    }

    // Precompute prefix sum of smoothed projection for O(1) range sums
    std::vector<int> prefix(zone_h + 1, 0);
    for (int y = 0; y < zone_h; y++)
        prefix[y + 1] = prefix[y] + smooth[y];

    // Find the sub-range within the zone with the highest average edge density
    int best_start = 0, best_end = 0, best_density = 0;
    const int min_name_h = std::max(8, h * 4 / 100);  // at least 4% of card
    const int max_name_h = std::min(zone_h, h * 14 / 100); // at most 14%

    for (int start = 0; start < zone_h; start++) {
        for (int cand_h = min_name_h; cand_h <= max_name_h && start + cand_h <= zone_h; cand_h++) {
            int total_edges = prefix[start + cand_h] - prefix[start];
            // Count rows with strong edge content
            int text_rows = 0;
            for (int y = start; y < start + cand_h; y++) {
                if (smooth[y] > w * 0.02f) text_rows++;
            }
            if (text_rows < cand_h * 3 / 4) continue;  // need ≥75% text rows
            int avg_per_row = total_edges / cand_h;
            if (avg_per_row > best_density) {
                best_density = avg_per_row;
                best_start = start;
                best_end = start + cand_h;
            }
        }
    }

    // Tighten: within the found range, trim low-edge rows from top/bottom
    if (best_density > 0) {
        int peak = 0;
        for (int y = best_start; y < best_end; y++)
            if (smooth[y] > peak) peak = smooth[y];
        int trim_thresh = peak * 25 / 100;  // 25% of peak
        while (best_start < best_end && smooth[best_start] < trim_thresh)
            best_start++;
        while (best_end > best_start && smooth[best_end - 1] < trim_thresh)
            best_end--;
        if (best_end - best_start < min_name_h)
            best_density = 0;  // trimmed too much, fallback
    }

    int name_top, name_bottom;
    if (best_density > 0) {
        name_top = zone_top + best_start;
        name_bottom = zone_top + best_end;
    } else {
        // Fallback to fixed position if no clear text band found
        name_top = h * 2 / 100;
        name_bottom = h * 14 / 100;
    }

    // Padding proporcional al alto de la carta (antes 2 px fijos): en cartas
    // grandes 2 px dejaban fuera ascendentes y descendentes del nombre.
    const int pad = std::max(2, h / 100);
    name_top = std::max(0, name_top - pad);
    name_bottom = std::min(h, name_bottom + pad);

    // El nombre impreso llega hasta el borde derecho en nombres largos.
    // Medido: 195 de 4149 cartas (4.7%) tienen nombres de mas de 25 chars,
    // el maximo 44 ("Malkavian Derangement: Alternate Personality").
    // El crop antiguo al 55% del ancho los cortaba a mitad.
    // Se ensancha al 88% y se deja que el filtro de basura de recognize()
    // descarte el ruido del borde (iconos de clan/disciplina).
    int name_left = (int)(w * 0.04f);
    int name_w = (int)(w * 0.88f);
    int name_h = name_bottom - name_top;

    cv::Rect name_roi(name_left, name_top, name_w, name_h);
    name_roi &= cv::Rect(0, 0, w, h);
    if (name_roi.width < 16 || name_roi.height < 4) return {};

    obs_log(LOG_INFO, "[OCR] detectNameRegion: zone=[%d,%d] search=%dpx best_density=%d → y=[%d,%d] h=%d",
            zone_top, zone_bot, zone_h, best_density, name_top, name_bottom, name_h);

    return card_bgr(name_roi).clone();
}

cv::Mat VtesOcrReader::preprocessForOcr(const cv::Mat& region, bool already_upscaled)
{
    if (region.empty()) return {};

    cv::Mat gray;
    if (region.channels() == 3) {
        cv::cvtColor(region, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = region.clone();
    }

    // Step 1: CLAHE contrast enhancement — boosts text vs. background
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    cv::Mat enhanced;
    clahe->apply(gray, enhanced);

    // Step 2: si NIS CUDA ya upscalo 4x, no reescalar otra vez. El codigo
    // anterior decidia por tamano (rows>80) y con una banda de nombre corta
    // (18 px -> 72 px tras NIS) aplicaba ademas un resize 3x que deshacia el
    // trabajo del NIS y dejaba el bloque del threshold minusculo para el
    // tamano real del texto.
    cv::Mat upscaled;
    if (already_upscaled) {
        upscaled = enhanced;
    } else {
        cv::resize(enhanced, upscaled, cv::Size(region.cols * 3, region.rows * 3),
                   0, 0, cv::INTER_LANCZOS4);
    }

    // Step 3: light blur to remove upscale noise
    cv::Mat blurred;
    cv::GaussianBlur(upscaled, blurred, cv::Size(3, 3), 0);

    // Step 4: Adaptive threshold instead of global OTSU
    // Works better for variable lighting on webcam captures.
    // Bloque proporcional al alto de la imagen: un texto de ~50-70 px necesita
    // un bloque mayor que uno de ~36 px (el valor fijo 31 anterior era valido
    // solo para la escala del resize 3x).
    int block = ((int)upscaled.rows / 2) | 1;
    if (block < 25) block = 25;
    if (block > 61) block = 61;
    cv::Mat binary;
    cv::adaptiveThreshold(blurred, binary, 255,
                          cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                          cv::THRESH_BINARY_INV,
                          block,  // block size (odd, ~2-4 x stroke)
                          6);     // subtract constant

    // Step 5: median blur to remove salt-and-pepper noise
    cv::Mat denoised;
    cv::medianBlur(binary, denoised, 3);

    // Step 6: connect broken characters with a 1×2 close kernel
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 2));
    cv::morphologyEx(denoised, denoised, cv::MORPH_CLOSE, kernel);

    return denoised;
}

// ─── Vampire oval portrait detection ──────────────────────────────────
// VTES vampire cards have a painted oval portrait frame in the center-top.
// Library and Master cards have rectangular art frames with straight edges.
// Uses HoughLinesP: an oval produces very few straight line segments,
// while a rectangle frame produces many.
bool VtesOcrReader::hasVampireOval(const cv::Mat& card_bgr)
{
    if (card_bgr.empty()) return false;

    int h = card_bgr.rows;
    int w = card_bgr.cols;

    // Portrait area: center-top of the card
    int px = (int)(w * 0.12f);
    int py = (int)(h * 0.08f);
    int pw = (int)(w * 0.76f);
    int ph = (int)(h * 0.45f);

    cv::Rect roi(px, py, pw, ph);
    roi &= cv::Rect(0, 0, w, h);
    if (roi.width < 30 || roi.height < 30) return false;

    cv::Mat portrait = card_bgr(roi);

    // Edge detection
    cv::Mat gray, edges;
    cv::cvtColor(portrait, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, gray, cv::Size(3, 3), 0);
    cv::Canny(gray, edges, 50, 150);

    // Count straight line segments in the portrait area
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(edges, lines, 1, CV_PI / 180, 40, 20, 10);

    int line_count = (int)lines.size();
    obs_log(LOG_DEBUG, "[OCR] hasVampireOval: %d lines in portrait area", line_count);

    // Vampires (oval portrait): few straight lines (< 10)
    // Non-vampires (rectangular art): many straight lines (≥ 10)
    return line_count < 10;
}

// ─── Vampire capacity detection ──────────────────────────────────────
// The capacity number appears inside a red circle in the bottom-right
// corner of VTES vampire cards. Returns -1 if not found.
int VtesOcrReader::detectVampireCapacity(const cv::Mat& card_bgr)
{
    if (!initialized_ || card_bgr.empty()) return -1;

    int h = card_bgr.rows;
    int w = card_bgr.cols;

    // Bottom-right quadrant
    int cx = (int)(w * 0.62f);
    int cy = (int)(h * 0.65f);
    int cw = w - cx;
    int ch = h - cy;

    cv::Rect cap_roi(cx, cy, cw, ch);
    cap_roi &= cv::Rect(0, 0, w, h);
    if (cap_roi.width < 12 || cap_roi.height < 12) return -1;

    cv::Mat cap_region = card_bgr(cap_roi);

    // Look for red circular region (the blood pool icon)
    cv::Mat hsv;
    cv::cvtColor(cap_region, hsv, cv::COLOR_BGR2HSV);

    cv::Mat red_mask1, red_mask2;
    cv::inRange(hsv, cv::Scalar(0, 50, 50), cv::Scalar(10, 255, 255), red_mask1);
    cv::inRange(hsv, cv::Scalar(160, 50, 50), cv::Scalar(180, 255, 255), red_mask2);
    cv::Mat red_mask = red_mask1 | red_mask2;

    cv::GaussianBlur(red_mask, red_mask, cv::Size(5, 5), 0);

    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(red_mask, circles, cv::HOUGH_GRADIENT, 1.5,
                     cap_region.rows / 4, 50, 25, 5, cap_region.rows / 2);

    if (circles.empty()) {
        obs_log(LOG_DEBUG, "[OCR] detectVampireCapacity: no red circle found");
        return -1;
    }

    obs_log(LOG_DEBUG, "[OCR] detectVampireCapacity: %zu circle(s) found", circles.size());

    // Save current PSM, set to single-char mode for digit reading
    fnSetVariable(tess_, "tessedit_pageseg_mode", "10");

    int result = -1;
    for (const auto& c : circles) {
        int c_x = (int)c[0];
        int c_y = (int)c[1];
        int c_r = (int)c[2];
        int inner_r = std::max(c_r / 2, 4);

        int inner_x = std::max(0, c_x - inner_r);
        int inner_y = std::max(0, c_y - inner_r);
        int inner_w = std::min(cap_region.cols - inner_x, inner_r * 2);
        int inner_h = std::min(cap_region.rows - inner_y, inner_r * 2);

        cv::Rect inner(inner_x, inner_y, inner_w, inner_h);
        if (inner.width < 6 || inner.height < 6) continue;

        cv::Mat digit_area = cap_region(inner);

        cv::Mat gray, binary;
        cv::cvtColor(digit_area, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, gray, cv::Size(3, 3), 0);
        cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

        cv::Mat upscaled;
        cv::resize(binary, upscaled, cv::Size(), 3.0, 3.0, cv::INTER_NEAREST);

        fnSetImage(tess_, upscaled.data, upscaled.cols, upscaled.rows, 1, (int)upscaled.step);
        char* text = fnGetUTF8Text(tess_);
        if (!text) continue;

        std::string num(text);
        fnDeleteText(text);

        num.erase(std::remove_if(num.begin(), num.end(),
            [](unsigned char c) { return std::isspace(c) || c == '\n' || c == '\r'; }), num.end());

        if (num.empty()) continue;

        obs_log(LOG_DEBUG, "[OCR] detectVampireCapacity: raw='%s'", num.c_str());

        try {
            int cap = std::stoi(num);
            if (cap >= 0 && cap <= 12) {
                result = cap;
                break;
            }
        } catch (...) {}
    }

    // Restore PSM 7 (single line) for subsequent name reads
    fnSetVariable(tess_, "tessedit_pageseg_mode", "7");

    obs_log(LOG_DEBUG, "[OCR] detectVampireCapacity: result=%d", result);
    return result;
}

// ─── Normalizacion sin acentos ───────────────────────────────────────
// Convierte UTF-8 (hasta 2 bytes por caracter en latin) a ASCII base,
// para que "Francois" y "François" normalizen igual. Solo cubre latin-1
// suplementado y latin extendido A, que es todo lo que aparece en los
// nombres de cartas de vtes.json.
static unsigned char foldLatin1(unsigned char c1, unsigned char c2, bool& consumed_two)
{
    consumed_two = false;
    // latin-1 suplementado: C3 80 .. C3 BF
    if (c1 == 0xC3) {
        consumed_two = true;
        switch (c2) {
            case 0x80: case 0x81: case 0x82: case 0x83: case 0x84: case 0x85:
            case 0x86: return 'A';                       // À Á Â Ã Ä Å Æ
            case 0x87: return 'C';                       // Ç
            case 0x88: case 0x89: case 0x8A: case 0x8B: return 'E'; // È É Ê Ë
            case 0x8C: case 0x8D: case 0x8E: case 0x8F: return 'I'; // Ì Í Î Ï
            case 0x90: return 'D';                       // Ð
            case 0x91: return 'N';                       // Ñ
            case 0x92: case 0x93: case 0x94: case 0x95:
            case 0x96: return 'O';                       // Ò Ó Ô Õ Ö ×
            case 0x98: return 'O';                       // Ø
            case 0x99: case 0x9A: case 0x9B: return 'U'; // Ù Ú Û
            case 0x9C: return 'U';                       // Ü
            case 0x9D: return 'Y';                       // Ý
            case 0x9E: return 'T';                       // Þ (thorn)
            case 0x9F: return 's';                       // ß (ssharp -> ss)
            case 0xA0: case 0xA1: case 0xA2: case 0xA3:
            case 0xA4: case 0xA5: case 0xA6: return 'a';
            case 0xA7: return 'c';                       // ç
            case 0xA8: case 0xA9: case 0xAA: case 0xAB: return 'e';
            case 0xAC: case 0xAD: case 0xAE: case 0xAF: return 'i';
            case 0xB0: return 'd';                       // ð
            case 0xB1: return 'n';                       // ñ
            case 0xB2: case 0xB3: case 0xB4: case 0xB5: case 0xB6:
            case 0xB7: return 'o';
            case 0xB8: return 'o';                       // ø
            case 0xB9: case 0xBA: case 0xBB: return 'u';
            case 0xBC: return 'u';                       // ü
            case 0xBD: return 'y';                       // ý
            case 0xBE: return 't';                       // þ
            case 0xBF: return 'y';                       // ÿ
            default: return '?';
        }
    }
    // latin extendido A (C4 xx / C5 xx): polaco, checo, eslovaco, esloveno.
    // Cubre C4 80..C4 BF y C5 80..C5 BF salvo los pocos puntos que caen en
    // simbolos no alfabeticos, que devuelven '?' y se descartan.
    // Cubre C4 80..C4 BF y C5 80..C5 BF salvo los pocos puntos que caen en
    // simbolos no alfabeticos, que devuelven '?' y se descartan.
    if (c1 == 0xC4) {
        consumed_two = true;
        switch (c2) {
            case 0x80: case 0x81: case 0x82: return 'A';
            case 0x84: return 'A';
            case 0x86: return 'C';
            case 0x87: case 0x88: case 0x89: return 'c';
            case 0x8A: case 0x8B: return 'C';
            case 0x8C: case 0x8D: return 'D';
            case 0x90: return 'D';
            case 0x91: case 0x92: return 'd';
            case 0x96: case 0x97: return 'E';
            case 0x98: case 0x99: return 'E';
            case 0x9A: case 0x9B: return 'E';
            case 0x9C: case 0x9D: return 'e';
            case 0xA0: case 0xA1: case 0xA2: return 'G';
            case 0xA6: case 0xA7: return 'G';
            case 0xA8: case 0xA9: case 0xAA: return 'I';
            case 0xAB: case 0xAC: return 'I';
            case 0xAD: case 0xAE: case 0xAF: return 'i';
            case 0xB0: case 0xB1: return 'I';
            case 0xB2: case 0xB3: return 'i';
            case 0xB4: case 0xB5: case 0xB6: return 'K';
            case 0xB7: case 0xB8: return 'k';
            case 0xB9: case 0xBA: return 'L';
            case 0xBB: return 'l';
            case 0xBC: case 0xBD: case 0xBE: return 'l';
            case 0xBF: return 'l';
            default: return '?';
        }
    }
    if (c1 == 0xC5) {
        consumed_two = true;
        switch (c2) {
            case 0x81: return 'L'; case 0x82: return 'l';
            case 0x84: return 'N'; case 0x85: return 'n';
            case 0x87: return 'N'; case 0x88: return 'n';
            case 0x92: return 'O'; case 0x93: return 'o';
            case 0x94: return 'O'; case 0x95: return 'o';
            case 0x98: case 0x99: case 0x9A: return 'R';
            case 0x9B: case 0x9C: case 0x9D: return 'r';
            case 0xA0: return 'S'; case 0xA1: return 's';
            case 0xA2: return 'T'; case 0xA3: return 't';
            case 0xA4: case 0xA5: return 'T';
            case 0xA6: case 0xA7: return 't';
            case 0xB9: case 0xBA: return 'Z';
            case 0xBB: case 0xBC: return 'z';
            case 0xBD: case 0xBE: return 'Z';
            default: return '?';
        }
    }
    if (c1 == 0xE2 && c2 == 0x80) { consumed_two = true; return '-'; } // – —
    return 0;
}

std::string VtesOcrReader::normalize(const std::string& s)
{
    // Normaliza a minusculas sin acentos conservando solo alfanumericos.
    // Opera sobre bytes UTF-8: primero pliega latin-1/extendido a ASCII
    // y luego filtra. "François Villon" -> "francoisvillon", igual que
    // "Francois Villon", para que el fuzzy match no las trate como distintas.
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        unsigned char c = (unsigned char)s[i];
        if (c < 0x80) {
            if (std::isalnum(c))
                out += (char)std::tolower(c);
            continue;
        }
        // secuencia UTF-8 multibyte
        unsigned char c2 = (i + 1 < s.size()) ? (unsigned char)s[i + 1] : 0;
        bool two = false;
        unsigned char folded = foldLatin1(c, c2, two);
        if (two) {
            i += 1;
            if (folded != '?')
                out += (char)std::tolower(folded);
            continue;
        }
        // em-dash / en-dash / trade mark: E2 80 93/94 y E2 84 A2
        if (c == 0xE2 && i + 2 < s.size()) {
            unsigned char b2 = (unsigned char)s[i + 1], b3 = (unsigned char)s[i + 2];
            if (b2 == 0x80 && (b3 == 0x93 || b3 == 0x94)) { i += 2; continue; }
            if (b2 == 0x84 && b3 == 0xA2) { i += 2; continue; }
        }
        // cualquier otro multibyte (comillas tipograficas, CJK...) se descarta
        while (i + 1 < s.size() && ((unsigned char)s[i + 1] & 0xC0) == 0x80) i++;
    }
    return out;
}

std::tuple<std::string, std::string, float>
VtesOcrReader::fuzzyMatch(const std::string& ocr_text)
{
    std::string ocr_norm = normalize(ocr_text);
    if (ocr_norm.empty()) return {"", "", 0.0f};

    std::string best_id;
    std::string best_name;
    float best_score = 0.0f;

    // Recorrido completo sobre el indice. Se probo un indice por longitud y
    // se descarto: ademas de no acelerar (26 ms -> 31 ms medido en Python),
    // cambiaba el orden de recorrido y con ello el desempate cuando varias
    // cartas empatan a igual score (p.ej. "vampire" puntua 0.571429 para
    // Ramiro, Inspire, Rampage y Respire). El resultado es correcto en ambos
    // casos, pero recorrer en el orden de la BD es determinista y evita
    // discusiones. El filtro de longitud de abajo ya recorta la inmensa
    // mayoria de candidatos antes de tocar Levenshtein.
    for (const auto& entry : card_names_) {
        const std::string& candidate = entry.normalized;

        // Exact match: corta aqui, no puede haber nada mejor.
        if (candidate == ocr_norm) {
            best_id = entry.id;
            best_name = entry.printed_name;
            best_score = 1.0f;
            break;
        }

        const size_t n = candidate.size();
        const size_t m = ocr_norm.size();
        const size_t max_len = std::max(m, n);
        const size_t min_len = std::min(m, n);

        // Substring match (heuristic: OCR may get extra/missing chars).
        // OJO: este atajo producia falsos positivos masivos con nombres
        // cortos ("Sela" vs "Esau" puntua 0.5 por substring). Se exige que
        // el termino corto tenga al menos 5 caracteres para que cuente.
        if (min_len >= 5 &&
            (candidate.find(ocr_norm) != std::string::npos ||
             ocr_norm.find(candidate) != std::string::npos)) {
            float score = (float)min_len / (float)max_len;
            if (score > best_score) {
                best_score = score;
                best_id = entry.id;
                best_name = entry.printed_name;
            }
        }

        if (min_len < 3) continue;
        if (max_len > min_len * 3) continue;

        // Prefiltro de longitud: el score maximo posible para este candidato
        // es 1 - min_len/max_len (dos cadenas de longitudes distintas parten
        // ya con esa distancia minima). Si ni asi alcanza el mejor score
        // encontrado, no hace falta calcular Levenshtein.
        {
            float ceiling = 1.0f - (float)min_len / (float)max_len;
            float len_ratio = (float)min_len / (float)max_len;
            if (len_ratio < 0.6f) ceiling *= len_ratio / 0.6f;
            if (ceiling <= best_score) continue;
        }

        std::vector<size_t> prev(n + 1), curr(n + 1);
        for (size_t j = 0; j <= n; j++) prev[j] = j;

        for (size_t i = 1; i <= m; i++) {
            curr[0] = i;
            for (size_t j = 1; j <= n; j++) {
                size_t cost = (ocr_norm[i - 1] == candidate[j - 1]) ? 0 : 1;
                curr[j] = std::min({prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost});
            }
            std::swap(prev, curr);
        }

        size_t dist = prev[n];
        float score = (max_len > 0) ? (1.0f - (float)dist / (float)max_len) : 0.0f;

        // Penaliza la diferencia de longitud de forma explicita: dos nombres
        // de longitudes muy distintas no deben puntuar alto solo por compartir
        // un patron Levenshtein barato (p.ej. "gasmsih" vs "samson").
        float len_ratio = (max_len > 0) ? (float)min_len / (float)max_len : 0.0f;
        if (len_ratio < 0.6f) score *= len_ratio / 0.6f;

        if (score > best_score) {
            best_score = score;
            best_id = entry.id;
            best_name = entry.printed_name;
        }
    }

    return {best_id, best_name, best_score};
}
