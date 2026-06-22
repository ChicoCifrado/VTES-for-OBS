// Define NOMINMAX before any Windows header (obs-module.h may pull in windows.h)
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#endif

#include <obs-module.h>
#include "plugin-support.h"
#include "vtes_ocr.hpp"
#include "vtes_api_lookup.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <cctype>
#include <string>
#include <cmath>
#include <cstring>
#include <regex>
#include <tuple>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

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

    fnSetVariable(tess_, "tessedit_char_whitelist",
                  "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789',-.!?:/&() ");
    fnSetVariable(tess_, "load_system_dawg", "false");
    fnSetVariable(tess_, "load_freq_dawg", "false");
    fnSetVariable(tess_, "classify_bln_numeric_mode", "0");
    fnSetVariable(tess_, "tessedit_do_invert", "0");

    initialized_ = true;
    obs_log(LOG_INFO, "[OCR] Tesseract initialized (%zu card names loaded)",
            card_names_.size());
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

    // Step 1: visually locate the card name region (instead of hardcoded crop)
    cv::Mat name_region = detectNameRegion(card_bgr, card_type_hint);
    if (name_region.empty()) return false;

    cv::Mat processed = preprocessForOcr(name_region);
    if (processed.empty()) return false;

    fnSetImage(tess_, processed.data, processed.cols, processed.rows, 1, (int)processed.step);

    char* text = fnGetUTF8Text(tess_);
    if (!text) return false;

    std::string ocr_text(text);
    fnDeleteText(text);

    ocr_text.erase(std::remove(ocr_text.begin(), ocr_text.end(), '\n'), ocr_text.end());
    ocr_text.erase(std::remove(ocr_text.begin(), ocr_text.end(), '\r'), ocr_text.end());

    if (ocr_text.empty()) return false;

    obs_log(LOG_INFO, "[OCR] Raw Tesseract: \"%s\" (%zu chars)", ocr_text.c_str(), ocr_text.size());

    // Step 2: validate that the first character is uppercase
    // Trim leading whitespace before checking
    auto first_char = std::find_if(ocr_text.begin(), ocr_text.end(),
        [](unsigned char c) { return !std::isspace(c); });
    if (first_char == ocr_text.end() || !std::isupper((unsigned char)*first_char)) {
        obs_log(LOG_INFO, "[OCR] Rejected: first char '%c' not uppercase",
                first_char != ocr_text.end() ? *first_char : '?');
        return false;
    }

    obs_log(LOG_INFO, "[OCR] First char '%c' uppercase, continuing", *first_char);

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
        if (total > 55) {
            obs_log(LOG_INFO, "[OCR] Rejected: too many chars (%d, max 55)", total);
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

    auto [matched_id, matched_name, score] = fuzzyMatch(ocr_text);
    obs_log(LOG_DEBUG, "[OCR] FuzzyMatch best=%s score=%.3f thresh=0.30", matched_name.c_str(), score);
    if (score < 0.30f) return false;

    out_name = matched_name;
    out_id = matched_id;
    out_confidence = score;
    return true;
}

// ─── Visual name region detection ────────────────────────────────────
// VTES card layout:
//   Non-Vamp:  type banner (~top 8%) → name (~10-18%) → art → text
//   Vampire:   type banner (~top 8%) → oval portrait (~8-53%) → name (~53-60%)
//
// Uses horizontal projection of vertical Sobel edges to locate the text
// band. Text rows produce a dense cluster of vertical edges (letter
// strokes) — this is far more selective than raw pixel stddev, which
// also responds to card art and background texture.
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

    // Expected name zone (fraction of card height)
    // Narrow bands that exclude type banner (<10%) and card art (>17%/60%)
    bool is_vampire = (!card_type_hint.empty() && card_type_hint == "Vampire");
    int zone_top, zone_bot;
    if (is_vampire) {
        zone_top = h * 53 / 100;  // 53%-60% for vampire names
        zone_bot = h * 60 / 100;
    } else {
        zone_top = h * 10 / 100;  // 10%-17% for non-vamp names
        zone_bot = h * 17 / 100;
    }
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
        if (is_vampire) {
            name_top = h * 53 / 100;
            name_bottom = h * 59 / 100;
        } else {
            name_top = h * 10 / 100;
            name_bottom = h * 18 / 100;
        }
    }

    // 2px padding
    name_top = std::max(0, name_top - 2);
    name_bottom = std::min(h, name_bottom + 2);

    int name_left = (int)(w * 0.03f);
    int name_w = (int)(w * 0.94f);
    int name_h = name_bottom - name_top;

    cv::Rect name_roi(name_left, name_top, name_w, name_h);
    name_roi &= cv::Rect(0, 0, w, h);
    if (name_roi.width < 16 || name_roi.height < 4) return {};

    obs_log(LOG_INFO, "[OCR] detectNameRegion: zone=[%d,%d] search=%dpx best_density=%d → y=[%d,%d] h=%d",
            zone_top, zone_bot, zone_h, best_density, name_top, name_bottom, name_h);

    cv::Mat debug_roi = card_bgr(name_roi).clone();
    static int debug_cnt = 0;
    if (debug_cnt < 20) {
        std::string debug_path = "vtes_ocr_name_" + std::to_string(debug_cnt++) + ".png";
        cv::imwrite(debug_path, debug_roi);
        obs_log(LOG_INFO, "[OCR] detectNameRegion: y=[%d,%d] h=%d → %s",
                name_top, name_bottom, name_h, debug_path.c_str());
    }
    return debug_roi;
}

cv::Mat VtesOcrReader::preprocessForOcr(const cv::Mat& region)
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

    // Step 2: 3x upscale with Lanczos (sharper than INTER_CUBIC for text)
    cv::Mat upscaled;
    cv::resize(enhanced, upscaled, cv::Size(region.cols * 3, region.rows * 3),
               0, 0, cv::INTER_LANCZOS4);

    // Step 3: light blur to remove upscale noise
    cv::Mat blurred;
    cv::GaussianBlur(upscaled, blurred, cv::Size(3, 3), 0);

    // Step 4: Adaptive threshold instead of global OTSU
    // Works better for variable lighting on webcam captures
    cv::Mat binary;
    cv::adaptiveThreshold(blurred, binary, 255,
                          cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                          cv::THRESH_BINARY_INV,
                          31,  // block size (odd, ~3 × font size)
                          6);  // subtract constant

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

    // Restore PSM to SINGLE_BLOCK for subsequent name reads
    fnSetVariable(tess_, "tessedit_pageseg_mode", "6");

    obs_log(LOG_DEBUG, "[OCR] detectVampireCapacity: result=%d", result);
    return result;
}

std::string VtesOcrReader::normalize(const std::string& s)
{
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if (std::isalnum(static_cast<unsigned char>(c))) {
            out += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
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

    for (const auto& entry : card_names_) {
        const std::string& candidate = entry.normalized;

        // ─── Exact match ────────────────────────────────────────────────
        if (candidate == ocr_norm) {
            best_score = 1.0f;
            best_id = entry.id;
            best_name = entry.printed_name;
            break;
        }

        size_t m = ocr_norm.size();
        size_t n = candidate.size();
        size_t max_len = std::max(m, n);
        size_t min_len = std::min(m, n);

        // ─── Substring match (heuristic: OCR may get extra/missing chars) ─
        if (candidate.find(ocr_norm) != std::string::npos ||
            ocr_norm.find(candidate) != std::string::npos) {
            float score = (float)min_len / (float)max_len;
            if (score > best_score) {
                best_score = score;
                best_id = entry.id;
                best_name = entry.printed_name;
            }
        }

        if (min_len < 3) continue;
        if (max_len > min_len * 3) continue;

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
        float score = 1.0f - (float)dist / (float)max_len;

        if (score > best_score) {
            best_score = score;
            best_id = entry.id;
            best_name = entry.printed_name;
        }
    }

    return {best_id, best_name, best_score};
}
