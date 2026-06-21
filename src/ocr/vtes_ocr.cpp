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
#include <algorithm>
#include <cctype>
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
                              float& out_confidence)
{
    out_name.clear();
    out_id.clear();
    out_confidence = 0.0f;

    if (!initialized_ || card_bgr.empty()) return false;

    cv::Mat name_region = extractNameRegion(card_bgr);
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

    obs_log(LOG_DEBUG, "[OCR] Raw Tesseract output: \"%s\" (%zu chars)", ocr_text.c_str(), ocr_text.size());

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

cv::Mat VtesOcrReader::extractNameRegion(const cv::Mat& card_bgr)
{
    if (card_bgr.empty()) return {};

    int h = card_bgr.rows;
    int w = card_bgr.cols;

    int name_top = (int)(h * 0.01f);
    int name_bottom = (int)(h * 0.22f);
    int name_h = name_bottom - name_top;
    if (name_h < 10) {};

    int name_left = (int)(w * 0.03f);
    int name_w = (int)(w * 0.94f);

    cv::Rect name_roi(name_left, name_top, name_w, name_h);
    name_roi &= cv::Rect(0, 0, w, h);
    if (name_roi.width < 16 || name_roi.height < 4) return {};

    return card_bgr(name_roi).clone();
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

    cv::Mat upscaled;
    cv::resize(gray, upscaled, cv::Size(region.cols * 3, region.rows * 3),
               0, 0, cv::INTER_CUBIC);

    cv::Mat blurred;
    cv::GaussianBlur(upscaled, blurred, cv::Size(3, 3), 0);

    cv::Mat binary;
    cv::threshold(blurred, binary, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

    cv::Mat denoised;
    cv::medianBlur(binary, denoised, 3);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 2));
    cv::morphologyEx(denoised, denoised, cv::MORPH_CLOSE, kernel);

    return denoised;
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
