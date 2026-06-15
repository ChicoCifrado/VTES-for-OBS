#include "vtes_ocr.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <regex>
#include <tuple>

#ifdef VTES_HAVE_TESSERACT
#include <tesseract/capi.h>
#endif

VtesOcrReader::VtesOcrReader() {}

VtesOcrReader::~VtesOcrReader()
{
#ifdef VTES_HAVE_TESSERACT
    if (tess_) {
        TessBaseAPIDelete(tess_);
        tess_ = nullptr;
    }
#endif
}

bool VtesOcrReader::init(const std::string& tessdata_path,
                         const std::vector<VTESCardNameEntry>& card_names)
{
    card_names_ = card_names;

#ifdef VTES_HAVE_TESSERACT
    tess_ = TessBaseAPICreate();
    if (!tess_) {
        fprintf(stderr, "[VtesOcr] Failed to create Tesseract API\n");
        return false;
    }

    int rc = TessBaseAPIInit3(tess_, tessdata_path.c_str(), "eng");
    if (rc != 0) {
        fprintf(stderr, "[VtesOcr] Tesseract init failed (rc=%d, path=%s)\n",
                rc, tessdata_path.c_str());
        TessBaseAPIDelete(tess_);
        tess_ = nullptr;
        return false;
    }

    TessBaseAPISetVariable(tess_, "tessedit_char_whitelist",
                           "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789',-&()");
    TessBaseAPISetVariable(tess_, "load_system_dawg", "false");
    TessBaseAPISetVariable(tess_, "load_freq_dawg", "false");

    initialized_ = true;
    fprintf(stderr, "[VtesOcr] Tesseract initialized (%zu card names loaded)\n",
            card_names_.size());
    return true;
#else
    (void)tessdata_path;
    fprintf(stderr, "[VtesOcr] Tesseract not available (recompile with VTES_HAVE_TESSERACT)\n");
    return false;
#endif
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

#ifdef VTES_HAVE_TESSERACT
    // 1. Extract name region (top ~10% of card)
    cv::Mat name_region = extractNameRegion(card_bgr);
    if (name_region.empty()) return false;

    // 2. Preprocess for OCR (grayscale + threshold)
    cv::Mat processed = preprocessForOcr(name_region);
    if (processed.empty()) return false;

    // 3. Run Tesseract using raw image API (no Leptonica needed)
    // processed is 8-bit grayscale (CV_8UC1)
    TessBaseAPISetImage(tess_, processed.data, processed.cols, processed.rows,
                        1, processed.step);

    // Set the resolution so Tesseract scales properly
    char* text = TessBaseAPIGetUTF8Text(tess_);
    if (!text) return false;

    std::string ocr_text(text);
    TessDeleteText(text);

    // Clean OCR output
    ocr_text.erase(std::remove(ocr_text.begin(), ocr_text.end(), '\n'), ocr_text.end());
    ocr_text.erase(std::remove(ocr_text.begin(), ocr_text.end(), '\r'), ocr_text.end());

    if (ocr_text.empty()) return false;

    // 4. Fuzzy match against card names
    auto [matched_id, matched_name, score] = fuzzyMatch(ocr_text);
    if (score < 0.3f) return false;

    out_name = matched_name;
    out_id = matched_id;
    out_confidence = score;
    return true;
#else
    (void)card_bgr;
    return false;
#endif
}

cv::Mat VtesOcrReader::extractNameRegion(const cv::Mat& card_bgr)
{
    if (card_bgr.empty()) return {};

    int h = card_bgr.rows;
    int w = card_bgr.cols;

    // Card name is in the top ~20% of the card
    int name_top = (int)(h * 0.02f);
    int name_bottom = (int)(h * 0.20f);
    int name_h = name_bottom - name_top;
    if (name_h < 8) return {};

    int name_left = (int)(w * 0.05f);
    int name_w = (int)(w * 0.90f);

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

    // Upscale 2x for better OCR
    cv::Mat upscaled;
    cv::resize(gray, upscaled, cv::Size(region.cols * 2, region.rows * 2),
               0, 0, cv::INTER_CUBIC);

    // Adaptive threshold
    cv::Mat binary;
    cv::adaptiveThreshold(upscaled, binary, 255,
                          cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                          cv::THRESH_BINARY_INV, 31, 8);

    // Denoise
    cv::Mat denoised;
    cv::medianBlur(binary, denoised, 3);

    // Morphological close to connect broken characters
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
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

        // Exact substring match
        if (candidate.find(ocr_norm) != std::string::npos ||
            ocr_norm.find(candidate) != std::string::npos) {
            float score = (float)std::min(ocr_norm.size(), candidate.size()) /
                          (float)std::max(ocr_norm.size(), candidate.size());
            if (candidate == ocr_norm) score = 1.0f;
            if (score > best_score) {
                best_score = score;
                best_id = entry.id;
                best_name = entry.printed_name;
            }
            continue;
        }

        // Levenshtein distance
        size_t m = ocr_norm.size();
        size_t n = candidate.size();
        if (std::abs((int)m - (int)n) > (int)(m * 0.5f)) continue;

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
        size_t max_len = std::max(m, n);
        float score = 1.0f - (float)dist / (float)max_len;

        if (score > best_score) {
            best_score = score;
            best_id = entry.id;
            best_name = entry.printed_name;
        }
    }

    return {best_id, best_name, best_score};
}
