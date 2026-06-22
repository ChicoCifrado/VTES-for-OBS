#ifndef VTES_OCR_HPP
#define VTES_OCR_HPP

#include <string>
#include <vector>
#include <opencv2/core.hpp>

#ifdef VTES_HAVE_TESSERACT
#include <tesseract/capi.h>
#endif

struct VTESCardNameEntry {
    std::string id;
    std::string printed_name;   // e.g. "Aabbt Kindred"
    std::string full_name;      // e.g. "Aabbt Kindred (G2)"
    std::string normalized;     // lowercase, no spaces/punct — for fuzzy matching
};

class VtesOcrReader {
public:
    VtesOcrReader();
    ~VtesOcrReader();

    // Initialize Tesseract. Returns true if OCR is available.
    // @param tessdata_path: path to tessdata directory (e.g. "./tessdata")
    // @param card_names: list of all VTES card names for fuzzy matching
    bool init(const std::string& tessdata_path,
              const std::vector<VTESCardNameEntry>& card_names);

    // Returns true if Tesseract was initialized successfully
    bool is_initialized() const { return initialized_; }

    // Normalize a string for fuzzy matching (public helper)
    static std::string normalize(const std::string& s);

    // Run OCR on a card crop, extract name from top region.
    // Returns the best matching card ID and name.
    // @param card_bgr: BGR card image (full card)
    // @param out_name: filled with matched card name on success
    // @param out_id: filled with matched card ID on success
    // @param out_confidence: OCR confidence (0-1)
    // @returns true if a match was found
    bool recognize(const cv::Mat& card_bgr,
                   std::string& out_name,
                   std::string& out_id,
                   float& out_confidence);

private:
    bool initialized_ = false;

#ifdef VTES_HAVE_TESSERACT
    TessBaseAPI* tess_ = nullptr;
#endif

    std::vector<VTESCardNameEntry> card_names_;

    // Crop the top portion of the card where the name is printed
    cv::Mat extractNameRegion(const cv::Mat& card_bgr);

    // Preprocess image for OCR (grayscale, threshold, denoise)
    cv::Mat preprocessForOcr(const cv::Mat& region);

    // Find the best fuzzy match among card names
    // Returns (id, name, score)
    std::tuple<std::string, std::string, float> fuzzyMatch(const std::string& ocr_text);
};

#endif // VTES_OCR_HPP
