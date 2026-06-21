#ifndef VTES_OCR_HPP
#define VTES_OCR_HPP

#include <string>
#include <vector>
#include <opencv2/core.hpp>

struct VTESCardNameEntry {
    std::string id;
    std::string printed_name;
    std::string full_name;
    std::string normalized;
};

// Opaque forward declaration of Tesseract's TessBaseAPI
struct TessBaseAPI;

class VtesOcrReader {
public:
    VtesOcrReader();
    ~VtesOcrReader();

    bool init(const std::string& tessdata_path,
              const std::vector<VTESCardNameEntry>& card_names);

    bool is_initialized() const { return initialized_; }

    static std::string normalize(const std::string& s);

    bool recognize(const cv::Mat& card_bgr,
                   std::string& out_name,
                   std::string& out_id,
                   float& out_confidence);

private:
    bool initialized_ = false;
    TessBaseAPI* tess_ = nullptr;
    void* tess_lib_ = nullptr;

    // Tesseract C API function pointers (loaded at runtime)
    TessBaseAPI* (*fnCreate)() = nullptr;
    void (*fnDelete)(TessBaseAPI*) = nullptr;
    int (*fnInit3)(TessBaseAPI*, const char*, const char*) = nullptr;
    void (*fnSetImage)(TessBaseAPI*, const unsigned char*, int, int, int, int) = nullptr;
    char* (*fnGetUTF8Text)(TessBaseAPI*) = nullptr;
    void (*fnDeleteText)(char*) = nullptr;
    void (*fnSetVariable)(TessBaseAPI*, const char*, const char*) = nullptr;
    void (*fnClear)(TessBaseAPI*) = nullptr;
    void (*fnEnd)(TessBaseAPI*) = nullptr;

    bool loadTesseractDLL();

    std::vector<VTESCardNameEntry> card_names_;

    cv::Mat extractNameRegion(const cv::Mat& card_bgr);
    cv::Mat preprocessForOcr(const cv::Mat& region);
    std::tuple<std::string, std::string, float> fuzzyMatch(const std::string& ocr_text);
};

#endif
