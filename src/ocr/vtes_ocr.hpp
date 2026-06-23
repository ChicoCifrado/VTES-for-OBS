#ifndef VTES_OCR_HPP
#define VTES_OCR_HPP

#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <memory>

struct VTESCardNameEntry {
    std::string id;
    std::string printed_name;
    std::string full_name;
    std::string normalized;
};

// Opaque forward declaration of Tesseract's TessBaseAPI
struct TessBaseAPI;
class NISUpscaler;

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
                   float& out_confidence,
                   const std::string& card_type_hint = "");

    bool init_upscaler(int device_id = 0);
    void set_upscaler_enabled(bool enabled) { use_upscaler_ = enabled; }
    bool is_upscaler_enabled() const { return use_upscaler_; }

    // ─── Vampire card validation ────────────────────────────────────
    // Returns true if the card has an oval portrait (vampire indicator)
    static bool hasVampireOval(const cv::Mat& card_bgr);
    // OCR the capacity number from the bottom-right of a vampire card.
    // Returns -1 if not found or unreadable.
    int detectVampireCapacity(const cv::Mat& card_bgr);

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

    std::unique_ptr<NISUpscaler> upscaler_;
    bool use_upscaler_ = false;

    std::vector<VTESCardNameEntry> card_names_;

    // Visually locate the name region on a perspective-corrected card
    // card_type_hint: "Vampire", "Master", etc. from classifier — used to
    // skip the card-type banner and target the actual card name below it.
    cv::Mat detectNameRegion(const cv::Mat& card_bgr,
                             const std::string& card_type_hint = "");
    cv::Mat preprocessForOcr(const cv::Mat& region);
    std::tuple<std::string, std::string, float> fuzzyMatch(const std::string& ocr_text);
};

#endif
