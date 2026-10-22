#ifndef DETECTION_TYPES_HPP
#define DETECTION_TYPES_HPP

#include <string>
#include <cstdint>
#include <opencv2/core/types.hpp>
#include <opencv2/video/tracking.hpp>

enum class InferenceDevice { CPU, CUDA, DirectML };

inline const char* inferenceDeviceToString(InferenceDevice dev) {
    switch (dev) {
        case InferenceDevice::CUDA: return "cuda";
        case InferenceDevice::DirectML: return "dml";
        default: return "cpu";
    }
}

namespace vtes_detection {

struct OBBObject
{
    // Centro y tamano de la carta. width/height ya son el tamano real de la
    // carta girada; x/y pueden quedar recortados si la carta toca un borde.
    cv::Rect2f rect;
    float angle;
    // Tamano real sin el recorte de bordes de rect. Se usa para extraer el
    // recorte de cartas que estan parcialmente fuera del frame.
    float obb_w = 0.0f;
    float obb_h = 0.0f;
    int label;
    float prob;
    int id;
    int unseenFrames;
    std::string card_id;
    std::string card_name;
    // Segunda mejor carta del embedding. El OCR puede corregir hacia ella
    // cuando la primera es un falso positivo, pero nunca hacia una carta
    // ajena al top-2.
    std::string alt_card_id;
    std::string alt_card_name;
    // dHash del recorte de esta carta. El OCR asincrono viaja con este hash
    // para no aplicar un resultado de una carta anterior al mismo track.
    uint64_t crop_hash = 0;
};

} // namespace vtes_detection

struct Object {
    cv::Rect_<float> rect;
    int label;
    float prob;
    uint64_t id;
    uint64_t unseenFrames;
    cv::KalmanFilter kf;
};

#endif
