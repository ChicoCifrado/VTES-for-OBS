#ifndef CONTOUR_DETECTOR_HPP
#define CONTOUR_DETECTOR_HPP

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

#include "yolo_detector.hpp"

namespace vtes_detection {

struct ContourParams {
    // Adaptive threshold (mtg_card_detector style, replaces Canny)
    // 0 = automatic (odd block ~3-5% of the shorter frame side). A fixed 5
    // produced fragmented contours on 720p/1080p frames and the card was not
    // an outermost contour, so the detector returned no objects.
    int adaptive_block_size = 0;    // 0 = auto; set an odd value to force it
    double adaptive_c = 5;          // constant subtracted from mean

    // Alternative: Canny fallback if adaptive fails on simple backgrounds
    int edge_low = 50;
    int edge_high = 150;

    int min_area_pixels = 4000;     // absolute min area in pixels
    double min_area_fraction = 0.02;
    double max_area_fraction = 0.30;
    double target_aspect = 63.0 / 88.0;
    double aspect_tolerance = 0.15;
    double min_solidity = 0.60;
};

struct ContourCandidate {
    std::vector<cv::Point> contour;
    int hierarchy_level;  // 0 = outermost, >0 = nested
    double area;
    double solidity;
    double aspect_ratio;
};

std::vector<OBBObject> detect_by_contour(
    const cv::Mat& bgr_frame, const ContourParams& params);

} // namespace vtes_detection

#endif
