#include "contour_detector.hpp"
#include "plugin-support.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/geometry.hpp>
#include <cmath>
#include <algorithm>

static constexpr float kPI = 3.14159265358979323846f;

namespace vtes_detection {

static bool is_outermost_contour(int idx, const std::vector<cv::Vec4i>& hierarchy)
{
    if (idx < 0 || idx >= (int)hierarchy.size()) return false;
    return hierarchy[idx][3] == -1;
}

// Compute a reasonable adaptive block size from image dimensions.
// Block size must be odd; target ~3-5% of the shorter dimension.
static int compute_block_size(int img_w, int img_h)
{
    int size = std::max(5, std::min(img_w, img_h) / 20);
    if (size % 2 == 0) size++;
    if (size > 99) size = 99;
    return size;
}

std::vector<OBBObject> detect_by_contour(
    const cv::Mat& bgr_frame, const ContourParams& params)
{
    std::vector<OBBObject> objects;
    int w = bgr_frame.cols;
    int h = bgr_frame.rows;
    if (w <= 0 || h <= 0) return objects;

    double frame_area = (double)w * (double)h;
    double min_area = std::max((double)params.min_area_pixels,
                                frame_area * params.min_area_fraction);
    double max_area = frame_area * params.max_area_fraction;

    // ── Preprocessing ──────────────────────────────────────────────────
    cv::Mat gray;
    cv::cvtColor(bgr_frame, gray, cv::COLOR_BGR2GRAY);

    // CLAHE for uniform lighting — clip limit 2.0, tile 8x8
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    cv::Mat enhanced;
    clahe->apply(gray, enhanced);

    // Bilateral filter preserves edges while smoothing noise
    cv::Mat smoothed;
    cv::bilateralFilter(enhanced, smoothed, 9, 50, 50);

    // ── Edge detection: combine adaptive threshold + Canny ──────────────
    int block_size = params.adaptive_block_size;
    if (block_size <= 0 || block_size % 2 == 0)
        block_size = compute_block_size(w, h);

    cv::Mat thresh;
    cv::adaptiveThreshold(smoothed, thresh, 255,
                           cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                           cv::THRESH_BINARY_INV, block_size, params.adaptive_c);

    // Canny edges, combined with adaptive threshold via OR
    cv::Mat edges;
    cv::Canny(smoothed, edges, params.edge_low, params.edge_high, 3, true);
    cv::bitwise_or(thresh, edges, thresh);

    // ── Morphology ──────────────────────────────────────────────────────
    // OPEN (erode→dilate) removes noise; CLOSE (dilate→erode) fills gaps
    cv::Mat morph;
    cv::Mat kernel3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::Mat kernel5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(thresh, morph, cv::MORPH_OPEN, kernel3);
    cv::morphologyEx(morph, morph, cv::MORPH_CLOSE, kernel5);

    // ── Find contours ───────────────────────────────────────────────────
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(morph, contours, hierarchy,
                      cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) return objects;

    // ── Filter outermost contours by area ───────────────────────────────
    std::vector<ContourCandidate> candidates;
    for (int i = 0; i < (int)contours.size(); i++) {
        if (!is_outermost_contour(i, hierarchy)) continue;
        double area = cv::contourArea(contours[i]);
        if (area < min_area || area > max_area) continue;

        ContourCandidate cand;
        cand.contour = contours[i];
        cand.hierarchy_level = 0;
        cand.area = area;
        candidates.push_back(cand);
    }

    // Sort by area descending
    std::sort(candidates.begin(), candidates.end(),
              [](const ContourCandidate& a, const ContourCandidate& b) {
                  return a.area > b.area;
              });

    // ── Convert valid candidates to OBBObjects ──────────────────────────
    int id_counter = 0;
    for (const auto& cand : candidates) {
        // Approximate polygon — look for 4-vertex shapes
        double peri = cv::arcLength(cand.contour, true);
        std::vector<cv::Point> approx;
        cv::approxPolyDP(cand.contour, approx, 0.04 * peri, true);
        if (approx.size() != 4) continue;

        cv::RotatedRect rect = cv::minAreaRect(cand.contour);
        float rw = rect.size.width;
        float rh = rect.size.height;
        if (rw < 1.0f || rh < 1.0f) continue;

        // Aspect ratio: normalize ≥1.0, compare to target
        float aspect = (rw > rh) ? (rw / rh) : (rh / rw);
        double target = (params.target_aspect > 1.0)
                            ? params.target_aspect
                            : (1.0 / params.target_aspect);
        if (std::abs(aspect - target) > params.aspect_tolerance * target) continue;

        // Solidity: area vs convex hull area
        std::vector<cv::Point> hull;
        cv::convexHull(cand.contour, hull);
        double hull_area = cv::contourArea(hull);
        if (hull_area < 1.0) continue;
        double solidity = cand.area / hull_area;
        if (solidity < params.min_solidity) continue;

        // Angle sanity: cards on table usually ±60° from horizontal
        float angle_deg = rect.angle;
        // OpenCV: angle from horizontal axis, range [-90, 0)
        // Normalize to [0, 180)
        if (angle_deg < 0) angle_deg += 180.0f;
        float angle_from_horiz = std::fmod(angle_deg, 90.0f);
        if (angle_from_horiz > 60.0f && angle_from_horiz < 120.0f)
            continue; // too tilted

        cv::RotatedRect rr(rect.center, rect.size, rect.angle);
        cv::Point2f rcorners[4];
        rr.points(rcorners);

        // Ensure corners are within frame bounds
        bool in_bounds = true;
        for (auto& p : rcorners) {
            if (p.x < -5 || p.x > w + 5 || p.y < -5 || p.y > h + 5) {
                in_bounds = false;
                break;
            }
        }
        if (!in_bounds) continue;

        // Find bounding box from rotated rect corners
        float x0 = std::min({rcorners[0].x, rcorners[1].x, rcorners[2].x, rcorners[3].x});
        float y0 = std::min({rcorners[0].y, rcorners[1].y, rcorners[2].y, rcorners[3].y});
        float x1 = std::max({rcorners[0].x, rcorners[1].x, rcorners[2].x, rcorners[3].x});
        float y1 = std::max({rcorners[0].y, rcorners[1].y, rcorners[2].y, rcorners[3].y});
        x0 = std::max(0.0f, x0);
        y0 = std::max(0.0f, y0);
        x1 = std::min((float)(w - 1), x1);
        y1 = std::min((float)(h - 1), y1);
        if (x1 <= x0 || y1 <= y0) continue;

        OBBObject obj;
        obj.rect.x = x0;
        obj.rect.y = y0;
        obj.rect.width = x1 - x0;
        obj.rect.height = y1 - y0;
        obj.angle = angle_deg * kPI / 180.0f;
        obj.label = 0;
        obj.prob = 1.0f;
        obj.id = ++id_counter;
        obj.unseenFrames = 0;
        objects.push_back(obj);
    }

    return objects;
}

} // namespace vtes_detection
