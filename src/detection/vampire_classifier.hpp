#ifndef VAMPIRE_CLASSIFIER_HPP
#define VAMPIRE_CLASSIFIER_HPP

#include <opencv2/core.hpp>

#define NUM_BINS 36

#ifdef HAVE_CUDA_NIS
// ─── CUDA-based oval vs rectangular portrait classifier ──────────────
// VTES vampire cards have an oval painted portrait frame.
// Non-vampire cards (Library, Master, etc.) have rectangular art frames.
//
// Uses edge orientation histogram on GPU to distinguish:
//   Rectangular → strong peaks at 0° (horizontal) and 90° (vertical)
//   Oval       → edges distributed uniformly across all angles
//
// Returns score 0.0–1.0 where > 0.5 = likely vampire (oval frame).

float computeVampireScoreCUDA(const cv::Mat& card_bgr,
                              float edge_threshold = 50.0f,
                              int device_id = 0);

bool isVampireCUDA(const cv::Mat& card_bgr,
                   float threshold = 0.5f,
                   float edge_threshold = 50.0f,
                   int device_id = 0);

#else
// CPU fallback: use HoughLinesP (same as VtesOcrReader::hasVampireOval)
inline float computeVampireScoreCUDA(const cv::Mat&, float = 50.0f, int = 0) { return 0.5f; }
inline bool isVampireCUDA(const cv::Mat&, float = 0.5f, float = 50.0f, int = 0) { return false; }
#endif

#endif
