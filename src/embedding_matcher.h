#ifndef EMBEDDING_MATCHER_H
#define EMBEDDING_MATCHER_H

#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <onnxruntime_cxx_api.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <nlohmann/json.hpp>

class EmbeddingMatcher {
public:
    EmbeddingMatcher()
        : env_(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "vtes-embedding"),
          threshold_(0.35f), embedding_dim_(0), num_cards_(0) {}

    // After load(), call this to enable type-based candidate filtering.
    // Maps each embedding entry (by index) to its vtes.json type(s).
    // `type_by_id` maps card_id → list of type strings (e.g. ["Vampire"])
    void set_card_types(const std::unordered_map<std::string, std::vector<std::string>>& type_by_id) {
        card_types_.clear();
        card_types_.reserve(card_ids_.size());
        for (const auto& cid : card_ids_) {
            auto it = type_by_id.find(cid);
            if (it != type_by_id.end())
                card_types_.push_back(it->second);
            else
                card_types_.push_back({}); // unknown types — never filtered out
        }
    }

    bool load(const std::string& onnx_path,
              const std::string& bin_path,
              const std::string& meta_path)
    {
        // 1. Load ONNX model
        try {
            Ort::SessionOptions opts;
            opts.SetIntraOpNumThreads(1);
            opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
#ifdef _WIN32
            // Convert UTF-8 to wstring for Windows
            int wlen = MultiByteToWideChar(CP_UTF8, 0, onnx_path.c_str(), -1, nullptr, 0);
            std::wstring wpath(wlen, L'\0');
            MultiByteToWideChar(CP_UTF8, 0, onnx_path.c_str(), -1, wpath.data(), wlen);
            session_ = std::make_unique<Ort::Session>(env_, wpath.c_str(), opts);
#else
            session_ = std::make_unique<Ort::Session>(env_, onnx_path.c_str(), opts);
#endif
        } catch (const Ort::Exception& e) {
            fprintf(stderr, "[EmbeddingMatcher] Failed to load ONNX model: %s\n", e.what());
            return false;
        }

        // 2. Load embeddings binary
        std::ifstream bin_file(bin_path, std::ios::binary | std::ios::ate);
        if (!bin_file.is_open()) {
            fprintf(stderr, "[EmbeddingMatcher] Cannot open embeddings file: %s\n", bin_path.c_str());
            return false;
        }
        size_t file_size = (size_t)bin_file.tellg();
        bin_file.seekg(0, std::ios::beg);

        if (file_size < sizeof(float)) {
            fprintf(stderr, "[EmbeddingMatcher] embeddings file too small: %zu bytes\n", file_size);
            return false;
        }

        size_t num_floats = file_size / sizeof(float);
        embeddings_.resize(num_floats);
        bin_file.read(reinterpret_cast<char*>(embeddings_.data()), file_size);

        // 3. Load metadata
        std::ifstream meta_file(meta_path);
        if (!meta_file.is_open()) {
            fprintf(stderr, "[EmbeddingMatcher] Cannot open metadata file: %s\n", meta_path.c_str());
            return false;
        }

        try {
            nlohmann::json j;
            meta_file >> j;

            if (!j.is_array()) {
                fprintf(stderr, "[EmbeddingMatcher] metadata is not a JSON array\n");
                return false;
            }

            card_names_.clear();
            card_ids_.clear();
            for (const auto& entry : j) {
                card_ids_.push_back(entry.value("id", ""));
                card_names_.push_back(entry.value("name", ""));
            }
        } catch (const std::exception& e) {
            fprintf(stderr, "[EmbeddingMatcher] Failed to parse metadata: %s\n", e.what());
            return false;
        }

        num_cards_ = (int)card_names_.size();
        if (num_cards_ == 0) {
            fprintf(stderr, "[EmbeddingMatcher] No cards in index\n");
            return false;
        }

        embedding_dim_ = (int)(num_floats / num_cards_);
        if (embedding_dim_ * num_cards_ != (int)num_floats) {
            fprintf(stderr, "[EmbeddingMatcher] Dimension mismatch: %zu floats, %d cards\n",
                    num_floats, num_cards_);
            return false;
        }

        fprintf(stderr, "[EmbeddingMatcher] Loaded: %d cards, %d dims, %.1f MB\n",
                num_cards_, embedding_dim_,
                embeddings_.size() * sizeof(float) / 1024.0f / 1024.0f);
        return true;
    }

    bool identify(const cv::Mat& card_bgr,
                  std::string& out_name,
                  std::string& out_id,
                  float& out_confidence,
                  const std::string& type_filter = "")
    {
        out_name.clear();
        out_id.clear();
        out_confidence = 0.0f;

        if (!session_ || num_cards_ == 0) return false;

        // 1. Preprocess card image
        std::vector<float> query_emb = compute_embedding(card_bgr);
        if (query_emb.empty()) return false;

        // 2. Match against index via cosine similarity
        float best_sim = -1.0f;
        int best_idx = -1;
        float fallback_sim = -1.0f;
        int fallback_idx = -1;

        bool has_filter = !type_filter.empty();
        const float* index_data = embeddings_.data();

        for (int i = 0; i < num_cards_; i++) {
            // Skip non-matching types when filter is active
            if (has_filter) {
                const auto& types = (i < (int)card_types_.size()) ? card_types_[i] : empty_types_;
                bool type_match = false;
                for (const auto& t : types) {
                    if (t == type_filter) { type_match = true; break; }
                }
                if (!type_match) {
                    // Still track best fallback (non-filtered) for later
                    float dot = 0.0f;
                    for (int j = 0; j < embedding_dim_; j++)
                        dot += query_emb[j] * index_data[i * embedding_dim_ + j];
                    if (dot > fallback_sim) { fallback_sim = dot; fallback_idx = i; }
                    continue;
                }
            }

            float dot = 0.0f;
            for (int j = 0; j < embedding_dim_; j++) {
                dot += query_emb[j] * index_data[i * embedding_dim_ + j];
            }
            if (dot > best_sim) {
                best_sim = dot;
                best_idx = i;
            }
        }

        // If filter is active and we found a good match within the type, use it
        if (has_filter && best_idx >= 0 && best_sim >= threshold_ * 0.5f) {
            out_name = card_names_[best_idx];
            out_id = card_ids_[best_idx];
            out_confidence = best_sim;
            if (best_sim >= threshold_) return true;
            if (best_sim >= threshold_ * 0.5f) return true;
        }

        // Fallback: if no good type-filtered match, or no filter, use the global best
        if (fallback_idx >= 0 && fallback_sim > best_sim) {
            best_sim = fallback_sim;
            best_idx = fallback_idx;
        }

        // If filter was set but no good type match, also try unfiltered
        if (has_filter && (best_idx < 0 || best_sim < threshold_ * 0.5f)) {
            for (int i = 0; i < num_cards_; i++) {
                float dot = 0.0f;
                for (int j = 0; j < embedding_dim_; j++)
                    dot += query_emb[j] * index_data[i * embedding_dim_ + j];
                if (dot > best_sim) {
                    best_sim = dot;
                    best_idx = i;
                }
            }
        }

        if (best_idx >= 0) {
            out_name = card_names_[best_idx];
            out_id = card_ids_[best_idx];
            out_confidence = best_sim;
            if (best_sim >= threshold_) return true;
            if (best_sim >= threshold_ * 0.5f) return true;
        }

        out_confidence = best_sim > 0 ? best_sim : 0.0f;
        return false;
    }

    bool is_loaded() const { return session_ != nullptr; }
    int num_cards() const { return num_cards_; }
    int embedding_dim() const { return embedding_dim_; }
    float threshold() const { return threshold_; }
    void set_threshold(float t) { threshold_ = t; }

private:
    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<float> embeddings_;
    std::vector<std::string> card_names_;
    std::vector<std::string> card_ids_;
    std::vector<std::vector<std::string>> card_types_;  // parallel to card_ids_
    std::vector<std::string> empty_types_;              // const ref for cards w/o type info
    float threshold_;
    int embedding_dim_;
    int num_cards_;

    std::vector<float> compute_embedding(const cv::Mat& bgr)
    {
        if (bgr.empty()) return {};

        // Resize to 224x224
        cv::Mat resized;
        cv::resize(bgr, resized, cv::Size(224, 224), 0, 0, cv::INTER_LINEAR);

        // Convert to float and normalize
        cv::Mat float_img;
        resized.convertTo(float_img, CV_32FC3, 1.0 / 255.0);

        // ImageNet normalization
        static const float mean[] = {0.485f, 0.456f, 0.406f};
        static const float std[] = {0.229f, 0.224f, 0.225f};

        // Prepare input blob: NCHW format
        std::vector<float> input_tensor(1 * 3 * 224 * 224);
        for (int y = 0; y < 224; y++) {
            for (int x = 0; x < 224; x++) {
                cv::Vec3f pixel = float_img.at<cv::Vec3f>(y, x);
                // OpenCV is BGR, model expects RGB
                input_tensor[0 * 224 * 224 + y * 224 + x] = (pixel[2] - mean[0]) / std[0];  // R
                input_tensor[1 * 224 * 224 + y * 224 + x] = (pixel[1] - mean[1]) / std[1];  // G
                input_tensor[2 * 224 * 224 + y * 224 + x] = (pixel[0] - mean[2]) / std[2];  // B
            }
        }

        // Run inference
        try {
            Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
                OrtArenaAllocator, OrtMemTypeDefault);

            std::vector<int64_t> input_shape = {1, 3, 224, 224};
            Ort::Value input_val = Ort::Value::CreateTensor<float>(
                mem_info, input_tensor.data(), input_tensor.size(),
                input_shape.data(), input_shape.size());

            const char* input_name = "input";
            const char* output_name = "embedding";
            Ort::AllocatorWithDefaultOptions allocator;

            // Get actual input/output names from the model
            Ort::AllocatedStringPtr in_name_ptr = session_->GetInputNameAllocated(0, allocator);
            Ort::AllocatedStringPtr out_name_ptr = session_->GetOutputNameAllocated(0, allocator);
            const char* actual_in = in_name_ptr.get();
            const char* actual_out = out_name_ptr.get();

            std::vector<Ort::Value> output = session_->Run(
                Ort::RunOptions{nullptr},
                &actual_in, &input_val, 1,
                &actual_out, 1);

            if (output.empty() || !output[0].IsTensor()) return {};

            float* output_data = output[0].GetTensorMutableData<float>();
            Ort::TensorTypeAndShapeInfo shape_info = output[0].GetTensorTypeAndShapeInfo();
            size_t output_size = shape_info.GetElementCount();

            // L2-normalize the output
            float norm = 0.0f;
            for (size_t i = 0; i < output_size; i++) norm += output_data[i] * output_data[i];
            norm = std::sqrt(norm);
            if (norm < 1e-6f) norm = 1.0f;

            std::vector<float> embedding(output_size);
            for (size_t i = 0; i < output_size; i++) embedding[i] = output_data[i] / norm;

            return embedding;
        } catch (const Ort::Exception& e) {
            fprintf(stderr, "[EmbeddingMatcher] Inference error: %s\n", e.what());
            return {};
        }
    }
};

#endif // EMBEDDING_MATCHER_H
