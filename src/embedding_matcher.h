#ifndef EMBEDDING_MATCHER_H
#define EMBEDDING_MATCHER_H

#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/dnn.hpp>
#include <nlohmann/json.hpp>

#include "detection/detection_types.hpp"

class EmbeddingMatcher {
public:
    EmbeddingMatcher()
        : threshold_(0.80f), embedding_dim_(0), num_cards_(0),
          inference_device_(InferenceDevice::CPU) {}

    void set_inference_device(InferenceDevice dev) { inference_device_ = dev; }

    void set_card_types(const std::unordered_map<std::string, std::vector<std::string>>& type_by_id) {
        card_types_.clear();
        card_types_.reserve(card_ids_.size());
        for (const auto& cid : card_ids_) {
            auto it = type_by_id.find(cid);
            if (it != type_by_id.end())
                card_types_.push_back(it->second);
            else
                card_types_.push_back({});
        }
    }

    bool load(const std::string& onnx_path,
              const std::string& bin_path,
              const std::string& meta_path)
    {
        // 1. Load ONNX model
        try {
            net_ = cv::dnn::readNetFromONNX(onnx_path);
            if (net_.empty()) {
                fprintf(stderr, "[EmbeddingMatcher] Failed to load ONNX model: %s\n", onnx_path.c_str());
                return false;
            }

            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        } catch (const std::exception& e) {
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

        if (net_.empty() || num_cards_ == 0) return false;

        // 1. Preprocess card image
        std::vector<float> query_emb = compute_embedding(card_bgr);
        if (query_emb.empty()) return false;
        // Seguridad: si el indice cargado no coincide con el formato de consulta,
        // abortar en vez de leer fuera de rango.
        if ((int)query_emb.size() < embedding_dim_) return false;

        // 2. Match against index via cosine similarity
        float best_sim = -1.0f;
        int best_idx = -1;
        float fallback_sim = -1.0f;
        int fallback_idx = -1;

        bool has_filter = !type_filter.empty();
        const float* index_data = embeddings_.data();

        for (int i = 0; i < num_cards_; i++) {
            if (has_filter) {
                const auto& types = (i < (int)card_types_.size()) ? card_types_[i] : empty_types_;
                bool type_match = false;
                for (const auto& t : types) {
                    if (t == type_filter) { type_match = true; break; }
                }
                if (!type_match) {
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

        if (has_filter && best_idx >= 0 && best_sim >= threshold_ * 0.5f) {
            out_name = card_names_[best_idx];
            out_id = card_ids_[best_idx];
            out_confidence = best_sim;
            if (best_sim >= threshold_) return true;
            if (best_sim >= threshold_ * 0.5f) return true;
        }

        if (fallback_idx >= 0 && fallback_sim > best_sim) {
            best_sim = fallback_sim;
            best_idx = fallback_idx;
        }

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

    bool is_loaded() const { return !net_.empty(); }
    int num_cards() const { return num_cards_; }
    int embedding_dim() const { return embedding_dim_; }
    float threshold() const { return threshold_; }
    void set_threshold(float t) { threshold_ = t; }

private:
    cv::dnn::Net net_;
    std::vector<float> embeddings_;
    std::vector<std::string> card_names_;
    std::vector<std::string> card_ids_;
    std::vector<std::vector<std::string>> card_types_;
    std::vector<std::string> empty_types_;
    float threshold_;
    int embedding_dim_;
    InferenceDevice inference_device_;
    int num_cards_;

    std::vector<float> compute_embedding(const cv::Mat& bgr)
    {
        if (bgr.empty()) return {};

        cv::Mat resized;
        cv::resize(bgr, resized, cv::Size(224, 224), 0, 0, cv::INTER_LINEAR);

        // NCHW blob, RGB channel order, pixel values in [0,1]
        cv::Mat blob = cv::dnn::blobFromImage(resized, 1.0 / 255.0, cv::Size(224, 224),
                                               cv::Scalar(), true, false);

        // Apply ImageNet normalization per channel: (x - mean) / std
        static const float mean[] = {0.485f, 0.456f, 0.406f};
        static const float stdv[] = {0.229f, 0.224f, 0.225f};
        float* data = blob.ptr<float>();
        size_t ch_size = 224 * 224;
        for (int c = 0; c < 3; c++) {
            for (size_t i = 0; i < ch_size; i++) {
                data[c * ch_size + i] = (data[c * ch_size + i] - mean[c]) / stdv[c];
            }
        }

        try {
            net_.setInput(blob);
            cv::Mat output = net_.forward();

            if (output.empty()) return {};

            float* output_data = output.ptr<float>();
            size_t output_size = output.total();

            float norm = 0.0f;
            for (size_t i = 0; i < output_size; i++) norm += output_data[i] * output_data[i];
            norm = std::sqrt(norm);
            if (norm < 1e-6f) norm = 1.0f;

            std::vector<float> embedding(output_size);
            for (size_t i = 0; i < output_size; i++) embedding[i] = output_data[i] / norm;

            // Indice fusionado (embedding + NCC): la consulta repite el embedding
            // 4 veces y anade el bloque NCC del recorte de arte. El producto punto
            // con cada fila del indice = media de las 4 similitudes de embedding
            // + similitud NCC. Ver scripts/build_embedding_index.py (--fused).
            if (embedding_dim_ == 4 * (int)output_size + 3 * 32 * 32) {
                std::vector<float> ncc = compute_ncc_block(bgr);
                if (!ncc.empty()) {
                    std::vector<float> fused;
                    fused.reserve(4 * output_size + ncc.size());
                    for (int r = 0; r < 4; r++)
                        fused.insert(fused.end(), embedding.begin(), embedding.end());
                    fused.insert(fused.end(), ncc.begin(), ncc.end());
                    return fused;
                }
            }

            return embedding;
        } catch (const std::exception& e) {
            fprintf(stderr, "[EmbeddingMatcher] Inference error: %s\n", e.what());
            return {};
        }
    }

    // Bloque NCC (32x32x3, BGR intercalado, media 0, norma 1) del recorte de arte.
    // Debe coincidir EXACTAMENTE con scripts/build_embedding_index.py.
    std::vector<float> compute_ncc_block(const cv::Mat& bgr)
    {
        if (bgr.empty() || bgr.cols < 8 || bgr.rows < 8) return {};

        int y0 = (int)(bgr.rows * 0.08);
        int y1 = (int)(bgr.rows * 0.64);
        int x0 = (int)(bgr.cols * 0.03);
        int x1 = (int)(bgr.cols * 0.97);
        if (y0 < 0) y0 = 0;
        if (x0 < 0) x0 = 0;
        if (y1 > bgr.rows) y1 = bgr.rows;
        if (x1 > bgr.cols) x1 = bgr.cols;
        if (y1 <= y0 || x1 <= x0) return {};

        cv::Mat art = bgr(cv::Rect(x0, y0, x1 - x0, y1 - y0));
        cv::Mat art32;
        cv::resize(art, art32, cv::Size(32, 32), 0, 0, cv::INTER_AREA);
        if (!art32.isContinuous()) art32 = art32.clone();

        std::vector<float> v(32 * 32 * 3);
        const unsigned char* p = art32.ptr<unsigned char>();
        float mean = 0.0f;
        for (size_t i = 0; i < v.size(); i++) { v[i] = (float)p[i]; mean += v[i]; }
        mean /= (float)v.size();

        float norm = 0.0f;
        for (auto& x : v) { x -= mean; norm += x * x; }
        norm = std::sqrt(norm);
        if (norm < 1e-6f) norm = 1.0f;
        for (auto& x : v) x /= norm;
        return v;
    }
};

#endif // EMBEDDING_MATCHER_H
