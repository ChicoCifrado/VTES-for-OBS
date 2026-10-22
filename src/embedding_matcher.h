#ifndef EMBEDDING_MATCHER_H
#define EMBEDDING_MATCHER_H

#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <thread>
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
                  const std::string& type_filter = "",
                  std::string* out_second_name = nullptr,
                  std::string* out_second_id = nullptr,
                  float* out_second_confidence = nullptr)
    {
        out_name.clear();
        out_id.clear();
        out_confidence = 0.0f;
        if (out_second_name) out_second_name->clear();
        if (out_second_id) out_second_id->clear();
        if (out_second_confidence) *out_second_confidence = -1.0f;

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
        const int dim = embedding_dim_;
        const float* query = query_emb.data();

        // Producto escalar con cuatro acumuladores independientes. El
        // producto escalar es la parte que domina el coste (4.446 x 7.168
        // floats): con un unico acumulador el compilador no puede vectorizar
        // bien la reduccion y el hilo de video se queda esperando.
        auto dot_row = [&](int idx) -> float {
            const float* row = index_data + (size_t)idx * dim;
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            int j = 0;
            for (; j + 3 < dim; j += 4) {
                s0 += row[j] * query[j];
                s1 += row[j + 1] * query[j + 1];
                s2 += row[j + 2] * query[j + 2];
                s3 += row[j + 3] * query[j + 3];
            }
            float s = (s0 + s1) + (s2 + s3);
            for (; j < dim; ++j) s += row[j] * query[j];
            return s;
        };

        // Top-K de filas para calcular el margen top-1/top-2. El indice
        // contiene variantes duplicadas de la misma carta, asi que despues
        // se busca la mejor fila de una carta DISTINTA.
        constexpr int kTopK = 12;
        float top_sim[kTopK];
        int top_idx[kTopK];
        for (int k = 0; k < kTopK; k++) {
            top_sim[k] = -1.0e30f;
            top_idx[k] = -1;
        }
        auto consider_candidate = [&](float dot, int idx) {
            if (has_filter) return;
            if (dot <= top_sim[kTopK - 1]) return;
            int pos = kTopK - 1;
            while (pos > 0 && dot > top_sim[pos - 1]) {
                top_sim[pos] = top_sim[pos - 1];
                top_idx[pos] = top_idx[pos - 1];
                --pos;
            }
            top_sim[pos] = dot;
            top_idx[pos] = idx;
        };

        if (!has_filter) {
            // El indice es de solo lectura. Repartimos las filas entre unos
            // pocos hilos para que la lectura de los ~127 MB del indice no
            // recaiga exclusivamente en el hilo de video de OBS. Se limita a
            // 8 hilos: mas no ayuda porque el cuello es la memoria y ademas
            // dejamos CPU para OBS, OCR y el resto del filtro.
            struct LocalTop {
                float best_sim = -1.0f;
                int best_idx = -1;
                float top_sim[kTopK];
                int top_idx[kTopK];
            };
            const unsigned hc = std::thread::hardware_concurrency();
            const int nthreads = (num_cards_ < 512)
                ? 1
                : (int)std::min<unsigned>(8u, std::max(1u, hc));
            std::vector<LocalTop> local(nthreads);
            auto init_local = [&](LocalTop &L) {
                L.best_sim = -1.0f;
                L.best_idx = -1;
                for (int k = 0; k < kTopK; k++) {
                    L.top_sim[k] = -1.0e30f;
                    L.top_idx[k] = -1;
                }
            };
            for (auto &L : local) init_local(L);

            auto work = [&](int t, int begin, int end) {
                LocalTop &L = local[t];
                for (int i = begin; i < end; i++) {
                    const float dot = dot_row(i);
                    if (dot > L.top_sim[kTopK - 1]) {
                        int pos = kTopK - 1;
                        while (pos > 0 && dot > L.top_sim[pos - 1]) {
                            L.top_sim[pos] = L.top_sim[pos - 1];
                            L.top_idx[pos] = L.top_idx[pos - 1];
                            --pos;
                        }
                        L.top_sim[pos] = dot;
                        L.top_idx[pos] = i;
                    }
                    if (dot > L.best_sim) {
                        L.best_sim = dot;
                        L.best_idx = i;
                    }
                }
            };

            if (nthreads == 1) {
                work(0, 0, num_cards_);
            } else {
                std::vector<std::thread> workers;
                workers.reserve((size_t)nthreads);
                const int chunk = (num_cards_ + nthreads - 1) / nthreads;
                for (int t = 0; t < nthreads; t++) {
                    const int begin = t * chunk;
                    const int end = std::min(num_cards_, begin + chunk);
                    if (begin >= end) break;
                    workers.emplace_back(work, t, begin, end);
                }
                for (auto &w : workers) w.join();
            }

            // Merge de los resultados parciales. Los 12 mejores de cada hilo
            // bastan para reconstruir el top-12 global.
            for (const auto &L : local) {
                if (L.best_sim > best_sim) {
                    best_sim = L.best_sim;
                    best_idx = L.best_idx;
                }
                for (int k = 0; k < kTopK; k++) {
                    if (L.top_idx[k] >= 0)
                        consider_candidate(L.top_sim[k], L.top_idx[k]);
                }
            }
        } else {
            for (int i = 0; i < num_cards_; i++) {
                const auto& types = (i < (int)card_types_.size()) ? card_types_[i] : empty_types_;
                bool type_match = false;
                for (const auto& t : types) {
                    if (t == type_filter) { type_match = true; break; }
                }
                if (!type_match) {
                    const float dot = dot_row(i);
                    if (dot > fallback_sim) { fallback_sim = dot; fallback_idx = i; }
                    continue;
                }

                const float dot = dot_row(i);
                consider_candidate(dot, i);
                if (dot > best_sim) {
                    best_sim = dot;
                    best_idx = i;
                }
            }
        }

        // Mejor carta distinta de la ganadora (ignorando filas duplicadas de
        // la misma carta). Solo se calcula en el matching global, que es el
        // que usa el filtro fusionado embedding+NCC.
        if (!has_filter && best_idx >= 0) {
            const std::string& best_id = card_ids_[best_idx];
            for (int k = 0; k < kTopK; k++) {
                if (top_idx[k] < 0 || top_idx[k] == best_idx) continue;
                if (card_ids_[top_idx[k]] == best_id) continue;
                if (out_second_name) *out_second_name = card_names_[top_idx[k]];
                if (out_second_id) *out_second_id = card_ids_[top_idx[k]];
                if (out_second_confidence) *out_second_confidence = top_sim[k];
                break;
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
                const float dot = dot_row(i);
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

    // El detector OBB deja unos pixeles de fieltro en los bordes, sobre todo
    // en cartas pequenas. Ese margen se amplifica al normalizar a 224x224 y
    // desplaza el embedding hacia otra carta. Recortamos un 5% por lado solo
    // para la consulta; el recorte completo sigue usandose para OCR, overlay y
    // la cola del realizador.
    cv::Mat trim_query_border(const cv::Mat& bgr) const
    {
        if (bgr.empty() || bgr.cols < 16 || bgr.rows < 16) return bgr;
        const int dx = std::max(1, (int)std::lround(bgr.cols * 0.05));
        const int dy = std::max(1, (int)std::lround(bgr.rows * 0.05));
        if (bgr.cols - 2 * dx < 8 || bgr.rows - 2 * dy < 8) return bgr;
        return bgr(cv::Rect(dx, dy, bgr.cols - 2 * dx, bgr.rows - 2 * dy)).clone();
    }

    std::vector<float> compute_embedding(const cv::Mat& bgr)
    {
        if (bgr.empty()) return {};

        const cv::Mat query = trim_query_border(bgr);
        cv::Mat resized;
        cv::resize(query, resized, cv::Size(224, 224), 0, 0, cv::INTER_LINEAR);

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
                std::vector<float> ncc = compute_ncc_block(query);
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
