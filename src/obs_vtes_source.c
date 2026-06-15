// obs_vtes_source.c - Plugin VTES-Card-Scanner OBS
// OBS Studio 32.x API
// Detección de cartas VTES con 14 tipos:
//   Action, Action Modifier, Ally, Combat, Conviction, Equipment,
//   Event, Imbued, Master, Political Action, Power, Reaction, Retainer, Vampire

#include <obs-module.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>
#include <memory>
#include <mutex>

// ============================================================================
// CONFIGURACIÓN
// ==============================================================================

static const char* MODEL_PATH_PROPERTY = "model_path";
static const char* DEFAULT_MODEL_PATH = "/mnt/e/VTES/yolov8m.onnx";

static const char* CONF_PROPERTY = "confidence";
static const float DEFAULT_CONF = 50.0f; // 0-100

static const int WIDTH = 1920; // Full HD
static const int HEIGHT = 1080; // 1080p

static const int FPS = 30;

// ============================================================================
// CATEGORÍAS VTES - Tipos extraídos de vtes.json
// ==============================================================================

// Tipos de carta de VTES (14 tipos únicos)
const std::vector<std::string> VTES_CLASSES = {
    "Action",
    "Action Modifier",
    "Ally",
    "Combat",
    "Conviction",
    "Equipment",
    "Event",
    "Imbued",
    "Master",
    "Political Action",
    "Power",
    "Reaction",
    "Retainer",
    "Vampire"
};

// ============================================================================
// PLUGIN STATE
// ==============================================================================

struct VTESSource {
    cv::dnn::Net net;
    bool is_initialized = false;
    std::string model_path;
    float confidence_threshold;
    int frame_width;
    int frame_height;
    int fps;
    std::vector<std::string> class_names;
    
    std::vector<cv::Rect> last_detections;
    std::shared_ptr<cv::Mat> last_frame;
    int detection_timer = 0;
    
    VTESSource() : net(), confidence_threshold(DEFAULT_CONF),
                   frame_width(WIDTH), frame_height(HEIGHT), fps(FPS),
                   last_detections(), last_frame(), detection_timer() {}
    
    bool init(const std::string& path, float conf) {
        model_path = path;
        confidence_threshold = conf;
        
        try {
            net = cv::dnn::readNetFromONNX(path);
            is_initialized = true;
            obs_log_info("YOLO model initialized: %s", path.c_str());
            obs_log_info("Model outputs: %zu", net.getUnconnectedOutputLayers().size());
            
            // Set class names
            class_names = VTES_CLASSES;
            
            return true;
        } catch (const std::exception& e) {
            obs_log_error("Failed to load YOLO model: %s", e.what());
            return false;
        }
    }
    
    std::vector<Detection> detect(const cv::Mat& frame) {
        if (!is_initialized) {
            return {};
        }
        
        // Preprocess frame to model size
        cv::Mat scaled_frame;
        cv::resize(frame, scaled_frame, cv::Size(frame_width, frame_height), 0, 0, cv::INTER_CUBIC);
        
        // BLOB preparation
        cv::dnn::blobFromImage(scaled_frame, scaled_frame, 1.0/255.0,
                              cv::Size(frame_width, frame_height),
                              cv::Scalar(0, 0, 0), true, false);
        
        // Run inference
        net.setInput(scaled_frame);
        std::vector<cv::Mat> outputs = net.forward();
        
        // Extract detections (simplified NMS)
        std::vector<Detection> detections;
        
        // TODO: Implement NMS y extracción de bounding boxes
        // Esta función necesita implementación completa
        
        return detections;
    }
};

// Global state
static VTESSource g_vtes;
static std::vector<Detection> g_detections;
static std::shared_ptr<cv::Mat> g_frame;
static int g_display_timer = 0;

// ============================================================================
// PLUGIN FUNCTIONS
// ==============================================================================

// Create - Initializar plugin
static bool create(obs_data_t* data, obs_source_t* source) {
    obs_log_info("VTES-Card-Scanner plugin initialized");
    obs_log_info("Types: %zu", VTES_CLASSES.size());
    return true;
}

// Destroy - Limpiar recursos
static void destroy(obs_source_t* source) {
    obs_log_info("VTES-Card-Scanner plugin destroyed");
    g_vtes = VTESSource();
    g_detections.clear();
    g_frame.reset();
}

// Update - Actualizar propiedades
static void update(obs_source_t* source) {
    // Actualizar configuración de propiedades
    const char* model_path = obs_source_get_property_string(source, MODEL_PATH_PROPERTY, DEFAULT_MODEL_PATH);
    if (model_path) {
        g_vtes.init(model_path, DEFAULT_CONF);
    }
}

// Render - Renderizar frame con detección
static void render(obs_source_t* source) {
    // Capturar frame de la escena
    cv::Mat frame;
    obs_source_get_video_frame(source, &frame);
    
    if (frame.empty() || !g_vtes.is_initialized) {
        return;
    }
    
    // Ejecutar detección YOLO
    g_detections = g_vtes.detect(frame);
    
    // Crear overlay
    cv::Mat output = frame.clone();
    cv::Scalar box_color(0, 255, 0); // Verde para cartas individuales
    cv::Scalar box_color_failed(255, 0, 0); // Rojo para fallos
    
    // Renderizar cada carta detectada
    for (const auto& detection : g_detections) {
        // Dibujar bounding box
        cv::rectangle(output, detection.rect, box_color, 2);
        
        // Añadir información
        char label[256];
        snprintf(label, sizeof(label), "%s\n%.1f%%", 
                 detection.name.c_str(),
                 detection.confidence * 100.0f);
        
        // Dibujar texto con nombre y confianza
        cv::putText(output, label, detection.br(), cv::FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1);
    }
    
    // Si no hay detecciones, mostrar placeholder
    if (g_detections.empty()) {
        cv::putText(output, "NO CARD DETECTED", 
                    cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 0.5, 
                    cv::Scalar(200, 200, 200), 1);
    }
    
    // Actualizar source
    obs_source_update_video(source, output);
}

// Activate - Activar fuente
static void activate(obs_source_t* source) {
    obs_log_info("VTES-Card-Scanner activated");
}

// Deactivate - Desactivar fuente
static void deactivate(obs_source_t* source) {
    obs_log_info("VTES-Card-Scanner deactivated");
}

// ============================================================================
// DEFAULTS
// ==============================================================================

static obs_data_t* get_defaults(obs_data_t* data) {
    obs_data_set_string(data, MODEL_PATH_PROPERTY, DEFAULT_MODEL_PATH);
    obs_data_set_float(data, CONF_PROPERTY, DEFAULT_CONF);
    obs_data_set_int(data, "width", WIDTH);
    obs_data_set_int(data, "height", HEIGHT);
    obs_data_set_int(data, "fps", FPS);
    obs_data_set_bool(data, "enable_detection", true);
    obs_data_set_bool(data, "show_information", true);
    obs_data_set_int(data, "display_time_seconds", 10);
    return data;
}

// Get - Obtener propiedades
static bool get(obs_source_t* source, obs_data_t** out) {
    *out = obs_data_create_from_json_file_safe(
        "/home/cifrado/.openclaw/workspace/projects/vtes-obs-detect/obs_vtes_source.json",
        ".json.bak"
    );
    return true;
}

// Set - Establecer propiedades
static void set(obs_source_t* source, obs_data_t* data, size_t index, const char* value) {
    // Handle property changes
}

// ============================================================================
// ==============================================================================

} // extern "C"