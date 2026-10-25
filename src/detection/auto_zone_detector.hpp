#ifndef AUTO_ZONE_DETECTOR_HPP
#define AUTO_ZONE_DETECTOR_HPP

#include <opencv2/core.hpp>
#include <cstdint>
#include <vector>

namespace vtes_detection {

// Una carta/funda candidata detectada geométricamente. Las coordenadas están
// en píxeles del frame original (no del frame reducido que usa el detector).
struct AutoZoneCard {
	cv::RotatedRect rect;
	float score = 0.0f;
	cv::Mat hist; // histograma de color normalizado (para validar el grupo)
};

// Grupo de cuatro cartas/fundas que representa la cripta boca abajo de un
// jugador. `stable` sólo se activa tras ver el mismo grupo varias veces.
struct AutoZoneCluster {
	int id = -1;
	std::vector<AutoZoneCard> cards;
	cv::Rect2f bbox; // AABB en píxeles del frame original
	cv::Point2f center;
	float card_short = 0.0f;
	float card_long = 0.0f;
	float median_long_angle_deg = 0.0f;
	float score = 0.0f;
	int hits = 0;
	int misses = 0;
	bool stable = false;
	int64_t first_seen_ns = 0;
	int64_t last_seen_ns = 0;
	int player_idx = -1; // lo rellena la integración, no el detector
};

// Zonas derivadas de un grupo. Las coordenadas están en píxeles del frame
// original. `angle` se guarda en grados, positivo = horario en pantalla.
struct AutoZoneGeometry {
	bool valid = false;
	cv::RotatedRect crypt;
	cv::RotatedRect minion;
	float zone_angle_deg = 0.0f;
	float lock_angle_deg = 0.0f;
};

// Detector de zonas a partir de cuatro cartas boca abajo. No usa identidad
// VTES: busca rectángulos parecidos, próximos y alineados. Está pensado para
// fundas con dorso uniforme, donde el detector de cartas normal falla.
class AutoZoneDetector {
public:
	AutoZoneDetector();

	void reset();

	// Devuelve los grupos seguidos en este frame. `exclude_rects` son cajas
	// de cartas ya identificadas (boca arriba) que no deben contar como
	// candidatas. El detector reduce internamente el frame y devuelve todas
	// las coordenadas en píxeles del frame original.
	std::vector<AutoZoneCluster> detect(
		const cv::Mat &bgr, int64_t now_ns,
		const std::vector<cv::Rect2f> &exclude_rects = {});

	const std::vector<AutoZoneCluster> &clusters() const { return tracks_; }

	// Calcula la zona de cripta y la zona de minions a partir del grupo.
	// `line_layout` se usa cuando los grupos están en fila (no alrededor de
	// una mesa): en ese caso la zona de minions va hacia el centro de la
	// imagen, no hacia la media de los grupos.
	static AutoZoneGeometry deriveGeometry(
		const AutoZoneCluster &cluster, const cv::Point2f &table_center,
		const cv::Size &frame_size, bool line_layout);

private:
	struct Candidate {
		cv::RotatedRect small_rect; // coordenadas del frame reducido
		cv::RotatedRect src_rect;   // coordenadas del frame original
		cv::Rect2f small_aabb;
		cv::Point2f center;
		float area = 0.0f;
		float short_side = 0.0f;
		float long_side = 0.0f;
		float long_angle = 0.0f;
		float score = 0.0f;
		cv::Mat hist;
	};

	struct Group {
		std::vector<int> indices;
		cv::Rect2f small_bbox;
		cv::Point2f small_center;
		float score = 0.0f;
	};

	std::vector<Candidate> extractCandidates(
		const cv::Mat &bgr, const std::vector<cv::Rect2f> &exclude_rects);
	// Desdobla un contorno que engloba un grupo 2x2 de cartas. Se usa cuando
	// las cartas se tocan y Canny devuelve un único rectángulo del tamaño del
	// grupo; sin esto ese rectángulo se confundiría con una carta grande.
	std::vector<Candidate> splitMergedGrids(
		const std::vector<Candidate> &candidates, const cv::Mat &small) const;
	std::vector<Group> groupCandidates(const std::vector<Candidate> &candidates) const;
	bool evaluateGroup(const std::vector<Candidate> &candidates,
			   const std::vector<int> &indices, Group &out) const;

	std::vector<AutoZoneCluster> tracks_;
	int next_id_ = 1;
	float scale_ = 1.0f;
	cv::Size source_size_;
};

} // namespace vtes_detection

#endif // AUTO_ZONE_DETECTOR_HPP
