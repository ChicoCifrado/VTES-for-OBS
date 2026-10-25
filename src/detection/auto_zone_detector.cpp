#include "auto_zone_detector.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/geometry.hpp>
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <functional>
#include <numeric>
#include <unordered_map>

namespace vtes_detection {

namespace {

constexpr float kPI = 3.14159265358979323846f;

static float clampf(float v, float lo, float hi)
{
	return std::max(lo, std::min(hi, v));
}

static float normalizeAngle180(float a)
{
	while (a >= 180.0f)
		a -= 360.0f;
	while (a < -180.0f)
		a += 360.0f;
	return a;
}

static float angleDiff180(float a, float b)
{
	const float d = std::fabs(normalizeAngle180(a - b));
	return std::min(d, 180.0f - d);
}

static float medianOf(std::vector<float> values)
{
	if (values.empty())
		return 0.0f;
	std::sort(values.begin(), values.end());
	const size_t n = values.size();
	if (n & 1)
		return values[n / 2];
	return 0.5f * (values[n / 2 - 1] + values[n / 2]);
}

static cv::Rect2f aabbOf(const cv::RotatedRect &rr)
{
	cv::Point2f pts[4];
	rr.points(pts);
	float x0 = pts[0].x, x1 = pts[0].x;
	float y0 = pts[0].y, y1 = pts[0].y;
	for (int i = 1; i < 4; ++i) {
		x0 = std::min(x0, pts[i].x);
		x1 = std::max(x1, pts[i].x);
		y0 = std::min(y0, pts[i].y);
		y1 = std::max(y1, pts[i].y);
	}
	return cv::Rect2f(x0, y0, x1 - x0, y1 - y0);
}

static float aabbIou(const cv::Rect2f &a, const cv::Rect2f &b)
{
	const float x0 = std::max(a.x, b.x);
	const float y0 = std::max(a.y, b.y);
	const float x1 = std::min(a.x + a.width, b.x + b.width);
	const float y1 = std::min(a.y + a.height, b.y + b.height);
	const float iw = std::max(0.0f, x1 - x0);
	const float ih = std::max(0.0f, y1 - y0);
	const float inter = iw * ih;
	const float uni = a.width * a.height + b.width * b.height - inter;
	return uni > 1.0f ? inter / uni : 0.0f;
}

// Canny con umbrales derivados de la mediana del canal. Evita depender de
// los umbrales fijos del modo contorno y funciona con cambios de luz.
static void autoCanny(const cv::Mat &gray, cv::Mat &edges)
{
	if (gray.empty()) {
		edges.release();
		return;
	}

	cv::Mat blurred;
	cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0.0);

	int hist[256] = {0};
	for (int y = 0; y < blurred.rows; ++y) {
		const uchar *row = blurred.ptr<uchar>(y);
		for (int x = 0; x < blurred.cols; ++x)
			hist[row[x]]++;
	}
	const int total = blurred.rows * blurred.cols;
	int acc = 0;
	int median = 128;
	for (int i = 0; i < 256; ++i) {
		acc += hist[i];
		if (acc * 2 >= total) {
			median = i;
			break;
		}
	}

	double low = std::max(10.0, (double)median * 0.55);
	double high = std::max(low + 20.0, (double)median * 1.35);
	high = std::min(255.0, high);
	cv::Canny(blurred, edges, low, high, 3, true);
}

static cv::Mat colorHistogram(const cv::Mat &bgr, const cv::RotatedRect &rr)
{
	cv::Point2f pts[4];
	rr.points(pts);

	// Ordenar las esquinas de forma estable para que el warp no dependa del
	// orden que devuelve OpenCV. El histograma es invariante a la rotación,
	// pero un warp coherente mejora la comparación.
	cv::Point2f c = (pts[0] + pts[1] + pts[2] + pts[3]) * 0.25f;
	std::sort(pts, pts + 4, [c](const cv::Point2f &a, const cv::Point2f &b) {
		return std::atan2(a.y - c.y, a.x - c.x) <
		       std::atan2(b.y - c.y, b.x - c.x);
	});

	const cv::Point2f dst[4] = {
		{0.0f, 0.0f}, {31.0f, 0.0f}, {31.0f, 43.0f}, {0.0f, 43.0f}};
	cv::Mat m = cv::getPerspectiveTransform(pts, dst);
	cv::Mat patch;
	cv::warpPerspective(bgr, patch, m, cv::Size(32, 44), cv::INTER_AREA,
			    cv::BORDER_REPLICATE);
	if (patch.empty())
		return {};

	cv::Mat hsv;
	cv::cvtColor(patch, hsv, cv::COLOR_BGR2HSV);
	int hbins = 8, sbins = 4, vbins = 4;
	int histSize[] = {hbins, sbins, vbins};
	float hranges[] = {0.0f, 180.0f};
	float sranges[] = {0.0f, 256.0f};
	float vranges[] = {0.0f, 256.0f};
	const float *ranges[] = {hranges, sranges, vranges};
	int channels[] = {0, 1, 2};
	cv::Mat hist;
	cv::calcHist(&hsv, 1, channels, cv::Mat(), hist, 3, histSize, ranges,
		     true, false);
	cv::normalize(hist, hist, 1.0, 0.0, cv::NORM_L1);
	return hist.reshape(1, 1);
}

static float histSimilarity(const cv::Mat &a, const cv::Mat &b)
{
	if (a.empty() || b.empty())
		return 0.5f;
	const double d = cv::compareHist(a, b, cv::HISTCMP_BHATTACHARYYA);
	return clampf(1.0f - (float)d, 0.0f, 1.0f);
}

static bool insideExpanded(const cv::Point2f &p, const cv::Rect2f &r,
			   float margin)
{
	return p.x >= r.x - margin && p.x <= r.x + r.width + margin &&
	       p.y >= r.y - margin && p.y <= r.y + r.height + margin;
}

} // namespace

AutoZoneDetector::AutoZoneDetector() = default;

void AutoZoneDetector::reset()
{
	tracks_.clear();
	next_id_ = 1;
}

std::vector<AutoZoneDetector::Candidate> AutoZoneDetector::extractCandidates(
	const cv::Mat &bgr, const std::vector<cv::Rect2f> &exclude_rects)
{
	std::vector<Candidate> candidates;
	if (bgr.empty())
		return candidates;

	source_size_ = bgr.size();
	const int max_side = 960;
	scale_ = std::min(1.0f, (float)max_side /
				       (float)std::max(bgr.cols, bgr.rows));
	cv::Mat small;
	if (scale_ < 0.999f)
		cv::resize(bgr, small, cv::Size(), scale_, scale_, cv::INTER_AREA);
	else
		small = bgr;

	const int w = small.cols;
	const int h = small.rows;
	if (w < 32 || h < 32)
		return candidates;

	cv::Mat gray;
	cv::cvtColor(small, gray, cv::COLOR_BGR2GRAY);
	cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
	cv::Mat enhanced;
	clahe->apply(gray, enhanced);

	// Sólo luminancia. Los canales de saturación/Lab ayudan con fundas de
	// color, pero en mesas oscuras generan miles de bordes falsos y rompen
	// los contornos de las cartas. La detección de cuatro grupos prioriza
	// estabilidad sobre sensibilidad; si un dorso no tiene contraste en
	// luminancia, se puede seguir usando la zona manual.
	cv::Mat edges;
	autoCanny(enhanced, edges);

	cv::Mat morph;
	cv::Mat kernel3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	cv::morphologyEx(edges, morph, cv::MORPH_CLOSE, kernel3);

	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(morph, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

	const double frame_area = (double)w * (double)h;
	const double min_area = std::max(90.0, frame_area * 0.00045);
	const double max_area = frame_area * 0.08;

	for (const auto &contour : contours) {
		const double area = cv::contourArea(contour);
		if (area < min_area || area > max_area)
			continue;

		cv::RotatedRect rr = cv::minAreaRect(contour);
		if (rr.size.width < 10.0f || rr.size.height < 10.0f)
			continue;

		const float rw = rr.size.width;
		const float rh = rr.size.height;
		const float rect_area = rw * rh;
		if (rect_area <= 1.0f)
			continue;

		const float rectangularity = (float)area / rect_area;
		if (rectangularity < 0.64f)
			continue;

		const float short_side = std::min(rw, rh);
		const float long_side = std::max(rw, rh);
		if (short_side < 10.0f)
			continue;

		const float aspect = short_side / long_side;
		if (aspect < 0.46f || aspect > 0.95f)
			continue;

		std::vector<cv::Point> hull;
		cv::convexHull(contour, hull);
		const double hull_area = cv::contourArea(hull);
		if (hull_area <= 1.0) {
			continue;
		}
		const float solidity = (float)(area / hull_area);
		if (solidity < 0.82f)
			continue;

		// Una carta completa debe estar dentro del frame. Si toca el borde
		// no se puede usar para definir una zona fiable.
		cv::Rect2f aabb = aabbOf(rr);
		if (aabb.x < 1.0f || aabb.y < 1.0f ||
		    aabb.x + aabb.width > w - 1.0f ||
		    aabb.y + aabb.height > h - 1.0f)
			continue;

		// Pasar del frame reducido al frame original.
		cv::Point2f src_center(rr.center.x / scale_, rr.center.y / scale_);
		cv::Size2f src_size(rr.size.width / scale_, rr.size.height / scale_);
		cv::RotatedRect src_rr(src_center, src_size, rr.angle);
		cv::Rect2f src_aabb = aabbOf(src_rr);

		bool excluded = false;
		for (const auto &ex : exclude_rects) {
			if (insideExpanded(src_center, ex, 0.10f *
					   std::max(ex.width, ex.height)) ||
			    aabbIou(src_aabb, ex) > 0.30f) {
				excluded = true;
				break;
			}
		}
		if (excluded)
			continue;

		float long_angle = rr.angle;
		if (rw > rh) {
			// rr.angle is the width axis; if width is the long side,
			// that is the long axis. Otherwise add 90°.
			long_angle = rr.angle;
		} else {
			long_angle = rr.angle + 90.0f;
		}
		long_angle = normalizeAngle180(long_angle);
		if (long_angle >= 90.0f)
			long_angle -= 180.0f;
		if (long_angle < -90.0f)
			long_angle += 180.0f;

		Candidate c;
		c.small_rect = rr;
		c.src_rect = src_rr;
		c.small_aabb = aabb;
		c.center = cv::Point2f(rr.center.x, rr.center.y);
		c.area = (float)area;
		c.short_side = short_side;
		c.long_side = long_side;
		c.long_angle = long_angle;
		c.score = (float)area * rectangularity *
			  clampf(1.0f - std::fabs(aspect - 0.72f) / 0.30f,
				 0.25f, 1.0f);
		c.hist = colorHistogram(small, rr);
		candidates.push_back(std::move(c));
	}

	// Si dos o más cartas se tocan, Canny puede devolver un único contorno
	// que las engloba. Ese contorno grande no debe usarse como si fuera una
	// carta: contiene a las cartas individuales y, al ordenar por área, las
	// eliminaría. Se marca como contenedor y se descarta.
	std::vector<char> drop(candidates.size(), 0);
	for (size_t i = 0; i < candidates.size(); ++i) {
		const Candidate &outer = candidates[i];
		const float outer_area =
			std::max(1.0f, outer.small_aabb.width * outer.small_aabb.height);
		std::vector<int> children;
		for (size_t j = 0; j < candidates.size(); ++j) {
			if (i == j)
				continue;
			const Candidate &inner = candidates[j];
			const float ratio = inner.area / outer.area;
			if (ratio < 0.20f || ratio > 0.86f)
				continue;
			const cv::Rect2f inter = inner.small_aabb & outer.small_aabb;
			const float inner_area = std::max(
				1.0f, inner.small_aabb.width * inner.small_aabb.height);
			const float inter_area =
				std::max(0.0f, inter.width * inter.height);
			if (inter_area > 0.88f * inner_area)
				children.push_back((int)j);
		}
		if (children.size() >= 2) {
			bool spread = false;
			for (size_t a = 0; a < children.size() && !spread; ++a) {
				for (size_t b = a + 1; b < children.size(); ++b) {
					const float d = (float)cv::norm(
						candidates[children[a]].center -
						candidates[children[b]].center);
					if (d > 0.15f * std::min(outer.short_side,
								  outer.long_side)) {
						spread = true;
						break;
					}
				}
			}
			if (spread && outer_area > 0.0f)
				drop[i] = 1;
		}
	}
	if (std::any_of(drop.begin(), drop.end(),
			[](char v) { return v != 0; })) {
		std::vector<Candidate> filtered;
		filtered.reserve(candidates.size());
		for (size_t i = 0; i < candidates.size(); ++i)
			if (!drop[i])
				filtered.push_back(std::move(candidates[i]));
		candidates.swap(filtered);
	}

	// Deduplicar contornos interiores (por ejemplo, el dibujo del dorso de
	// la funda) y contornos exteriores repetidos. Se conserva el candidato
	// de mayor área, que normalmente es el borde real de la funda.
	std::sort(candidates.begin(), candidates.end(),
		  [](const Candidate &a, const Candidate &b) {
			  return a.area > b.area;
		  });

	std::vector<Candidate> unique;
	for (auto &c : candidates) {
		bool duplicate = false;
		const float c_aabb_area =
			std::max(1.0f, c.small_aabb.width * c.small_aabb.height);
		for (const auto &k : unique) {
			const float d = (float)cv::norm(c.center - k.center);
			const float tol = 0.30f * std::min(c.short_side, k.short_side);
			// Un patrón interior del dorso puede tener un centro distinto
			// al de la funda; si su caja está casi contenida en una
			// candidata mayor, no es una carta independiente.
			const cv::Rect2f inter = c.small_aabb & k.small_aabb;
			const float inter_area = std::max(
				0.0f, inter.width * inter.height);
			if (d < tol || aabbIou(c.small_aabb, k.small_aabb) > 0.32f ||
			    inter_area > 0.88f * c_aabb_area) {
				duplicate = true;
				break;
			}
		}
		if (!duplicate)
			unique.push_back(std::move(c));
	}
	return splitMergedGrids(unique, small);
}

std::vector<AutoZoneDetector::Candidate> AutoZoneDetector::splitMergedGrids(
	const std::vector<Candidate> &candidates, const cv::Mat &small) const
{
	if (candidates.size() < 4 || small.empty())
		return candidates;

	std::vector<float> areas;
	std::vector<float> shorts;
	std::vector<float> longs;
	areas.reserve(candidates.size());
	shorts.reserve(candidates.size());
	longs.reserve(candidates.size());
	for (const Candidate &c : candidates) {
		areas.push_back(c.area);
		shorts.push_back(c.short_side);
		longs.push_back(c.long_side);
	}
	auto pick_median = [](std::vector<float> values) {
		std::sort(values.begin(), values.end());
		return values.empty() ? 0.0f : values[values.size() / 2];
	};
	const float med_area = pick_median(areas);
	const float med_short = pick_median(shorts);
	const float med_long = pick_median(longs);
	if (med_area <= 0.0f || med_short <= 0.0f || med_long <= 0.0f)
		return candidates;

	const int W = 160;
	const int H = 220;
	std::vector<Candidate> out;
	out.reserve(candidates.size() + 16);

	// Dos cartas consecutivas pueden quedar unidas por un contorno si se
	// tocan. Su rectángulo envolvente es alargado (no llega a 2x2) y el eje
	// largo de ese rectángulo es la dirección en la que están alineadas las
	// dos cartas, no el eje largo de cada carta. Se parte por el centro en
	// dos cartas completas, conservando la orientación de la pareja.
	auto split_merged_pair = [&](const Candidate &c,
				     std::vector<Candidate> &dst) -> bool {
		if (!(c.area > 1.78f * med_area &&
		      c.short_side > 1.28f * med_short &&
		      c.long_side > 1.28f * med_long))
			return false;

		const float Lu = c.long_side;
		const float Su = c.short_side;
		// Caso A: las cartas están alineadas por su eje corto; el lado
		// largo del contorno es 2 * carta_corta. Caso B: alineadas por su
		// eje largo; el lado largo es 2 * carta_larga.
		const float aspect_a = (Lu * 0.5f) / Su;
		const float aspect_b = Su / (Lu * 0.5f);
		const float target = 0.72f;
		const bool along_short =
			std::fabs(aspect_a - target) <= std::fabs(aspect_b - target);
		const float card_short = along_short ? Lu * 0.5f : Su;
		const float card_long = along_short ? Su : Lu * 0.5f;
		const float aspect = card_short / card_long;
		if (aspect < 0.52f || aspect > 0.90f)
			return false;

		// Un pequeño margen evita volver a tocar los bordes del contorno
		// fusionado y mejora la comparación de los dorsos.
		const float cell_short = card_short * 0.92f;
		const float cell_long = card_long * 0.92f;
		const float long_angle =
			along_short ? c.long_angle + 90.0f : c.long_angle;
		const float cell_angle = long_angle + 90.0f;
		const float rad = c.long_angle * (float)CV_PI / 180.0f;
		const cv::Point2f axis(std::cos(rad), std::sin(rad));
		const float offset = c.long_side * 0.25f;

		std::vector<Candidate> cells;
		cells.reserve(2);
		for (int side : {-1, 1}) {
			const cv::Point2f center = c.center + axis * (side * offset);
			const cv::RotatedRect cell_rr(
				center, cv::Size2f(cell_short, cell_long), cell_angle);
			Candidate cell;
			cell.small_rect = cell_rr;
			cell.src_rect = cv::RotatedRect(
				cv::Point2f(center.x / scale_, center.y / scale_),
				cv::Size2f(cell_short / scale_, cell_long / scale_),
				cell_angle);
			cell.small_aabb = aabbOf(cell_rr);
			cell.center = center;
			cell.area = cell_short * cell_long;
			cell.short_side = std::min(cell_short, cell_long);
			cell.long_side = std::max(cell_short, cell_long);
			cell.long_angle = normalizeAngle180(long_angle);
			cell.score = c.score * 0.5f;
			cell.hist = colorHistogram(small, cell_rr);
			cells.push_back(std::move(cell));
		}

		if (cells.size() != 2)
			return false;
		const float sim = histSimilarity(cells[0].hist, cells[1].hist);
		if (sim < 0.45f)
			return false;

		dst.push_back(std::move(cells[0]));
		dst.push_back(std::move(cells[1]));
		return true;
	};

	for (const Candidate &c : candidates) {
		// Un grupo 2x2 fusionado ocupa aproximadamente el doble de alto y
		// ancho que una carta. Se compara con la mediana de los candidatos
		// del frame para no partir una carta con un dibujo en cruz.
		if (!(c.area > 2.2f * med_area &&
		      c.short_side > 1.55f * med_short &&
		      c.long_side > 1.55f * med_long)) {
			if (split_merged_pair(c, out))
				continue;
			out.push_back(c);
			continue;
		}

		cv::Point2f corners[4];
		c.small_rect.points(corners);
		std::vector<cv::Point2f> pts(corners, corners + 4);
		const cv::Point2f cc = c.small_rect.center;
		std::sort(pts.begin(), pts.end(),
			  [&cc](const cv::Point2f &a, const cv::Point2f &b) {
				  return std::atan2(a.y - cc.y, a.x - cc.x) <
					 std::atan2(b.y - cc.y, b.x - cc.x);
			  });
		const std::vector<cv::Point2f> dst = {
			cv::Point2f(0.0f, 0.0f),
			cv::Point2f((float)(W - 1), 0.0f),
			cv::Point2f((float)(W - 1), (float)(H - 1)),
			cv::Point2f(0.0f, (float)(H - 1))};
		cv::Mat M = cv::getPerspectiveTransform(pts, dst);
		cv::Mat patch;
		cv::warpPerspective(small, patch, M, cv::Size(W, H),
				    cv::INTER_AREA, cv::BORDER_REPLICATE);

		cv::Mat gray, gx, gy, mag;
		cv::cvtColor(patch, gray, cv::COLOR_BGR2GRAY);
		cv::Sobel(gray, gx, CV_32F, 1, 0, 3);
		cv::Sobel(gray, gy, CV_32F, 0, 1, 3);
		cv::magnitude(gx, gy, mag);

		// La separación entre dos cartas cae en la línea media, pero los
		// bordes fuertes están a unos píxeles a cada lado. Se mira una
		// banda estrecha y se exige que recorra la mayor parte de la línea.
		const int bx = std::max(3, (int)std::lround(W * 0.055));
		const int by = std::max(3, (int)std::lround(H * 0.055));
		int v_hits = 0, v_total = 0;
		for (int y = H / 10; y < H - H / 10; ++y) {
			float best = 0.0f;
			const int x0 = std::max(0, W / 2 - bx);
			const int x1 = std::min(W - 1, W / 2 + bx);
			for (int x = x0; x <= x1; ++x)
				best = std::max(best, mag.at<float>(y, x));
			v_hits += best > 45.0f ? 1 : 0;
			++v_total;
		}
		int h_hits = 0, h_total = 0;
		for (int x = W / 10; x < W - W / 10; ++x) {
			float best = 0.0f;
			const int y0 = std::max(0, H / 2 - by);
			const int y1 = std::min(H - 1, H / 2 + by);
			for (int y = y0; y <= y1; ++y)
				best = std::max(best, mag.at<float>(y, x));
			h_hits += best > 45.0f ? 1 : 0;
			++h_total;
		}
		const float v_support =
			v_total > 0 ? (float)v_hits / (float)v_total : 0.0f;
		const float h_support =
			h_total > 0 ? (float)h_hits / (float)h_total : 0.0f;

		const float ang = c.small_rect.angle * (float)CV_PI / 180.0f;
		const cv::Point2f u(std::cos(ang), std::sin(ang));
		const cv::Point2f v(-std::sin(ang), std::cos(ang));
		const float group_w = c.small_rect.size.width;
		const float group_h = c.small_rect.size.height;
		std::vector<Candidate> cells;
		cells.reserve(4);
		for (int iy : {-1, 1}) {
			for (int ix : {-1, 1}) {
				const cv::Point2f center =
					c.small_rect.center +
					u * (ix * group_w * 0.25f) +
					v * (iy * group_h * 0.25f);
				const cv::RotatedRect cell_rr(
					center,
					cv::Size2f(group_w * 0.46f,
						   group_h * 0.46f),
					c.small_rect.angle);
				const cv::RotatedRect hist_rr(
					center,
					cv::Size2f(group_w * 0.34f,
						   group_h * 0.34f),
					c.small_rect.angle);

				Candidate cell;
				cell.small_rect = cell_rr;
				cell.src_rect = cv::RotatedRect(
					cv::Point2f(center.x / scale_,
						    center.y / scale_),
					cv::Size2f(group_w * 0.46f / scale_,
						   group_h * 0.46f / scale_),
					c.small_rect.angle);
				cell.small_aabb = aabbOf(cell_rr);
				cell.center = center;
				cell.area = group_w * group_h * 0.46f * 0.46f;
				cell.short_side = std::min(group_w, group_h) * 0.46f;
				cell.long_side = std::max(group_w, group_h) * 0.46f;
				cell.long_angle = c.long_angle;
				cell.score = c.score * 0.5f;
				cell.hist = colorHistogram(small, hist_rr);
				cells.push_back(std::move(cell));
			}
		}

		float sum_sim = 0.0f;
		float min_sim = 1.0f;
		int pairs = 0;
		for (size_t a = 0; a < cells.size(); ++a) {
			for (size_t b = a + 1; b < cells.size(); ++b) {
				const float s =
					histSimilarity(cells[a].hist, cells[b].hist);
				sum_sim += s;
				min_sim = std::min(min_sim, s);
				++pairs;
			}
		}
		const float color_sim = pairs > 0 ? sum_sim / (float)pairs : 0.0f;
		const bool valid = v_support > 0.18f && h_support > 0.18f &&
				   color_sim > 0.55f && min_sim > 0.40f;
		if (valid) {
			for (auto &cell : cells)
				out.push_back(std::move(cell));
		} else {
			out.push_back(c);
		}
	}
	return out;
}

bool AutoZoneDetector::evaluateGroup(
	const std::vector<Candidate> &candidates,
	const std::vector<int> &indices, Group &out) const
{
	if (indices.size() != 4)
		return false;

	std::vector<float> areas, short_sides, long_sides, angles;
	cv::Rect2f bbox;
	bool first = true;
	cv::Point2f center(0.0f, 0.0f);
	for (int idx : indices) {
		const Candidate &c = candidates[idx];
		areas.push_back(c.area);
		short_sides.push_back(c.short_side);
		long_sides.push_back(c.long_side);
		angles.push_back(c.long_angle);
		center += c.center;
		if (first) {
			bbox = c.small_aabb;
			first = false;
		} else {
			const float x0 = std::min(bbox.x, c.small_aabb.x);
			const float y0 = std::min(bbox.y, c.small_aabb.y);
			const float x1 = std::max(bbox.x + bbox.width,
						 c.small_aabb.x + c.small_aabb.width);
			const float y1 = std::max(bbox.y + bbox.height,
						 c.small_aabb.y + c.small_aabb.height);
			bbox = cv::Rect2f(x0, y0, x1 - x0, y1 - y0);
		}
	}
	center *= 0.25f;

	if (bbox.width <= 1.0f || bbox.height <= 1.0f)
		return false;

	const float mean_area =
		std::accumulate(areas.begin(), areas.end(), 0.0f) / 4.0f;
	const float mean_short = std::accumulate(short_sides.begin(),
						 short_sides.end(), 0.0f) / 4.0f;
	const float mean_long = std::accumulate(long_sides.begin(),
						long_sides.end(), 0.0f) / 4.0f;
	if (mean_area <= 1.0f || mean_short <= 1.0f || mean_long <= 1.0f)
		return false;

	// La cripta inicial son cuatro cartas en fila, no un bloque 2x2. Se
	// exige que los centros estén alineados, que la fila sea claramente
	// más larga que una carta y que los huecos entre cartas sean
	// regulares. Así se rechazan grupos 2x2, parejas de dos jugadores y
	// cuatro cartas sueltas que casualmente caen cerca.
	float sxx = 0.0f, syy = 0.0f, sxy = 0.0f;
	for (int idx : indices) {
		const cv::Point2f d = candidates[idx].center - center;
		sxx += d.x * d.x;
		syy += d.y * d.y;
		sxy += d.x * d.y;
	}
	cv::Point2f row_dir(1.0f, 0.0f);
	if (sxx + syy > 1.0f) {
		const float theta = 0.5f * std::atan2(2.0f * sxy, sxx - syy);
		row_dir = cv::Point2f(std::cos(theta), std::sin(theta));
	}
	const cv::Point2f row_perp(-row_dir.y, row_dir.x);
	std::vector<float> row_proj;
	row_proj.reserve(4);
	float max_perp = 0.0f;
	for (int idx : indices) {
		const cv::Point2f d = candidates[idx].center - center;
		row_proj.push_back(d.dot(row_dir));
		max_perp = std::max(max_perp, std::fabs(d.dot(row_perp)));
	}
	std::sort(row_proj.begin(), row_proj.end());
	const float row_span = row_proj.back() - row_proj.front();
	if (row_span < 2.4f * mean_short || max_perp > 0.60f * mean_short)
		return false;
	std::vector<float> row_gaps;
	for (size_t i = 1; i < row_proj.size(); ++i)
		row_gaps.push_back(row_proj[i] - row_proj[i - 1]);
	const float gap_med = medianOf(row_gaps);
	if (gap_med <= 1.0f)
		return false;
	for (float g : row_gaps) {
		if (g < 0.35f * gap_med || g > 2.25f * gap_med)
			return false;
	}

	float var_area = 0.0f;
	for (float a : areas) {
		const float d = (a - mean_area) / mean_area;
		var_area += d * d;
	}
	const float std_area = std::sqrt(var_area / 4.0f);
	const float size_consistency = clampf(1.0f - std_area, 0.0f, 1.0f);

	float angle_sum = 0.0f;
	int angle_n = 0;
	for (int i = 0; i < 4; ++i) {
		for (int j = i + 1; j < 4; ++j) {
			angle_sum += angleDiff180(angles[i], angles[j]);
			angle_n++;
		}
	}
	const float angle_consistency =
		clampf(1.0f - (angle_sum / std::max(1, angle_n)) / 28.0f, 0.0f,
		       1.0f);

	float color_sum = 0.0f;
	int color_n = 0;
	for (int i = 0; i < 4; ++i) {
		for (int j = i + 1; j < 4; ++j) {
			color_sum += histSimilarity(candidates[indices[i]].hist,
						    candidates[indices[j]].hist);
			color_n++;
		}
	}
	const float color_consistency =
		color_n > 0 ? color_sum / color_n : 0.5f;

	float min_dist = FLT_MAX;
	for (int i = 0; i < 4; ++i) {
		for (int j = i + 1; j < 4; ++j) {
			min_dist = std::min(
				min_dist,
				(float)cv::norm(candidates[indices[i]].center -
						candidates[indices[j]].center));
		}
	}
	if (min_dist < 0.38f * mean_short)
		return false;
	const float max_span = std::max(bbox.width, bbox.height);
	if (max_span > 5.5f * mean_long)
		return false;

	const float sum_area = mean_area * 4.0f;
	const float bbox_area = bbox.width * bbox.height;
	const float compactness = clampf(sum_area / bbox_area, 0.0f, 1.0f);
	const float spacing = clampf(
		(min_dist / mean_short - 0.35f) / 1.65f, 0.0f, 1.0f);

	const float score = 0.40f * compactness + 0.22f * size_consistency +
			    0.15f * angle_consistency + 0.15f * color_consistency +
			    0.08f * spacing;

	// La apariencia es importante: cuatro cartas boca abajo con las mismas
	// fundas tienen histogramas muy parecidos. Esto evita aceptar cuatro
	// cartas boca arriba de colores distintos.
	if (score < 0.46f || color_consistency < 0.34f ||
	    angle_consistency < 0.25f)
		return false;

	out.indices = indices;
	out.small_bbox = bbox;
	out.small_center = center;
	out.score = score;
	return true;
}

std::vector<AutoZoneDetector::Group> AutoZoneDetector::groupCandidates(
	const std::vector<Candidate> &candidates) const
{
	std::vector<Group> groups;
	const int n = (int)candidates.size();
	if (n < 4)
		return groups;

	std::vector<int> parent(n);
	std::iota(parent.begin(), parent.end(), 0);
	std::function<int(int)> find = [&](int a) {
		while (parent[a] != a) {
			parent[a] = parent[parent[a]];
			a = parent[a];
		}
		return a;
	};
	auto unite = [&](int a, int b) {
		a = find(a);
		b = find(b);
		if (a != b)
			parent[b] = a;
	};

	for (int i = 0; i < n; ++i) {
		for (int j = i + 1; j < n; ++j) {
			const Candidate &a = candidates[i];
			const Candidate &b = candidates[j];
			const float d = (float)cv::norm(a.center - b.center);
			const float max_dim = std::max(a.long_side, b.long_side);
			const float area_ratio =
				std::min(a.area, b.area) / std::max(a.area, b.area);
			const float ad = angleDiff180(a.long_angle, b.long_angle);
			const float cs = histSimilarity(a.hist, b.hist);
			if (d <= 2.05f * max_dim && area_ratio >= 0.58f &&
			    ad <= 24.0f && cs >= 0.40f &&
			    aabbIou(a.small_aabb, b.small_aabb) < 0.48f) {
				unite(i, j);
			}
		}
	}

	std::unordered_map<int, std::vector<int>> components;
	for (int i = 0; i < n; ++i)
		components[find(i)].push_back(i);

	std::vector<Group> candidates_groups;
	for (auto &kv : components) {
		const std::vector<int> &comp = kv.second;
		if (comp.size() < 4)
			continue;

		if (comp.size() <= 8) {
			// Enumerar todas las combinaciones de 4. Es barato y permite
			// descartar una quinta carta próxima (por ejemplo, la parte
			// superior del mazo de cripta).
			std::vector<int> idx(4);
			for (int a = 0; a < (int)comp.size() - 3; ++a) {
				idx[0] = comp[a];
				for (int b = a + 1; b < (int)comp.size() - 2; ++b) {
					idx[1] = comp[b];
					for (int c = b + 1; c < (int)comp.size() - 1; ++c) {
						idx[2] = comp[c];
						for (int d = c + 1; d < (int)comp.size(); ++d) {
							idx[3] = comp[d];
							Group g;
							if (evaluateGroup(candidates, idx, g))
								candidates_groups.push_back(std::move(g));
						}
					}
				}
			}
		} else {
			// Componentes muy ruidosos: probar cada candidato con sus tres
			// vecinos más próximos, en vez de enumerar combinaciones.
			for (int seed : comp) {
				std::vector<std::pair<float, int>> near;
				for (int other : comp) {
					if (other == seed)
						continue;
					near.push_back({(float)cv::norm(candidates[seed].center -
									candidates[other].center),
							other});
				}
				std::sort(near.begin(), near.end());
				if (near.size() < 3)
					continue;
				std::vector<int> idx = {seed, near[0].second,
							near[1].second, near[2].second};
				Group g;
				if (evaluateGroup(candidates, idx, g))
					candidates_groups.push_back(std::move(g));
			}
		}
	}

	// Non-maximum suppression por cartas compartidas y por solapamiento de
	// las cajas. Así un mismo grupo no se propone varias veces.
	std::sort(candidates_groups.begin(), candidates_groups.end(),
		  [](const Group &a, const Group &b) { return a.score > b.score; });
	std::vector<Group> kept;
	for (auto &g : candidates_groups) {
		bool overlaps = false;
		for (const auto &k : kept) {
			int shared = 0;
			for (int a : g.indices)
				if (std::find(k.indices.begin(), k.indices.end(), a) !=
				    k.indices.end())
					shared++;
			if (shared >= 2 ||
			    aabbIou(g.small_bbox, k.small_bbox) > 0.45f) {
				overlaps = true;
				break;
			}
		}
		if (!overlaps)
			kept.push_back(std::move(g));
	}
	return kept;
}

std::vector<AutoZoneCluster> AutoZoneDetector::detect(
	const cv::Mat &bgr, int64_t now_ns,
	const std::vector<cv::Rect2f> &exclude_rects)
{
	std::vector<Candidate> candidates =
		extractCandidates(bgr, exclude_rects);
	std::vector<Group> groups = groupCandidates(candidates);

	// Convertir los grupos del frame reducido al frame original y buscar su
	// seguimiento temporal.
	std::vector<AutoZoneCluster> incoming;
	incoming.reserve(groups.size());
	for (const auto &g : groups) {
		AutoZoneCluster c;
		c.bbox = cv::Rect2f(g.small_bbox.x / scale_,
				    g.small_bbox.y / scale_,
				    g.small_bbox.width / scale_,
				    g.small_bbox.height / scale_);
		c.center = g.small_center / scale_;
		c.score = g.score;
		std::vector<float> short_sides, long_sides, long_angles;
		for (int idx : g.indices) {
			const Candidate &cand = candidates[idx];
			AutoZoneCard card;
			card.rect = cand.src_rect;
			card.score = cand.score;
			card.hist = cand.hist;
			c.cards.push_back(std::move(card));
			short_sides.push_back(cand.short_side / scale_);
			long_sides.push_back(cand.long_side / scale_);
			long_angles.push_back(cand.long_angle);
		}
		c.card_short = medianOf(short_sides);
		c.card_long = medianOf(long_sides);
		c.median_long_angle_deg = medianOf(long_angles);
		incoming.push_back(std::move(c));
	}

	// Emparejar los grupos de este frame con los tracks existentes. El
	// centro puede moverse unos píxeles por ruido; se exige que el grupo
	// siga siendo el mismo y no se mezclen dos jugadores.
	std::vector<char> used(incoming.size(), 0);
	std::vector<char> track_used(tracks_.size(), 0);
	for (size_t ti = 0; ti < tracks_.size(); ++ti) {
		int best = -1;
		float best_score = -1.0f;
		for (size_t gi = 0; gi < incoming.size(); ++gi) {
			if (used[gi])
				continue;
			const AutoZoneCluster &a = tracks_[ti];
			const AutoZoneCluster &b = incoming[gi];
			const float span =
				std::max({a.bbox.width, a.bbox.height,
					  b.bbox.width, b.bbox.height});
			if (span <= 1.0f)
				continue;
			const float dist = (float)cv::norm(a.center - b.center);
			if (dist > 0.55f * span)
				continue;
			const float area_a = std::max(1.0f, a.card_short * a.card_long);
			const float area_b = std::max(1.0f, b.card_short * b.card_long);
			const float ratio = std::min(area_a, area_b) /
					    std::max(area_a, area_b);
			if (ratio < 0.55f)
				continue;
			const float iou = aabbIou(a.bbox, b.bbox);
			if (iou < 0.08f && dist > 0.25f * span)
				continue;
			const float score = 0.65f * iou +
					    0.35f * (1.0f - dist / span);
			if (score > best_score) {
				best_score = score;
				best = (int)gi;
			}
		}
		if (best >= 0 && best_score > 0.12f) {
			AutoZoneCluster &t = tracks_[ti];
			AutoZoneCluster &b = incoming[best];
			t.bbox = b.bbox;
			t.center = b.center;
			t.cards = std::move(b.cards);
			t.card_short = b.card_short;
			t.card_long = b.card_long;
			t.median_long_angle_deg = b.median_long_angle_deg;
			t.score = b.score;
			t.hits = std::min(1000, t.hits + 1);
			t.misses = 0;
			t.last_seen_ns = now_ns;
			t.stable = t.hits >= 3;
			used[best] = 1;
			track_used[ti] = 1;
		}
	}

	// Tracks que no aparecen en este frame: se conservan un poco para
	// tolerar una oclusión breve (una mano, un movimiento de cámara).
	for (size_t ti = 0; ti < tracks_.size(); ++ti) {
		if (!track_used[ti]) {
			tracks_[ti].misses++;
			tracks_[ti].stable = false;
		}
	}
	tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
				     [](const AutoZoneCluster &c) {
					     return c.misses >= 6;
				     }),
		      tracks_.end());

	// Altas nuevas.
	for (size_t gi = 0; gi < incoming.size(); ++gi) {
		if (used[gi])
			continue;
		AutoZoneCluster c = std::move(incoming[gi]);
		c.id = next_id_++;
		c.hits = 1;
		c.misses = 0;
		c.first_seen_ns = now_ns;
		c.last_seen_ns = now_ns;
		c.stable = false;
		c.player_idx = -1;
		tracks_.push_back(std::move(c));
	}

	// Ordenar por id para que la salida sea determinista.
	std::sort(tracks_.begin(), tracks_.end(),
		  [](const AutoZoneCluster &a, const AutoZoneCluster &b) {
			  return a.id < b.id;
		  });
	return tracks_;
}

AutoZoneGeometry AutoZoneDetector::deriveGeometry(
	const AutoZoneCluster &cluster, const cv::Point2f &table_center,
	const cv::Size &frame_size, bool line_layout)
{
	AutoZoneGeometry out;
	if (cluster.cards.empty() || frame_size.width < 16 ||
	    frame_size.height < 16)
		return out;

	const float fw = (float)frame_size.width;
	const float fh = (float)frame_size.height;
	const float card_short = std::max(8.0f, cluster.card_short);
	const float card_long = std::max(10.0f, cluster.card_long);

	// La fila de cuatro cartas define el eje de la zona. El eje largo de
	// cada carta es aproximadamente radial; el eje de la fila es la
	// tangente. Ajustar la recta con los centros evita que una fila algo
	// torcida produzca una cripta inclinada respecto a las cartas.
	cv::Point2f row_dir(1.0f, 0.0f);
	{
		cv::Point2f mean(0.0f, 0.0f);
		for (const auto &card : cluster.cards)
			mean += card.rect.center;
		mean *= 1.0f / std::max<size_t>(1, cluster.cards.size());
		float sxx = 0.0f, syy = 0.0f, sxy = 0.0f;
		for (const auto &card : cluster.cards) {
			const cv::Point2f d = card.rect.center - mean;
			sxx += d.x * d.x;
			syy += d.y * d.y;
			sxy += d.x * d.y;
		}
		if (sxx + syy > 1.0f) {
			const float theta =
				0.5f * std::atan2(2.0f * sxy, sxx - syy);
			row_dir = cv::Point2f(std::cos(theta), std::sin(theta));
		}
	}

	cv::Point2f forward(-row_dir.y, row_dir.x);
	const cv::Point2f to_center = table_center - cluster.center;
	const float forward_dot = forward.dot(to_center);
	const float min_center_dist = 0.08f * std::min(fw, fh);
	if (std::fabs(forward_dot) >= min_center_dist) {
		if (forward_dot < 0.0f)
			forward = -forward;
	} else if (line_layout) {
		// Fila alineada con el centro de la imagen: elegir el lado de la
		// mesa (arriba o abajo) para que los minions queden por delante.
		const cv::Point2f inward(
			0.0f, cluster.center.y < table_center.y ? 1.0f : -1.0f);
		if (forward.dot(inward) < 0.0f)
			forward = -forward;
	} else {
		// Sin componente perpendicular clara, usar el vector al centro
		// como referencia y conservar la perpendicular de la fila.
		if (to_center.dot(forward) < 0.0f)
			forward = -forward;
	}

	cv::Point2f tangent(-forward.y, forward.x);

	// Proyectar las cuatro cartas sobre los ejes de la zona.
	float min_t = FLT_MAX, max_t = -FLT_MAX;
	float min_f = FLT_MAX, max_f = -FLT_MAX;
	for (const auto &card : cluster.cards) {
		cv::Point2f pts[4];
		card.rect.points(pts);
		for (const auto &p : pts) {
			const cv::Point2f d = p - cluster.center;
			const float pt = d.dot(tangent);
			const float pf = d.dot(forward);
			min_t = std::min(min_t, pt);
			max_t = std::max(max_t, pt);
			min_f = std::min(min_f, pf);
			max_f = std::max(max_f, pf);
		}
	}
	if (min_t > max_t || min_f > max_f)
		return out;

	const float crypt_w = std::max(max_t - min_t + 0.30f * card_short,
				       1.35f * card_short);
	const float crypt_h = std::max(max_f - min_f + 0.30f * card_short,
				       1.30f * card_long);
	const cv::Point2f crypt_center =
		cluster.center + tangent * ((min_t + max_t) * 0.5f) +
		forward * ((min_f + max_f) * 0.5f);

	// La zona de minions es la mitad superior del cuadrado que define la
	// fila: mismo ancho que la cripta y una profundidad de media fila.
	// Antes se usaba 2.4 * largo de carta, que en una fila de cuatro se
	// extendía casi hasta el centro de la mesa y solapaba a todos los
	// jugadores.
	const float gap = 0.28f * card_long;
	float minion_depth = std::min(
		0.32f * fh, std::max(0.50f * crypt_w, 1.2f * card_long));
	if (!line_layout) {
		const float dist_center =
			(float)cv::norm(table_center - cluster.center);
		const float available =
			std::max(0.0f, dist_center - crypt_h * 0.5f - gap);
		if (available > 1.0f)
			minion_depth = std::min(
				minion_depth,
				std::max(0.07f * fh, available * 0.92f));
	}
	const float minion_w = std::max(crypt_w, 2.0f * card_short);
	const cv::Point2f minion_center =
		crypt_center +
		forward * (crypt_h * 0.5f + gap + minion_depth * 0.5f);

	const float angle = std::atan2(tangent.y, tangent.x) * 180.0f / kPI;

	auto clampRect = [&](cv::RotatedRect &rr) {
		rr.size.width = std::min(rr.size.width, 0.88f * fw);
		rr.size.height = std::min(rr.size.height, 0.88f * fh);
		rr.center.x = clampf(rr.center.x, 0.04f * fw, 0.96f * fw);
		rr.center.y = clampf(rr.center.y, 0.04f * fh, 0.96f * fh);
	};

	out.crypt = cv::RotatedRect(crypt_center,
				    cv::Size2f(crypt_w, crypt_h), angle);
	out.minion = cv::RotatedRect(minion_center,
				     cv::Size2f(minion_w, minion_depth), angle);
	clampRect(out.crypt);
	clampRect(out.minion);
	out.zone_angle_deg = angle;

	// El detector de cartas entrega `obj.angle` siguiendo el eje corto de
	// la carta (el ancho del rectángulo), igual que las zonas manuales.
	// En una fila, ese eje corto es precisamente la tangente de la fila,
	// no el eje largo de la carta girado 90 grados.
	float lock_angle = std::atan2(tangent.y, tangent.x) * 180.0f / kPI;
	lock_angle = normalizeAngle180(lock_angle);
	if (lock_angle >= 90.0f)
		lock_angle -= 180.0f;
	if (lock_angle < -90.0f)
		lock_angle += 180.0f;
	out.lock_angle_deg = lock_angle;
	out.valid = true;
	return out;
}

} // namespace vtes_detection
