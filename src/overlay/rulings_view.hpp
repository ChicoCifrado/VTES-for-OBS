#ifndef VTES_OVERLAY_RULINGS_VIEW_HPP
#define VTES_OVERLAY_RULINGS_VIEW_HPP

// Rendering of the official card rulings on the OBS overlay panel.
//
// The rulings live in data/vtes.json (5116 rulings across 2018 cards) and
// VTESCardDatabase already parses them into VTESCardEntry.rulings, but until
// now nothing consumed them. This unit turns that data into a list of lines
// that fits the dark panel on the left of the overlay, so the drawing code in
// detect-filter-obb.cpp only has to iterate and call putText.
//
// It is deliberately free of any dependency on OBS, OpenCV or CUDA: everything
// here is plain C++ over std::string, which makes it directly unit-testable.
// The single exception is the width measurement, which the caller injects as
// a std::function, because OpenCV's putText is the only truth about the width
// of proportional text.

#include <algorithm>
#include <cstddef>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "vtes_database.hpp"

namespace vtes_overlay {

// One line ready to draw. `is_heading` marks the optional grouping label that
// some rulings carry (145 distinct groups exist); the caller renders those in
// a different colour so the block stays scannable.
struct RulingsLine {
    std::string text;
    bool is_heading = false;
};

// Where the rulings block goes, in overlay pixels at the 1920x1080 reference.
//
// Panel geometry measured on the design: the opaque left panel spans x=150 to
// x=446 (296 px wide) once the 3:2 image is scaled to fit a 16:9 scene. The
// rulings are drawn over the detected card, so the caller sets the box on the
// card itself.
struct RulingsBox {
    int x = 158;
    int y = 210;
    int w = 280;
    int h = 290;
};

// Font scale of FONT_HERSHEY_SIMPLEX used to draw the rulings.
//
// Character advance measured with cv::getTextSize over all 8132 real
// ruling texts and group labels of vtes.json at this scale. The value
// depends on the OpenCV build, not only on the font scale: the same
// texts measure 7.07 px/char on OpenCV 5.0.0 (the one the plugin links)
// and 9.15 px/char on 4.13. No fixed advance is right for both, which
// is why wrapWords measures each candidate line instead of counting
// characters.
constexpr double kRulingsFontScale = 0.55;

// Height of the glyph at 0.55 is 13 px; 16 px between baselines keeps the
// lines separable without eating the box.
constexpr int kRulingsLineHeight = 16;

// Fallback width estimate, used only when the caller provides no measurement.
// It is a rough constant per character at kRulingsFontScale, deliberately the
// pessimistic end so that a stubbed caller wraps no wider than OpenCV would.
constexpr int kRulingsFallbackAdvance = 10;

// Returns the pixel width of a laid-out line. The overlay source passes a
// lambda around cv::getTextSize; tests can pass a fixed estimate.
using RulingsMeasure = std::function<int(const std::string&)>;

// Result of laying one card's rulings out for the panel.
//
// The layout holds every line, not only those that fit: the panel is a
// viewport over the full text and the scroll offset decides what is drawn.
struct RulingsLayout {
    std::vector<RulingsLine> lines;  // every line of every ruling, in draw order
    int total = 0;                   // rulings on the card
};

// Wrap text to fit within max_px, measuring every candidate line.
//
// The line grows word by word; the first word that overflows closes it. A
// single word wider than the box is hard-broken by characters rather than
// dropped, so no information is ever lost.
std::vector<std::string> wrapWords(const std::string& text, int max_px,
                                   const RulingsMeasure& measure);

// Build the drawable lines for a card's rulings.
//
// `card` may be null (no card detected): the result is then empty and the
// caller should draw nothing.
RulingsLayout buildRulingsLayout(const VTESCardEntry* card,
                                 const RulingsBox& box,
                                 const RulingsMeasure& measure);

// Height in pixels of the whole rulings text.
inline int contentHeight(const RulingsLayout& layout, int line_height) {
    if (line_height <= 0) return 0;
    return (int)layout.lines.size() * line_height;
}

// How far the rulings can be scrolled inside the box. Zero when everything
// fits, which is the only case where no scrollbar is drawn.
inline int maxScroll(const RulingsLayout& layout, const RulingsBox& box,
                     int line_height) {
    return std::max(0, contentHeight(layout, line_height) - box.h);
}

// Half-open range [first, last) of the lines whose baseline stays inside
// the box when scrolled by scroll_px. putText draws from the baseline
// down, so line i is fully visible only while
// box.y + lh <= box.y + (i+1)*lh - scroll <= box.y + box.h.
//
// The bounds are tight on purpose: the overlay draws into the frame
// directly, with no scissor rectangle, so a line left half outside the
// box would paint its glyphs over the UI below it. The cost is that the
// last line of the box is dropped instead of half-shown.
inline std::pair<size_t, size_t> visibleLineRange(const RulingsLayout& layout,
                                                  const RulingsBox& box,
                                                  int line_height,
                                                  int scroll_px) {
    const size_t n = layout.lines.size();
    if (n == 0 || box.h <= 0 || line_height <= 0) return {0, 0};

    const int scroll = std::max(0, scroll_px);
    // Smallest i whose baseline is at or below box.y + lh: rounding up
    // matters, because a fractional scroll would otherwise admit a line
    // whose glyphs poke out above the box.
    size_t first = (size_t)((scroll + line_height - 1) / line_height);
    if (first > n) first = n;
    // Largest i whose baseline is at or above box.y + box.h, exclusive.
    size_t last = (size_t)((scroll + box.h) / line_height);
    if (last > n) last = n;
    if (last < first) last = first;
    return {first, last};
}

}  // namespace vtes_overlay

#endif  // VTES_OVERLAY_RULINGS_VIEW_HPP
