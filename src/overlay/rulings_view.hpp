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

#include <cstddef>
#include <string>
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
// card sits over the ankh from y=507. x/y are only a starting point: the
// caller must set y below the player pool (whose height depends on
// player_count) and h up to the ankh, so the box never collides with either.
struct RulingsBox {
    int x = 158;
    int y = 210;
    int w = 280;
    int h = 290;
};

// Font scale of FONT_HERSHEY_SIMPLEX used to draw the rulings.
//
// Character advance measured with cv::getTextSize over the 5116 real ruling
// texts of vtes.json at this scale: mean 4.38 px/char, p90 4.59, worst case
// 4.83. The wrap must never overflow, so the constant is the worst case
// rounded up, not the mean — at 7 px/char the text wraps ~40 % short of what
// the panel actually fits and rulings that would fit get dropped.
constexpr double kRulingsFontScale = 0.38;
constexpr int kRulingsCharAdvance = 5;
// Glyph height at 0.38 is 10 px; 16 px between baselines keeps the lines
// separable without eating the box.
constexpr int kRulingsLineHeight = 16;

// Result of laying one card's rulings out for the panel.
struct RulingsLayout {
    std::vector<RulingsLine> lines;  // already wrapped, in draw order
    int shown = 0;                   // rulings actually placed
    int total = 0;                   // rulings the card has
    bool truncated = false;          // true when not everything fitted
};

// Characters that fit on one line of a box of the given width.
inline int charsPerLine(int box_w, int char_advance = kRulingsCharAdvance) {
    if (box_w <= 0 || char_advance <= 0) return 0;
    return box_w / char_advance;
}

// Lines that fit in a box of the given height.
inline int linesPerBox(int box_h, int line_height = kRulingsLineHeight) {
    if (box_h <= 0 || line_height <= 0) return 0;
    return box_h / line_height;
}

// Wrap text to at most max_cols characters, breaking on spaces only.
// A word longer than the line is hard-broken rather than dropped, so no
// information is ever lost.
std::vector<std::string> wrapWords(const std::string& text, size_t max_cols);

// Build the drawable lines for a card's rulings.
//
// `card` may be null (no card detected): the result is then empty and the
// caller should draw nothing. When the card has more rulings than fit in the
// box, the first ones are kept and `truncated` is set so the caller can show
// a "+N more" marker.
RulingsLayout buildRulingsLayout(const VTESCardEntry* card,
                                 const RulingsBox& box,
                                 int char_advance = kRulingsCharAdvance,
                                 int line_height = kRulingsLineHeight);

}  // namespace vtes_overlay

#endif  // VTES_OVERLAY_RULINGS_VIEW_HPP
