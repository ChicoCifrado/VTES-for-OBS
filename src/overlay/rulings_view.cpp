#include "rulings_view.hpp"

#include <string>
#include <vector>

namespace vtes_overlay {

std::vector<std::string> wrapWords(const std::string& text, int max_px,
                                   const RulingsMeasure& measure) {
    std::vector<std::string> out;
    if (text.empty() || max_px <= 0) return out;

    // A stubbed caller must still wrap conservatively: the fallback estimate is
    // deliberately wider than the mean glyph, so a test never wraps tighter
    // than the real renderer would.
    const RulingsMeasure width =
        measure ? measure : RulingsMeasure([](const std::string& s) {
            return (int)(s.size() * kRulingsFallbackAdvance);
        });

    // Split on runs of spaces; the source texts use single spaces, but any
    // number of them is treated as one separator.
    std::vector<std::string> words;
    size_t i = 0;
    while (i < text.size()) {
        while (i < text.size() && text[i] == ' ') ++i;
        if (i >= text.size()) break;
        const size_t start = i;
        while (i < text.size() && text[i] != ' ') ++i;
        words.push_back(text.substr(start, i - start));
    }
    if (words.empty()) return out;

    std::string current;
    auto close_line = [&out, &current]() {
        if (!current.empty()) {
            out.push_back(current);
            current.clear();
        }
    };

    for (const auto& word : words) {
        const std::string candidate =
            current.empty() ? word : current + " " + word;
        if (width(candidate) <= max_px) {
            current = candidate;
            continue;
        }

        // The word does not fit after what the line already holds.
        close_line();
        if (width(word) <= max_px) {
            current = word;
            continue;
        }

        // The word alone is wider than the whole box: break it by characters.
        std::string chunk;
        for (char c : word) {
            const std::string grown = chunk + c;
            if (!chunk.empty() && width(grown) > max_px) {
                out.push_back(chunk);
                chunk.clear();
            }
            chunk += c;
        }
        current = chunk;
    }
    close_line();

    return out;
}

RulingsLayout buildRulingsLayout(const VTESCardEntry* card,
                                 const RulingsBox& box,
                                 const RulingsMeasure& measure) {
    RulingsLayout out;
    if (!card || box.w <= 0) return out;

    out.total = (int)card->rulings.size();

    std::string last_group;
    for (const auto& r : card->rulings) {
        // A group heading is emitted only when it changes: 145 distinct groups
        // exist and repeating the heading on every ruling would waste the box.
        const bool new_group = !r.group.empty() && r.group != last_group;
        const std::vector<std::string> head =
            new_group ? wrapWords(r.group, box.w, measure)
                      : std::vector<std::string>{};
        const std::vector<std::string> body = wrapWords(r.text, box.w, measure);

        for (auto& h : head) {
            RulingsLine line;
            line.text = std::move(h);
            line.is_heading = true;
            out.lines.push_back(std::move(line));
        }
        if (new_group) last_group = r.group;

        for (auto& b : body) {
            RulingsLine line;
            line.text = std::move(b);
            line.is_heading = false;
            out.lines.push_back(std::move(line));
        }
    }

    return out;
}

}  // namespace vtes_overlay
