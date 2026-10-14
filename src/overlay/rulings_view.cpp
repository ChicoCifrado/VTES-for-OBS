#include "rulings_view.hpp"

namespace vtes_overlay {

std::vector<std::string> wrapWords(const std::string& text, size_t max_cols) {
    std::vector<std::string> out;
    if (text.empty()) return out;
    if (max_cols == 0) {
        // No room to wrap: hand the text back whole and let the caller decide.
        out.push_back(text);
        return out;
    }

    const size_t n = text.size();
    size_t i = 0;
    while (i < n) {
        while (i < n && text[i] == ' ') i++;
        if (i >= n) break;

        size_t line_end = i + max_cols;
        if (line_end > n) line_end = n;

        // A break is needed only if the text continues past the window. When
        // the window ends exactly on a space (or on the end of the text) the
        // whole window is word content and fits as is: breaking there would
        // drop usable characters from the line.
        size_t brk = line_end;
        if (line_end < n && text[line_end] != ' ') {
            // Break on the last space inside the window. If there is none, the
            // word is longer than the line and has to be hard-broken.
            size_t sp = text.rfind(' ', line_end - 1);
            if (sp != std::string::npos && sp > i) brk = sp;
        }

        while (brk > i && text[brk - 1] == ' ') brk--;
        out.push_back(text.substr(i, brk - i));
        i = brk;
    }
    return out;
}

RulingsLayout buildRulingsLayout(const VTESCardEntry* card,
                                 const RulingsBox& box,
                                 int char_advance,
                                 int line_height) {
    RulingsLayout out;
    if (!card) return out;

    out.total = (int)card->rulings.size();

    const int cols = charsPerLine(box.w, char_advance);
    const int room = linesPerBox(box.h, line_height);
    if (cols <= 0 || room <= 0) return out;

    std::string last_group;
    int used = 0;

    for (const auto& r : card->rulings) {
        // A group heading is emitted only when it changes: 145 distinct groups
        // exist and repeating the heading on every ruling would waste the box.
        // The heading is wrapped like the text because 53 of the 145 groups are
        // longer than one panel line (longest is 56 chars).
        const bool new_group = !r.group.empty() && r.group != last_group;
        std::vector<std::string> head =
            new_group ? wrapWords(r.group, (size_t)cols) : std::vector<std::string>{};

        std::vector<std::string> body = wrapWords(r.text, (size_t)cols);
        const int need = (int)head.size() + (int)body.size();

        if (used + need > room) {
            out.truncated = true;
            break;
        }

        for (auto& h : head) {
            RulingsLine line;
            line.text = std::move(h);
            line.is_heading = true;
            out.lines.push_back(std::move(line));
            used++;
        }
        if (new_group) last_group = r.group;

        for (auto& l : body) {
            RulingsLine line;
            line.text = std::move(l);
            out.lines.push_back(std::move(line));
        }
        used += (int)body.size();
        out.shown++;
    }

    return out;
}

}  // namespace vtes_overlay
