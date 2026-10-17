#ifndef VTES_DATABASE_H
#define VTES_DATABASE_H

#include <string>
#include <vector>
#include <unordered_map>
#include <nlohmann/json.hpp>
#include <fstream>
#include <algorithm>
#include <cstring>
#include <cstdlib>

// A rule reference (LSJ ruling citation). The ruling text already embeds the
// citation between brackets, so this only carries the link target.
struct VTESCardReference {
    std::string label;
    std::string url;
};

// One official ruling for a card. 5116 rulings across 2018 cards in vtes.json.
struct VTESCardRuling {
    std::string text;                      // full text, citation already included
    std::string group;                     // optional grouping heading
    std::vector<VTESCardReference> references;
};

struct VTESCardEntry {
    int id = 0;
    std::string name;
    std::string printed_name;
    std::string card_text;
    std::string url;
    std::string name_es;
    std::string name_fr;
    std::vector<std::string> types;
    std::vector<std::string> clans;
    std::vector<std::string> disciplines;
    std::vector<std::string> name_variants;
    std::vector<VTESCardRuling> rulings;
    int capacity = 0;
    int group = 0;

    bool has_rulings() const { return !rulings.empty(); }
};

class VTESCardDatabase {
public:
    bool load(const std::string& json_path) {
        std::ifstream f(json_path);
        if (!f.is_open()) return false;

        try {
            nlohmann::json j;
            f >> j;

            if (!j.is_array()) return false;

            for (const auto& card : j) {
                VTESCardEntry entry;
                entry.id = card.value("id", 0);
                entry.name = card.value("name", "");
                entry.printed_name = card.value("printed_name", "");
                entry.card_text = card.value("card_text", "");
                entry.url = card.value("url", "");

                if (card.contains("types")) {
                    for (const auto& t : card["types"])
                        entry.types.push_back(t.get<std::string>());
                }
                if (card.contains("clans")) {
                    for (const auto& c : card["clans"])
                        entry.clans.push_back(c.get<std::string>());
                }
                if (card.contains("disciplines")) {
                    for (const auto& d : card["disciplines"])
                        entry.disciplines.push_back(d.get<std::string>());
                }
                if (card.contains("name_variants")) {
                    for (const auto& v : card["name_variants"])
                        entry.name_variants.push_back(v.get<std::string>());
                }
                if (card.contains("rulings") && card["rulings"].is_array()) {
                    for (const auto& r : card["rulings"]) {
                        if (!r.is_object()) continue;
                        VTESCardRuling ruling;
                        ruling.text = r.value("text", "");
                        ruling.group = r.value("group", "");

                        // Ruling text always embeds the citation (e.g. "[LSJ 20011022]"),
                        // so only the link target is kept from each reference.
                        if (r.contains("references") && r["references"].is_array()) {
                            for (const auto& ref : r["references"]) {
                                if (!ref.is_object()) continue;
                                VTESCardReference reference;
                                reference.label = ref.value("label", "");
                                reference.url = ref.value("url", "");
                                if (!reference.url.empty())
                                    ruling.references.push_back(std::move(reference));
                            }
                        }

                        // Drop empty rulings: they carry no information for either
                        // the web view or the overlay.
                        if (!ruling.text.empty())
                            entry.rulings.push_back(std::move(ruling));
                    }
                }
                if (card.contains("_i18n") && card["_i18n"].is_object()) {
                    auto &i18n = card["_i18n"];
                    if (i18n.contains("es") && i18n["es"].is_object())
                        entry.name_es = i18n["es"].value("name", "");
                    if (i18n.contains("fr") && i18n["fr"].is_object())
                        entry.name_fr = i18n["fr"].value("name", "");
                }

                if (card.contains("capacity") && !card["capacity"].is_null()) {
                    if (card["capacity"].is_number_integer())
                        entry.capacity = card["capacity"];
                }
                if (card.contains("group") && !card["group"].is_null()) {
                    if (card["group"].is_number_integer())
                        entry.group = card["group"];
                    else if (card["group"].is_string()) {
                        try { entry.group = std::stoi(card["group"].get<std::string>()); }
                        catch (...) {}
                    }
                }

                std::string id_str = std::to_string(entry.id);
                by_id_[id_str] = entry;

                // Index by type
                for (const auto& t : entry.types)
                    ids_by_type_[t].push_back(id_str);
            }

            return !by_id_.empty();
        } catch (const std::exception& e) {
            fprintf(stderr, "[VTESCardDatabase] Error: %s\n", e.what());
            return false;
        }
    }

    const VTESCardEntry* get_by_id(const std::string& id) const {
        auto it = by_id_.find(id);
        return it != by_id_.end() ? &it->second : nullptr;
    }

    const std::vector<std::string>* get_ids_by_type(const std::string& type) const {
        auto it = ids_by_type_.find(type);
        return it != ids_by_type_.end() ? &it->second : nullptr;
    }

    // Build a map of card_id → type strings for the EmbeddingMatcher
    std::unordered_map<std::string, std::vector<std::string>> build_type_map() const {
        std::unordered_map<std::string, std::vector<std::string>> result;
        result.reserve(by_id_.size());
        for (const auto& [id_str, entry] : by_id_)
            result[id_str] = entry.types;
        return result;
    }

    bool is_empty() const { return by_id_.empty(); }
    size_t size() const { return by_id_.size(); }

    // Iterate all card entries (for building OCR name list, etc.)
    // Returns a reference to the internal map
    const std::unordered_map<std::string, VTESCardEntry>& all_entries() const { return by_id_; }

    // Map classifier CardType to vtes.json type strings
    // When the classifier says "Vampire" → search only "Vampire" cards
    // When "Master" → only "Master" cards
    // When "Combat" → only "Combat"
    // When "LibraryAction" → all non-Vampire, non-Master (too broad, returns empty = no filter)
    // Returns empty string when no type filter should be applied
    static std::string classifier_type_to_vtes_filter(int classifier_type_int) {
        switch (classifier_type_int) {
            case 0: return "Vampire";
            case 1: return "Master";
            case 3: return "Reaction";
            case 4: return "Combat";
            case 5: return "Equipment";
            case 6: return "Political Action";
            case 2:  // LibraryAction — too broad, no filter
            default: return "";
        }
    }

private:
    std::unordered_map<std::string, VTESCardEntry> by_id_;
    std::unordered_map<std::string, std::vector<std::string>> ids_by_type_;
};

#endif // VTES_DATABASE_H
