#ifndef VTES_DATABASE_H
#define VTES_DATABASE_H

#include <string>
#include <vector>
#include <unordered_map>
#include <nlohmann/json.hpp>
#include <fstream>
#include <algorithm>
#include <cstring>

struct VTESCardEntry {
    int id = 0;
    std::string name;
    std::string printed_name;
    std::vector<std::string> types;
    std::vector<std::string> clans;
    std::vector<std::string> disciplines;
    int capacity = 0;
    int group = 0;
    std::string card_text;
    std::string url;
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
