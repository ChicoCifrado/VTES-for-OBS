#ifndef VTES_API_LOOKUP_HPP
#define VTES_API_LOOKUP_HPP

#include <string>

struct CardLookupResult {
    std::string printed_name;
    std::string canonical_name;
    int id = 0;
    std::string type;
};

CardLookupResult lookup_card_by_ocr(const std::string& ocr_text);

#endif // VTES_API_LOOKUP_HPP
