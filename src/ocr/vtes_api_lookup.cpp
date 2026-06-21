#include "vtes_api_lookup.hpp"

#include <obs-module.h>
#include <curl/curl.h>
#include "vendor/nlohmann/json.hpp"
#include "plugin-support.h"

#include <cstring>

static size_t curl_write_cb(char* ptr, size_t size, size_t nmemb, std::string* out) {
    out->append(ptr, size * nmemb);
    return size * nmemb;
}

CardLookupResult lookup_card_by_ocr(const std::string& ocr_text) {
    if (ocr_text.empty()) return {};

    CURL* curl = curl_easy_init();
    if (!curl) return {};

    char* encoded = curl_easy_escape(curl, ocr_text.c_str(), (int)ocr_text.size());
    if (!encoded) {
        curl_easy_cleanup(curl);
        return {};
    }
    std::string url = std::string("http://localhost:8080/api/search?q=") + encoded;
    curl_free(encoded);

    std::string response;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, 500L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS, 200L);
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);

    CURLcode res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        obs_log(LOG_DEBUG, "[vtes-ocr] API unreachable (%s), falling back to fuzzy match",
                curl_easy_strerror(res));
        return {};
    }

    try {
        auto j = nlohmann::json::parse(response);
        if (!j.is_array() || j.empty()) return {};

        auto& card = j[0];
        CardLookupResult result;

        if (card.contains("printed_name") && card["printed_name"].is_string())
            result.printed_name = card["printed_name"].get<std::string>();

        if (card.contains("name") && card["name"].is_string())
            result.canonical_name = card["name"].get<std::string>();

        if (card.contains("id") && card["id"].is_number_integer())
            result.id = card["id"].get<int>();

        if (card.contains("types") && card["types"].is_array() && !card["types"].empty())
            result.type = card["types"][0].get<std::string>();

        if (result.printed_name.empty()) return {};
        return result;

    } catch (const std::exception& e) {
        obs_log(LOG_DEBUG, "[vtes-ocr] JSON parse error: %s", e.what());
        return {};
    }
}
