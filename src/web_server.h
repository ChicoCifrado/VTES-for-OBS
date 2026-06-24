#ifndef VTES_WEB_SERVER_H
#define VTES_WEB_SERVER_H

#include "vtes_database.hpp"
#include <string>
#include <thread>
#include <atomic>
#include <memory>
#include <sstream>
#include <cctype>
#include <cstring>
#include <cstdio>
#include <functional>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32")
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <arpa/inet.h>
#define SOCKET int
#define INVALID_SOCKET -1
#define closesocket close
#endif

struct DetectionEntry {
    std::string card_name;
    std::string card_id;
    std::string card_url;
    int64_t timestamp_ns;
};

class WebServer {
public:
    WebServer() : running_(false), port_(8080), sock_(INVALID_SOCKET) {}

    ~WebServer() { stop(); }

    // Callbacks to communicate with the filter instance
    std::function<std::vector<DetectionEntry>()> get_detections_fn;
    std::function<void(const std::string& url)> on_overlay_fn;
    std::function<void()> on_overlay_clear_fn;

    bool start(int port, const VTESCardDatabase *db) {
        if (running_) stop();
        db_ = db;
        port_ = port;

#ifdef _WIN32
        WSADATA wsa;
        if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
            blog(LOG_ERROR, "[WebServer] WSAStartup failed");
            return false;
        }
#endif

        sock_ = socket(AF_INET, SOCK_STREAM, 0);
        if (sock_ == INVALID_SOCKET) {
            blog(LOG_ERROR, "[WebServer] socket() failed");
            return false;
        }

        int opt = 1;
        setsockopt(sock_, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));

        struct sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_port = htons((uint16_t)port);
        addr.sin_addr.s_addr = INADDR_ANY;

        if (bind(sock_, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
            blog(LOG_ERROR, "[WebServer] bind() failed on port %d", port);
            closesocket(sock_); sock_ = INVALID_SOCKET;
            return false;
        }
        if (listen(sock_, 5) != 0) {
            blog(LOG_ERROR, "[WebServer] listen() failed on port %d", port);
            closesocket(sock_); sock_ = INVALID_SOCKET;
            return false;
        }

        running_ = true;
        thread_ = std::thread([this]() { acceptLoop(); });
        blog(LOG_INFO, "[WebServer] Started on port %d", port);
        return true;
    }

    void stop() {
        running_ = false;
        if (sock_ != INVALID_SOCKET) {
            closesocket(sock_); sock_ = INVALID_SOCKET;
        }
        if (thread_.joinable()) thread_.join();
#ifdef _WIN32
        WSACleanup();
#endif
    }

    bool is_running() const { return running_; }
    int port() const { return port_; }

private:
    void acceptLoop() {
        while (running_) {
            struct sockaddr_in client;
            socklen_t clientLen = sizeof(client);
            SOCKET clientSock = accept(sock_, (struct sockaddr*)&client, &clientLen);
            if (clientSock == INVALID_SOCKET) continue;

            // Set timeout for recv
#ifdef _WIN32
            int timeout = 2000;
            setsockopt(clientSock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeout, sizeof(timeout));
#endif

            char buf[4096];
            int n = recv(clientSock, buf, sizeof(buf) - 1, 0);
            if (n > 0) {
                buf[n] = '\0';
                handleRequest(clientSock, buf);
            }
            closesocket(clientSock);
        }
    }

    void handleRequest(SOCKET client, const char *request) {
        char method[16], path[1024];
        path[0] = '\0';
        if (sscanf(request, "%15s %1023s", method, path) < 2) {
            send404(client);
            return;
        }
        if (strcmp(method, "GET") != 0) {
            send405(client);
            return;
        }

        std::string uri, query;
        const char *qp = strchr(path, '?');
        if (qp) {
            uri = std::string(path, qp - path);
            query = std::string(qp + 1);
        } else {
            uri = path;
        }

        std::string q;
        if (!query.empty()) {
            size_t pos = query.find("q=");
            if (pos != std::string::npos) {
                q = urlDecode(query.substr(pos + 2));
                size_t amp = q.find('&');
                if (amp != std::string::npos) q.resize(amp);
            }
        }

        if (uri == "/api/status") {
            serveStatus(client);
        } else if (uri == "/api/search") {
            serveJson(client, q);
        } else if (uri == "/api/detections") {
            serveDetectionsJson(client);
        } else if (uri == "/api/overlay") {
            serveOverlaySet(client, query);
        } else if (uri == "/api/overlay/clear") {
            serveOverlayClear(client);
        } else if (uri == "/search") {
            serveHtmlResult(client, q);
        } else if (uri == "/detections") {
            serveDetectionsPage(client);
        } else if (uri == "/favicon.ico") {
            send204(client);
        } else if (uri == "/") {
            serveForm(client);
        } else {
            send404(client);
        }
    }

    void serveStatus(SOCKET client) {
        nlohmann::json status;
        status["db_loaded"] = db_ && !db_->is_empty();
        status["db_size"] = db_ ? (int)db_->size() : 0;
        status["server_running"] = (bool)running_;
        status["server_port"] = port_;
        std::string body = status.dump(2);
        std::string header =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: application/json; charset=utf-8\r\n"
            "Content-Length: " + std::to_string(body.size()) + "\r\n"
            "Connection: close\r\n"
            "\r\n";
        sendAll(client, header);
        sendAll(client, body);
        fprintf(stderr, "[WebServer] /api/status: db_loaded=%d db_size=%d\n",
            (db_ && !db_->is_empty()) ? 1 : 0, db_ ? (int)db_->size() : 0);
    }

    void serveForm(SOCKET client) {
        std::string body = buildFormPage();
        std::string header =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/html; charset=utf-8\r\n"
            "Content-Length: " + std::to_string(body.size()) + "\r\n"
            "Connection: close\r\n"
            "\r\n";
        sendAll(client, header);
        sendAll(client, body);
    }

    void serveHtmlResult(SOCKET client, const std::string &q) {
        std::string body;
        if (q.empty()) {
            body = buildFormPage();
        } else {
            body = buildResultPage(q);
        }
        std::string header =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/html; charset=utf-8\r\n"
            "Content-Length: " + std::to_string(body.size()) + "\r\n"
            "Connection: close\r\n"
            "\r\n";
        sendAll(client, header);
        sendAll(client, body);
    }

    struct SearchFilter {
        std::vector<std::string> text_terms;
        std::string type_filter;
        std::string clan_filter;
        std::string disc_filter;
        int cap_min = -1;
        int cap_max = -1;
        int group_val = -1;
        std::string text_search;
    };

    static SearchFilter parseQuery(const std::string &q) {
        SearchFilter f;
        std::vector<std::string> tokens = split(q, ' ');
        for (const auto &t : tokens) {
            if (t.size() > 5 && t.substr(0, 5) == "type:") {
                f.type_filter = toLower(t.substr(5));
            } else if (t.size() > 5 && t.substr(0, 5) == "clan:") {
                f.clan_filter = toLower(t.substr(5));
            } else if (t.size() > 5 && t.substr(0, 5) == "disc:") {
                f.disc_filter = toLower(t.substr(5));
            } else if (t.size() > 5 && t.substr(0, 5) == "text:") {
                f.text_search = toLower(t.substr(5));
            } else if (t.size() > 6 && t.substr(0, 6) == "group:") {
                try { f.group_val = std::stoi(t.substr(6)); } catch (...) {}
            } else if (t.size() > 4 && t.substr(0, 4) == "cap:") {
                try {
                    int v = std::stoi(t.substr(4));
                    f.cap_min = v; f.cap_max = v;
                } catch (...) {}
            } else if (t.size() > 4 && t.substr(0, 4) == "cap>") {
                try { f.cap_min = std::stoi(t.substr(4)) + 1; } catch (...) {}
            } else if (t.size() > 4 && t.substr(0, 4) == "cap<") {
                try { f.cap_max = std::stoi(t.substr(4)) - 1; } catch (...) {}
            } else if (!t.empty()) {
                f.text_terms.push_back(toLower(t));
            }
        }
        return f;
    }

    static bool matchesCard(const VTESCardEntry &e, const SearchFilter &f) {
        // Check type filter
        if (!f.type_filter.empty()) {
            bool found = false;
            for (const auto &t : e.types)
                if (toLower(t).find(f.type_filter) != std::string::npos) { found = true; break; }
            if (!found) return false;
        }
        // Check clan filter
        if (!f.clan_filter.empty()) {
            bool found = false;
            for (const auto &c : e.clans)
                if (toLower(c).find(f.clan_filter) != std::string::npos) { found = true; break; }
            if (!found) return false;
        }
        // Check discipline filter
        if (!f.disc_filter.empty()) {
            bool found = false;
            for (const auto &d : e.disciplines)
                if (toLower(d).find(f.disc_filter) != std::string::npos) { found = true; break; }
            if (!found) return false;
        }
        // Check capacity range
        if (f.cap_min >= 0 && e.capacity < f.cap_min) return false;
        if (f.cap_max >= 0 && e.capacity > f.cap_max) return false;
        // Check group
        if (f.group_val >= 0 && e.group != f.group_val) return false;
        // Check text search
        if (!f.text_search.empty())
            if (toLower(e.card_text).find(f.text_search) == std::string::npos) return false;
        // Check plain text terms (AND) — match any name field
        if (!f.text_terms.empty()) {
            for (const auto &term : f.text_terms) {
                bool found = false;
                // Check all name fields
                if (toLower(e.name).find(term) != std::string::npos) found = true;
                else if (toLower(e.printed_name).find(term) != std::string::npos) found = true;
                else if (toLower(e.name_es).find(term) != std::string::npos) found = true;
                else if (toLower(e.name_fr).find(term) != std::string::npos) found = true;
                else {
                    for (const auto &v : e.name_variants)
                        if (toLower(v).find(term) != std::string::npos) { found = true; break; }
                }
                if (!found) return false;
            }
        }
        return true;
    }

    void toJson(nlohmann::json &j, const VTESCardEntry &entry) const {
        nlohmann::json card;
        card["id"] = entry.id;
        card["name"] = entry.name;
        card["printed_name"] = entry.printed_name;
        card["types"] = entry.types;
        card["clans"] = entry.clans;
        card["disciplines"] = entry.disciplines;
        card["capacity"] = entry.capacity;
        card["group"] = entry.group;
        card["card_text"] = entry.card_text;
        card["url"] = entry.url;
        if (!entry.name_es.empty()) card["name_es"] = entry.name_es;
        if (!entry.name_fr.empty()) card["name_fr"] = entry.name_fr;
        j.push_back(std::move(card));
    }

    void serveJson(SOCKET client, const std::string &q) {
        nlohmann::json j = nlohmann::json::array();
        if (!q.empty() && db_) {
            SearchFilter f = parseQuery(q);
            for (const auto &[id, entry] : db_->all_entries()) {
                if (matchesCard(entry, f)) {
                    toJson(j, entry);
                    if ((int)j.size() >= 50) break;
                }
            }
        }
        std::string body = j.dump(2);
        std::string header =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: application/json; charset=utf-8\r\n"
            "Content-Length: " + std::to_string(body.size()) + "\r\n"
            "Connection: close\r\n"
            "\r\n";
        sendAll(client, header);
        sendAll(client, body);
    }

    void serveDetectionsJson(SOCKET client) {
        nlohmann::json j = nlohmann::json::array();
        if (get_detections_fn) {
            auto entries = get_detections_fn();
            for (const auto& e : entries) {
                nlohmann::json card;
                card["card_name"] = e.card_name;
                card["card_id"] = e.card_id;
                card["card_url"] = e.card_url;
                card["timestamp_ns"] = e.timestamp_ns;
                j.push_back(std::move(card));
            }
        }
        std::string body = j.dump(2);
        std::string header =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: application/json; charset=utf-8\r\n"
            "Content-Length: " + std::to_string(body.size()) + "\r\n"
            "Connection: close\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "\r\n";
        sendAll(client, header);
        sendAll(client, body);
    }

    void serveOverlaySet(SOCKET client, const std::string& query) {
        // Extract url= parameter from query string
        std::string url;
        size_t pos = query.find("url=");
        if (pos != std::string::npos) {
            url = urlDecode(query.substr(pos + 4));
            size_t amp = url.find('&');
            if (amp != std::string::npos) url.resize(amp);
        }
        if (on_overlay_fn && !url.empty()) {
            on_overlay_fn(url);
        }
        std::string body = "{\"ok\":true}";
        std::string header =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: application/json; charset=utf-8\r\n"
            "Content-Length: " + std::to_string(body.size()) + "\r\n"
            "Connection: close\r\n"
            "\r\n";
        sendAll(client, header);
        sendAll(client, body);
    }

    void serveOverlayClear(SOCKET client) {
        if (on_overlay_clear_fn) on_overlay_clear_fn();
        std::string body = "{\"ok\":true}";
        std::string header =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: application/json; charset=utf-8\r\n"
            "Content-Length: " + std::to_string(body.size()) + "\r\n"
            "Connection: close\r\n"
            "\r\n";
        sendAll(client, header);
        sendAll(client, body);
    }

    void serveDetectionsPage(SOCKET client) {
        std::ostringstream h;
        h << "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"UTF-8\">"
          << "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
          << "<title>VTES Detecciones del Directo</title><style>"
          << PAGE_CSS
          << "</style></head><body><div class=\"container\">"
          << "<h1>Detecciones del Directo</h1>"
          << "<div class=\"nav-bar\">"
          << "<a href=\"/\">Buscar Cartas</a>"
          << "<a href=\"/detections\" class=\"active\">Detecciones</a>"
          << "<span class=\"status\" id=\"status\">Conectado</span>"
          << "</div>"
          << "<div id=\"grid\" class=\"detection-grid\"></div>"
          << "<script>"
          << "async function loadDetections(){"
          << "const r=await fetch('/api/detections');"
          << "const data=await r.json();"
          << "const grid=document.getElementById('grid');"
          << "grid.innerHTML='';"
          << "if(data.length===0){grid.innerHTML='<div class=\"empty\">Aún no hay detecciones</div>';return}"
          << "for(const d of data){"
          << "const div=document.createElement('div');"
          << "div.className='detection-card';"
          << "const time=new Date(Number(d.timestamp_ns)/1e6);"
          << "const timeStr=time.toLocaleTimeString();"
          << "div.innerHTML=(d.card_url?'<img src=\"'+d.card_url+'\" alt=\"'+d.card_name+'\" loading=\"lazy\">':'')"
          << "+'<div class=\"name\">'+d.card_name+'</div>'"
          << "'<div class=\"time\">'+timeStr+'</div>';"
          << "div.onclick=function(){"
          << "fetch('/api/overlay?url='+encodeURIComponent(d.card_url));"
          << "document.querySelectorAll('.detection-card').forEach(c=>c.classList.remove('active'));"
          << "this.classList.add('active');"
          << "};"
          << "grid.appendChild(div);"
          << "}"
          << "}"
          << "loadDetections();"
          << "setInterval(loadDetections,2000);"
          << "</script>"
          << "<div class=\"footer\">VTES Card Scanner — Haz clic en una carta para mostrarla en el overlay de OBS</div>"
          << "</div></body></html>";
        std::string body = h.str();
        std::string header =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/html; charset=utf-8\r\n"
            "Content-Length: " + std::to_string(body.size()) + "\r\n"
            "Connection: close\r\n"
            "\r\n";
        sendAll(client, header);
        sendAll(client, body);
    }

    void send404(SOCKET client) {
        const char *body = "<h1>404 Not Found</h1>";
        std::string resp = "HTTP/1.1 404 Not Found\r\nContent-Length: " +
            std::to_string(strlen(body)) + "\r\nConnection: close\r\n\r\n" + body;
        sendAll(client, resp);
    }

    void send405(SOCKET client) {
        const char *body = "<h1>405 Method Not Allowed</h1>";
        std::string resp = "HTTP/1.1 405 Method Not Allowed\r\nContent-Length: " +
            std::to_string(strlen(body)) + "\r\nConnection: close\r\n\r\n" + body;
        sendAll(client, resp);
    }

    void send204(SOCKET client) {
        std::string resp = "HTTP/1.1 204 No Content\r\nConnection: close\r\n\r\n";
        sendAll(client, resp);
    }

    void sendAll(SOCKET client, const std::string &data) {
#ifdef _WIN32
        send(client, data.data(), (int)data.size(), 0);
#else
        send(client, data.data(), data.size(), 0);
#endif
    }

    std::string buildFormPage() {
        std::ostringstream h;
        size_t dbSize = (db_ && !db_->is_empty()) ? db_->size() : 0;
        const char *dbStatus = (db_ && !db_->is_empty()) ? "ready" : "empty";
        h << "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"UTF-8\">"
          << "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
          << "<title>VTES Card Search</title><style>"
          << PAGE_CSS
          << "</style></head><body><div class=\"container\">"
          << "<h1>VTES Card Search</h1>"
          << "<form class=\"search-box\" action=\"/search\" method=\"get\">"
          << "<input type=\"text\" name=\"q\" placeholder=\"e.g. Aabbt or type:Vampire clan:Ventrue cap>5\" autofocus>"
          << "<button type=\"submit\">Search</button></form>"
          << "<div class=\"nav-bar\">"
          << "<a href=\"/\">Buscar</a>"
          << "<a href=\"/detections\">Detecciones</a>"
          << "<span class=\"status-bar\">Database: <span class=\"status-" << dbStatus << "\">"
          << dbStatus << "</span>"
          << (dbSize > 0 ? " (" + std::to_string(dbSize) + " cards)" : "")
          << " | <a href=\"/api/status\">API</a>"
          << "</span></div>"
          << "<div class=\"legend\">"
          << "<h3>Search tips</h3>"
          << "<table><tr><td><code>Aabbt</code></td><td>by name (EN / ES / FR / variants)</td></tr>"
          << "<tr><td><code>type:Vampire</code></td><td>card type: Vampire, Master, Combat, etc.</td></tr>"
          << "<tr><td><code>clan:Ventrue</code></td><td>clan</td></tr>"
          << "<tr><td><code>disc:aus</code></td><td>discipline abbreviation</td></tr>"
          << "<tr><td><code>cap:5</code></td><td>exact capacity</td></tr>"
          << "<tr><td><code>cap>5</code> <code>cap&lt;4</code></td><td>capacity range</td></tr>"
          << "<tr><td><code>group:4</code></td><td>group number</td></tr>"
          << "<tr><td><code>text:bleed</code></td><td>search card text</td></tr>"
          << "<tr><td colspan=\"2\">Combine terms: <code>type:Vampire clan:Nosferatu disc:obf cap>4</code></td></tr>"
          << "</table></div>"
          << "<div class=\"footer\">VTES Card Scanner</div></div></body></html>";
        return h.str();
    }

    std::string buildResultPage(const std::string &q) {
        std::ostringstream h;
        size_t dbSize = (db_ && !db_->is_empty()) ? db_->size() : 0;
        const char *dbStatus = (db_ && !db_->is_empty()) ? "ready" : "empty";
        h << "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"UTF-8\">"
          << "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
          << "<title>VTES Card Search</title><style>" << PAGE_CSS
          << "</style></head><body><div class=\"container\">"
          << "<h1>VTES Card Search</h1>"
          << "<form class=\"search-box\" action=\"/search\" method=\"get\">"
          << "<input type=\"text\" name=\"q\" value=\"" << escapeHtml(q) << "\" autofocus>"
          << "<button type=\"submit\">Search</button></form>"
          << "<div class=\"nav-bar\">"
          << "<a href=\"/\">Buscar</a>"
          << "<a href=\"/detections\">Detecciones</a>"
          << "<span class=\"status-bar\">Database: <span class=\"status-" << dbStatus << "\">"
          << dbStatus << "</span>"
          << (dbSize > 0 ? " (" + std::to_string(dbSize) + " cards)" : "")
          << " | <a href=\"/\">tips</a>"
          << "</span></div>"
          << "<div id=\"results\">";

        if (!db_ || db_->is_empty()) {
            h << "<div class=\"error\">Database not loaded or empty"
              << (db_ ? ". Check OBS log for details." : "")
              << "</div></div></div></body></html>";
            return h.str();
        }

        SearchFilter f = parseQuery(q);
        int count = 0;
        for (const auto &[id, entry] : db_->all_entries()) {
            if (matchesCard(entry, f)) {
                // Determine display name — show localised name as subtitle
                std::string display_name = entry.printed_name.empty() ? entry.name : entry.printed_name;
                h << "<div class=\"result\">"
                  << "<h2>" << escapeHtml(display_name) << "</h2>";
                if (!entry.name_es.empty() || !entry.name_fr.empty()) {
                    h << "<div class=\"i18n\">";
                    if (!entry.name_es.empty()) h << "ES: " << escapeHtml(entry.name_es);
                    if (!entry.name_es.empty() && !entry.name_fr.empty()) h << " &middot; ";
                    if (!entry.name_fr.empty()) h << "FR: " << escapeHtml(entry.name_fr);
                    h << "</div>";
                }
                h << "<div>";
                for (const auto &t : entry.types)
                    h << "<span class=\"type\">" << escapeHtml(t) << "</span>";
                h << "</div><div class=\"detail\">";
                if (!entry.clans.empty()) {
                    h << "Clan: ";
                    for (size_t i = 0; i < entry.clans.size(); i++) {
                        if (i > 0) h << "/";
                        h << escapeHtml(entry.clans[i]);
                    }
                }
                if (entry.capacity > 0) h << " | Cap: " << entry.capacity;
                if (entry.group > 0) h << " | Group: " << entry.group;
                if (!entry.disciplines.empty()) {
                    h << " | ";
                    for (size_t i = 0; i < entry.disciplines.size(); i++) {
                        if (i > 0) h << "/";
                        h << escapeHtml(entry.disciplines[i]);
                    }
                }
                h << "</div>";
                if (!entry.url.empty())
                    h << "<img src=\"" << entry.url << "\" alt=\"" << escapeHtml(entry.name) << "\" loading=\"lazy\">";
                h << "</div>";
                count++;
                if (count >= 50) {
                    h << "<div class=\"empty\">Showing first 50 results. Narrow your search.</div>";
                    break;
                }
            }
        }

        if (count == 0)
            h << "<div class=\"empty\">No cards found</div>";

        h << "</div><div class=\"footer\">VTES Card Scanner</div></div></body></html>";
        return h.str();
    }

    static std::vector<std::string> split(const std::string &s, char delim) {
        std::vector<std::string> out;
        size_t start = 0, end;
        while ((end = s.find(delim, start)) != std::string::npos) {
            if (end > start) out.push_back(s.substr(start, end - start));
            start = end + 1;
        }
        if (start < s.size()) out.push_back(s.substr(start));
        return out;
    }

    static std::string toLower(const std::string &s) {
        std::string out;
        out.reserve(s.size());
        for (char c : s) out.push_back((char)std::tolower((unsigned char)c));
        return out;
    }

    static std::string escapeHtml(const std::string &s) {
        std::string out;
        out.reserve(s.size());
        for (char c : s) {
            switch (c) {
                case '&': out += "&amp;"; break;
                case '<': out += "&lt;"; break;
                case '>': out += "&gt;"; break;
                case '"': out += "&quot;"; break;
                default: out += c;
            }
        }
        return out;
    }

    static std::string urlDecode(const std::string &s) {
        std::string out;
        out.reserve(s.size());
        for (size_t i = 0; i < s.size(); i++) {
            if (s[i] == '%' && i + 2 < s.size()) {
                int hi = s[i+1], lo = s[i+2];
                auto hex = [](int c) -> int {
                    if (c >= '0' && c <= '9') return c - '0';
                    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
                    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
                    return 0;
                };
                out.push_back((char)((hex(hi) << 4) | hex(lo)));
                i += 2;
            } else if (s[i] == '+') {
                out.push_back(' ');
            } else {
                out.push_back(s[i]);
            }
        }
        return out;
    }

    static constexpr const char *PAGE_CSS =
        "*{box-sizing:border-box;margin:0;padding:0}"
        "body{font-family:system-ui,-apple-system,sans-serif;background:#1a1a2e;color:#e0e0e0;padding:20px}"
        ".container{max-width:960px;margin:0 auto}"
        "h1{color:#c9a84c;font-size:1.5rem;margin-bottom:20px}"
        ".search-box{display:flex;gap:8px;margin-bottom:24px}"
        ".search-box input{flex:1;padding:10px 14px;border:1px solid #333;border-radius:6px;font-size:1rem;background:#16213e;color:#e0e0e0;outline:none}"
        ".search-box input:focus{border-color:#c9a84c}"
        ".search-box button{padding:10px 20px;background:#c9a84c;color:#1a1a2e;border:none;border-radius:6px;font-size:1rem;font-weight:600;cursor:pointer}"
        ".search-box button:hover{background:#dbb58c}"
        ".nav-bar{display:flex;gap:12px;margin-bottom:20px;align-items:center;padding:8px 12px;background:#0d1b3e;border-radius:4px}"
        ".nav-bar a{color:#c9a84c;text-decoration:none;padding:6px 12px;border:1px solid #333;border-radius:4px;font-size:.85rem}"
        ".nav-bar a:hover{background:#0f3460}"
        ".nav-bar .status-bar,.nav-bar .status{font-size:.8rem;color:#888;margin-left:auto}"
        ".detection-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:16px;padding:0}"
        ".detection-card{border:1px solid #333;border-radius:8px;padding:12px;background:#16213e;cursor:pointer;transition:border-color .2s}"
        ".detection-card:hover{border-color:#c9a84c}"
        ".detection-card img{width:100%;border-radius:4px;margin-bottom:8px}"
        ".detection-card .name{color:#c9a84c;font-size:.85rem;text-align:center}"
        ".detection-card .time{color:#666;font-size:.7rem;text-align:center;margin-top:4px}"
        ".detection-card.active{border-color:#4caf50;box-shadow:0 0 12px rgba(76,175,80,.3)}"
        ".status-bar{margin-bottom:20px;font-size:.85rem;color:#888;padding:8px 12px;background:#0d1b3e;border-radius:4px}"
        ".status-bar a{color:#c9a84c;text-decoration:none}"
        ".status-bar a:hover{text-decoration:underline}"
        ".status-ready{color:#4caf50;font-weight:600}"
        ".status-empty{color:#ff5722;font-weight:600}"
        ".legend{margin-top:20px;padding:16px;background:#0d1b3e;border-radius:6px;font-size:.85rem}"
        ".legend h3{color:#c9a84c;font-size:.95rem;margin-bottom:10px}"
        ".legend table{width:100%}"
        ".legend td{padding:3px 8px;vertical-align:top}"
        ".legend td:first-child{white-space:nowrap;color:#aaa}"
        ".legend code{color:#c9a84c;background:#1a1a2e;padding:1px 5px;border-radius:3px;font-size:.8rem}"
        ".result{border:1px solid #333;border-radius:6px;padding:16px;margin-bottom:12px;background:#16213e}"
        ".result h2{font-size:1rem;color:#c9a84c;margin-bottom:4px}"
        ".result .i18n{font-size:.8rem;color:#888;margin-bottom:4px}"
        ".result .type{display:inline-block;background:#0f3460;color:#aaa;font-size:.75rem;padding:2px 8px;border-radius:3px;margin-right:4px}"
        ".result .detail{font-size:.85rem;color:#999;margin-top:4px}"
        ".result img{max-width:200px;margin-top:8px;border-radius:4px;box-shadow:0 2px 8px rgba(0,0,0,.4)}"
        ".result a{color:#c9a84c;text-decoration:none}"
        ".result a:hover{text-decoration:underline}"
        ".empty{text-align:center;color:#666;padding:40px;font-size:1.1rem}"
        ".error{text-align:center;color:#ff5722;padding:40px;font-size:1.1rem}"
        ".footer{text-align:center;color:#444;font-size:.75rem;margin-top:40px}";

    std::atomic<bool> running_;
    int port_;
    SOCKET sock_;
    std::thread thread_;
    const VTESCardDatabase *db_ = nullptr;
};

#endif // VTES_WEB_SERVER_H
