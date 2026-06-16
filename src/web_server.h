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

class WebServer {
public:
    WebServer() : running_(false), port_(8080), sock_(INVALID_SOCKET) {}

    ~WebServer() { stop(); }

    bool start(int port, const VTESCardDatabase *db) {
        if (running_) stop();
        db_ = db;
        port_ = port;

#ifdef _WIN32
        WSADATA wsa;
        if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) return false;
#endif

        sock_ = socket(AF_INET, SOCK_STREAM, 0);
        if (sock_ == INVALID_SOCKET) return false;

        int opt = 1;
        setsockopt(sock_, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));

        struct sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_port = htons((uint16_t)port);
        addr.sin_addr.s_addr = INADDR_ANY;

        if (bind(sock_, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
            closesocket(sock_); sock_ = INVALID_SOCKET;
            return false;
        }
        if (listen(sock_, 5) != 0) {
            closesocket(sock_); sock_ = INVALID_SOCKET;
            return false;
        }

        running_ = true;
        thread_ = std::thread([this]() { acceptLoop(); });
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
        // Parse the first line: GET /path HTTP/1.1
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

        // Extract query parameter "q" from path
        std::string uri, query;
        const char *qp = strchr(path, '?');
        if (qp) {
            uri = std::string(path, qp - path);
            query = std::string(qp + 1);
        } else {
            uri = path;
        }

        std::string q;
        // Parse query string for "q=" param
        if (!query.empty()) {
            size_t pos = query.find("q=");
            if (pos != std::string::npos) {
                q = urlDecode(query.substr(pos + 2));
               
                size_t amp = q.find('&');
                if (amp != std::string::npos) q.resize(amp);
            }
        }

        if (uri == "/api/search") {
            serveJson(client, q);
        } else if (uri == "/search") {
            serveHtmlResult(client, q);
        } else {
            serveForm(client);
        }
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

    void serveJson(SOCKET client, const std::string &q) {
        nlohmann::json j = nlohmann::json::array();
        if (!q.empty() && db_) {
            std::string lower = toLower(q);
            for (const auto &[id, entry] : db_->all_entries()) {
                if (toLower(entry.name).find(lower) != std::string::npos) {
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
                    j.push_back(card);
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

    void sendAll(SOCKET client, const std::string &data) {
#ifdef _WIN32
        send(client, data.data(), (int)data.size(), 0);
#else
        send(client, data.data(), data.size(), 0);
#endif
    }

    std::string buildFormPage() {
        std::ostringstream h;
        h << "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"UTF-8\">"
          << "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
          << "<title>VTES Card Search</title><style>"
          << PAGE_CSS
          << "</style></head><body><div class=\"container\">"
          << "<h1>VTES Card Search</h1>"
          << "<form class=\"search-box\" action=\"/search\" method=\"get\">"
          << "<input type=\"text\" name=\"q\" placeholder=\"Search by card name...\" autofocus>"
          << "<button type=\"submit\">Search</button></form>"
          << "<div id=\"results\"></div>"
          << "<div class=\"footer\">VTES Card Scanner</div></div></body></html>";
        return h.str();
    }

    std::string buildResultPage(const std::string &q) {
        std::ostringstream h;
        h << "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"UTF-8\">"
          << "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
          << "<title>VTES Card Search</title><style>" << PAGE_CSS
          << "</style></head><body><div class=\"container\">"
          << "<h1>VTES Card Search</h1>"
          << "<form class=\"search-box\" action=\"/search\" method=\"get\">"
          << "<input type=\"text\" name=\"q\" value=\"" << escapeHtml(q) << "\" autofocus>"
          << "<button type=\"submit\">Search</button></form><div id=\"results\">";

        if (!db_) {
            h << "<div class=\"error\">Database not loaded</div></div></div></body></html>";
            return h.str();
        }

        std::string lower = toLower(q);
        int count = 0;
        for (const auto &[id, entry] : db_->all_entries()) {
            if (toLower(entry.name).find(lower) != std::string::npos) {
                h << "<div class=\"result\">"
                  << "<h2>" << escapeHtml(entry.printed_name.empty() ? entry.name : entry.printed_name)
                  << "</h2><div>";
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
            h << "<div class=\"empty\">No cards found for \"" << escapeHtml(q) << "\"</div>";

        h << "</div><div class=\"footer\">VTES Card Scanner</div></div></body></html>";
        return h.str();
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
        ".result{border:1px solid #333;border-radius:6px;padding:16px;margin-bottom:12px;background:#16213e}"
        ".result h2{font-size:1rem;color:#c9a84c;margin-bottom:6px}"
        ".result .type{display:inline-block;background:#0f3460;color:#aaa;font-size:.75rem;padding:2px 8px;border-radius:3px;margin-right:4px}"
        ".result .detail{font-size:.85rem;color:#999;margin-top:4px}"
        ".result img{max-width:200px;margin-top:8px;border-radius:4px;box-shadow:0 2px 8px rgba(0,0,0,.4)}"
        ".result a{color:#c9a84c;text-decoration:none}"
        ".result a:hover{text-decoration:underline}"
        ".empty{text-align:center;color:#666;padding:40px;font-size:1.1rem}"
        ".footer{text-align:center;color:#444;font-size:.75rem;margin-top:40px}";

    std::atomic<bool> running_;
    int port_;
    SOCKET sock_;
    std::thread thread_;
    const VTESCardDatabase *db_ = nullptr;
};

#endif // VTES_WEB_SERVER_H
