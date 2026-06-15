#ifndef WS_CLIENT_H
#define WS_CLIENT_H

#include <string>
#include <vector>
#include <cstring>
#include <stdexcept>

#ifndef _WIN32
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <netdb.h>
#else
#ifndef _WINSOCK2API_
#include <winsock2.h>
#include <ws2tcpip.h>
#endif
#pragma comment(lib, "ws2_32.lib")
#endif

class WebSocketClient {
public:
#ifdef _WIN32
	SOCKET sockfd;
#else
	int sockfd;
#endif

	WebSocketClient() : sockfd(INVALID_SOCKET), connected(false) {}

	~WebSocketClient() { disconnect(); }

	bool connect(const std::string &host, int port)
	{
		if (connected) disconnect();

#ifndef _WIN32
		sockfd = socket(AF_INET, SOCK_STREAM, 0);
		if (sockfd < 0) return false;

		struct sockaddr_in serv_addr;
		std::memset(&serv_addr, 0, sizeof(serv_addr));
		serv_addr.sin_family = AF_INET;
		serv_addr.sin_port = htons(port);

		if (inet_pton(AF_INET, host.c_str(), &serv_addr.sin_addr) <= 0) {
			struct hostent *he = gethostbyname(host.c_str());
			if (!he) {
				close(sockfd);
				sockfd = -1;
				return false;
			}
			std::memcpy(&serv_addr.sin_addr, he->h_addr_list[0], he->h_length);
		}

		if (::connect(sockfd, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
			close(sockfd);
			sockfd = -1;
			return false;
		}
#else
		WSADATA wsaData;
		WSAStartup(MAKEWORD(2, 2), &wsaData);
		sockfd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
		if (sockfd == INVALID_SOCKET) return false;

		struct sockaddr_in serv_addr;
		std::memset(&serv_addr, 0, sizeof(serv_addr));
		serv_addr.sin_family = AF_INET;
		serv_addr.sin_port = htons(static_cast<u_short>(port));

		if (inet_pton(AF_INET, host.c_str(), &serv_addr.sin_addr) <= 0) {
			closesocket(sockfd);
			sockfd = INVALID_SOCKET;
			return false;
		}

		if (::connect(sockfd, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) == SOCKET_ERROR) {
			closesocket(sockfd);
			sockfd = INVALID_SOCKET;
			return false;
		}
#endif

		std::string handshake = "GET / HTTP/1.1\r\n"
					"Host: " + host + ":" + std::to_string(port) + "\r\n"
					"Upgrade: websocket\r\n"
					"Connection: Upgrade\r\n"
					"Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
					"Sec-WebSocket-Version: 13\r\n"
					"\r\n";

		send_raw(handshake);

		char response[1024];
		int bytes = recv_raw(response, sizeof(response) - 1);
		if (bytes <= 0) {
			disconnect();
			return false;
		}
		response[bytes] = '\0';

		std::string resp(response);
		if (resp.find("101") == std::string::npos) {
			disconnect();
			return false;
		}

		connected = true;
		return true;
	}

	void disconnect()
	{
#ifdef _WIN32
		if (sockfd != INVALID_SOCKET) {
			closesocket(sockfd);
			WSACleanup();
#else
		if (sockfd >= 0) {
			close(sockfd);
#endif
		}
		sockfd = INVALID_SOCKET;
		connected = false;
	}

	bool send_text(const std::string &message)
	{
		if (!connected || sockfd == INVALID_SOCKET) return false;

		std::vector<uint8_t> frame;
		frame.push_back(0x81);

		size_t len = message.size();
		if (len < 126) {
			frame.push_back(128 | static_cast<uint8_t>(len));
		} else if (len < 65536) {
			frame.push_back(128 | 126);
			frame.push_back(static_cast<uint8_t>((len >> 8) & 0xFF));
			frame.push_back(static_cast<uint8_t>(len & 0xFF));
		} else {
			frame.push_back(128 | 127);
			for (int i = 7; i >= 0; i--) {
				frame.push_back(static_cast<uint8_t>((len >> (8 * i)) & 0xFF));
			}
		}

		uint8_t mask[4] = {0x12, 0x34, 0x56, 0x78};
		frame.insert(frame.end(), mask, mask + 4);

		for (size_t i = 0; i < len; i++) {
			frame.push_back(static_cast<uint8_t>(message[i] ^ mask[i % 4]));
		}

#ifndef _WIN32
		return ::send(sockfd, frame.data(), static_cast<int>(frame.size()), MSG_NOSIGNAL) > 0;
#else
		return ::send(sockfd, reinterpret_cast<const char *>(frame.data()), static_cast<int>(frame.size()), 0) > 0;
#endif
	}

	bool is_connected() const { return connected; }

private:
	bool connected;

	int send_raw(const std::string &data)
	{
#ifndef _WIN32
		return ::send(sockfd, data.c_str(), static_cast<int>(data.size()), MSG_NOSIGNAL);
#else
		return ::send(sockfd, data.c_str(), static_cast<int>(data.size()), 0);
#endif
	}

	int recv_raw(char *buf, int len)
	{
#ifndef _WIN32
		return ::recv(sockfd, buf, len, 0);
#else
		return ::recv(sockfd, buf, len, 0);
#endif
	}
};

#endif
