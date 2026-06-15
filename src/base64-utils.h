#ifndef BASE64_UTILS_H
#define BASE64_UTILS_H

#include <string>
#include <vector>
#include <cstdint>

static const char base64_chars[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

inline std::string base64_encode(const uint8_t *data, size_t len)
{
	std::string ret;
	ret.reserve(((len + 2) / 3) * 4);
	int i = 0;
	uint8_t char_array_3[3];
	uint8_t char_array_4[4];

	while (len--) {
		char_array_3[i++] = *(data++);
		if (i == 3) {
			char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
			char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
			char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
			char_array_4[3] = char_array_3[2] & 0x3f;
			for (i = 0; i < 4; i++)
				ret += base64_chars[char_array_4[i]];
			i = 0;
		}
	}

	if (i) {
		for (int j = i; j < 3; j++)
			char_array_3[j] = '\0';
		char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
		char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
		char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
		char_array_4[3] = char_array_3[2] & 0x3f;
		for (int j = 0; j < i + 1; j++)
			ret += base64_chars[char_array_4[j]];
		while (i++ < 3)
			ret += '=';
	}

	return ret;
}

inline std::string base64_encode(const std::vector<uint8_t> &data)
{
	return base64_encode(data.data(), data.size());
}

#endif
