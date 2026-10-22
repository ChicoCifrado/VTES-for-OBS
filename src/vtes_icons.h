#ifndef VTES_ICONS_H
#define VTES_ICONS_H

// Mapeo de los iconos oficiales de VTES (tipos, clanes/credos y
// disciplinas/virtudes) a los glifos de las fuentes de KRCP:
//   - fonts/ankha2.otf     : tipos, disciplinas, virtudes, conviccion, poder
//   - fonts/vtes-clans.otf : clanes y credos
//
// Los caracteres se obtuvieron del mapa oficial de ANKHA_SYMBOLS (krcg) y,
// para clanes/credos, comparando cada SVG oficial nuevo de static.krcg.org
// con los glifos de vtes-clans.otf (coincidencia de forma practicamente
// exacta). El glifo es la version "nueva" del icono; para los clanes que
// solo tienen version antigua se usa la variante disponible.
//
// Se usa tanto en el overlay (rasterizado con stb_truetype) como en la web
// (fuente servida por /fonts/...), para que ambos muestren lo mismo.

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "vtes_database.hpp"

struct VtesIcon {
	int font = 0;      // 0 = ankha2 (tipos/disciplinas), 1 = vtes-clans
	std::string glyph; // caracter UTF-8 de la fuente
	std::string label; // nombre legible (title/tooltip)
};

namespace vtes_icons {

inline const std::unordered_map<std::string, std::string> &typeMap()
{
	// Caracteres oficiales de ankha2.otf. Los tipos "Vampire" e "Imbued"
	// no llevan icono de tipo: usan el icono de clan/credo.
	static const std::unordered_map<std::string, std::string> m = {
		{"Action", "0"},
		{"Action Modifier", "1"},
		{"Political Action", "2"},
		{"Ally", "3"},
		{"Combat", "4"},
		{"Equipment", "5"},
		{"Reflex", "6"},
		{"Reaction", "7"},
		{"Retainer", "8"},
		{"Master", "9"},
		{"Event", "["},
		{"Conviction", "\xC2\xA4"}, // ¤
		{"Power", "\xC2\xA7"},      // §
	};
	return m;
}

inline const std::unordered_map<std::string, std::string> &clanMap()
{
	// Glifos de vtes-clans.otf. Las letras mayusculas son la version nueva
	// del icono; las minusculas que aparecen aqui son los clanes cuyo icono
	// nuevo vive en minuscula (Banu Haqim, Ministry y varios independientes/
	// Sabbat). Los credos de Imbued usan los digitos 1-7.
	static const std::unordered_map<std::string, std::string> m = {
		{"Abomination", "A"},
		{"Ahrimane", "B"},
		{"Akunanse", "C"},
		{"Avenger", "1"},
		{"Baali", "E"},
		{"Banu Haqim", "n"},
		{"Blood Brother", "F"},
		{"Brujah", "G"},
		{"Brujah antitribu", "H"},
		{"Caitiff", "I"},
		{"Daughter of Cacophony", "J"},
		{"Defender", "2"},
		{"Gangrel", "L"},
		{"Gangrel antitribu", "M"},
		{"Gargoyle", "N"},
		{"Giovanni", "O"},
		{"Guruhi", "P"},
		{"Harbinger of Skulls", "Q"},
		{"Innocent", "3"},
		{"Ishtarri", "R"},
		{"Judge", "4"},
		{"Kiasyd", "S"},
		{"Lasombra", "T"},
		{"Malkavian", "U"},
		{"Malkavian antitribu", "V"},
		{"Martyr", "5"},
		{"Ministry", "r"},
		{"Nagaraja", "W"},
		{"Nosferatu", "X"},
		{"Nosferatu antitribu", "Y"},
		{"Osebo", "Z"},
		{"Pander", "a"},
		{"Ravnos", "b"},
		{"Redeemer", "6"},
		{"Salubri", "c"},
		{"Salubri antitribu", "d"},
		{"Samedi", "e"},
		{"Toreador", "f"},
		{"Toreador antitribu", "g"},
		{"Tremere", "h"},
		{"Tremere antitribu", "i"},
		{"True Brujah", "j"},
		{"Tzimisce", "k"},
		{"Ventrue", "l"},
		{"Ventrue antitribu", "m"},
		{"Visionary", "7"},
		// Hecata no tiene SVG propio en krcg-static (es la continuacion de
		// Giovanni en V5): se usa el icono de Giovanni.
		{"Hecata", "O"},
		// Nombres antiguos, por si aparecen en datos historicos.
		{"Assamite", "D"},
		{"Followers of Set", "K"},
	};
	return m;
}

inline const std::unordered_map<std::string, std::string> &disciplineMap()
{
	// ANKHA_SYMBOLS oficial: minuscula = inferior, MAYUSCULA = superior.
	static const std::unordered_map<std::string, std::string> m = {
		{"abo", "w"}, {"ani", "i"}, {"aus", "a"}, {"cel", "c"},
		{"chi", "k"}, {"dai", "y"}, {"dem", "e"}, {"dom", "d"},
		{"for", "f"}, {"mal", "<"}, {"mel", "m"}, {"myt", "x"},
		{"nec", "n"}, {"obe", "b"}, {"obf", "o"}, {"obl", "\xC3\xB8"}, // ø
		{"obt", "$"}, {"pot", "p"}, {"pre", "r"}, {"pro", "j"},
		{"qui", "q"}, {"san", "g"}, {"ser", "s"}, {"spi", "z"},
		{"str", "+"}, {"tem", "?"}, {"thn", "h"}, {"tha", "t"},
		{"val", "l"}, {"vic", "v"}, {"vis", "u"},

		{"ABO", "W"}, {"ANI", "I"}, {"AUS", "A"}, {"CEL", "C"},
		{"CHI", "K"}, {"DAI", "Y"}, {"DEM", "E"}, {"DOM", "D"},
		{"FOR", "F"}, {"MAL", ">"}, {"MEL", "M"}, {"MYT", "X"},
		{"NEC", "N"}, {"OBE", "B"}, {"OBF", "O"}, {"OBL", "\xC3\x98"}, // Ø
		{"OBT", "\xC2\xA3"},                                           // £
		{"POT", "P"}, {"PRE", "R"}, {"PRO", "J"}, {"QUI", "Q"},
		{"SAN", "G"}, {"SER", "S"}, {"SPI", "Z"}, {"STR", "="},
		{"TEM", "!"}, {"THN", "H"}, {"THA", "T"}, {"VAL", "L"},
		{"VIC", "V"}, {"VIS", "U"},

		// Virtudes de los Imbued.
		{"viz", ")"}, {"def", "@"}, {"jud", "%"}, {"inn", "#"},
		{"mar", "&"}, {"ven", "("}, {"red", "*"},

		// Otros simbolos que pueden aparecer en textos/datos.
		{"FLIGHT", "^"},
		{"MERGED", "\xC2\xB5"},   // µ
		{"CONVICTION", "\xC2\xA4"}, // ¤
		{"POWER", "\xC2\xA7"},      // §
	};
	return m;
}

inline std::string typeGlyph(const std::string &t)
{
	const auto &m = typeMap();
	auto it = m.find(t);
	return it == m.end() ? std::string() : it->second;
}

inline std::string clanGlyph(const std::string &c)
{
	const auto &m = clanMap();
	auto it = m.find(c);
	return it == m.end() ? std::string() : it->second;
}

inline std::string discGlyph(const std::string &d)
{
	const auto &m = disciplineMap();
	auto it = m.find(d);
	return it == m.end() ? std::string() : it->second;
}

// Iconos de una carta en orden de lectura: tipos, clan/credo y disciplinas.
// Se deduplican (una carta puede repetir el mismo simbolo) y se omiten los
// tipos "Vampire"/"Imbued", que ya aportan su clan/credo.
inline std::vector<VtesIcon> forCard(const VTESCardEntry &e)
{
	std::vector<VtesIcon> out;
	out.reserve(e.types.size() + e.clans.size() + e.disciplines.size());
	std::unordered_set<std::string> seen;

	auto add = [&](int font, const std::string &glyph, const std::string &label) {
		if (glyph.empty())
			return;
		const std::string key = std::to_string(font) + ":" + glyph;
		if (!seen.insert(key).second)
			return;
		out.push_back({font, glyph, label});
	};

	for (const auto &t : e.types) {
		if (t == "Vampire" || t == "Imbued")
			continue;
		add(0, typeGlyph(t), t);
	}
	for (const auto &c : e.clans)
		add(1, clanGlyph(c), c);
	for (const auto &d : e.disciplines)
		add(0, discGlyph(d), d);

	return out;
}

} // namespace vtes_icons

#endif // VTES_ICONS_H
