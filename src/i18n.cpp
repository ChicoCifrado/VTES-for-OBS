/*
 * Implementacion del sistema de textos (ver i18n.h).
 *
 * Los textos se cargan una vez al arrancar el plugin desde los .ini de
 * data/locale/ y se guardan en dos mapas inmutables (es/en), de modo que
 * tf_text() es seguro desde los hilos del servidor web.
 */

#include "i18n.h"

#include <obs-module.h>
#include <plugin-support.h>

#include <atomic>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <commctrl.h>
#endif

namespace {

std::unordered_map<std::string, std::string> g_text_es;
std::unordered_map<std::string, std::string> g_text_en;
std::atomic<int> g_lang{0}; /* 0 = espanol (nativo), 1 = ingles */
bool g_first_run = false;
bool g_initialized = false;

std::string trim(const std::string &s)
{
    static const char *ws = " \t\r\n";
    size_t begin = s.find_first_not_of(ws);
    if (begin == std::string::npos)
        return {};
    size_t end = s.find_last_not_of(ws);
    return s.substr(begin, end - begin + 1);
}

/* Parser minimo de los .ini (clave="valor", una por linea). */
void load_ini(const std::string &path, std::unordered_map<std::string, std::string> &out)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
        return;

    std::string line;
    while (std::getline(f, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#' || line[0] == ';')
            continue;

        size_t eq = line.find('=');
        if (eq == std::string::npos)
            continue;

        std::string key = trim(line.substr(0, eq));
        std::string val = trim(line.substr(eq + 1));
        if (val.size() >= 2 && val.front() == '"' && val.back() == '"')
            val = val.substr(1, val.size() - 2);
        if (!key.empty())
            out[key] = val;
    }
}

std::string config_file_path()
{
    char *p = obs_module_config_path("language.txt");
    if (!p)
        return {};
    std::string path = p;
    bfree(p);
    return path;
}

void save_language(const char *code)
{
    std::string path = config_file_path();
    if (path.empty())
        return;

    /* El directorio del plugin puede no existir en una instalacion nueva. */
    size_t sep = path.find_last_of("/\\");
    if (sep != std::string::npos) {
        std::error_code ec;
        std::filesystem::create_directories(path.substr(0, sep), ec);
    }

    std::ofstream f(path, std::ios::trunc);
    if (f)
        f << code << "\n";
}

#ifdef _WIN32
typedef HRESULT(WINAPI *TaskDialogIndirectFn)(const TASKDIALOGCONFIG *, int *, int *, BOOL *);

/*
 * TaskDialogIndirect se resuelve en runtime (comctl32 v6): asi el plugin no
 * depende de una importacion estatica y, si no estuviera disponible, se
 * mantiene el idioma nativo sin romper la carga.
 */
TaskDialogIndirectFn resolve_task_dialog()
{
    HMODULE h = GetModuleHandleW(L"comctl32.dll");
    if (!h)
        h = LoadLibraryW(L"comctl32.dll");
    if (!h)
        return nullptr;
    return reinterpret_cast<TaskDialogIndirectFn>(GetProcAddress(h, "TaskDialogIndirect"));
}

/* Dialogo de primera carga: dos botones, Espanol (por defecto) e English. */
int show_language_dialog()
{
    TaskDialogIndirectFn task_dialog = resolve_task_dialog();
    if (!task_dialog)
        return -1;

    TASKDIALOG_BUTTON buttons[2] = {
        {101, L"Español"},
        {102, L"English"},
    };
    TASKDIALOGCONFIG cfg = {};
    cfg.cbSize = sizeof(cfg);
    cfg.pszWindowTitle = L"VTES Card Scanner";
    cfg.pszMainIcon = TD_INFORMATION_ICON;
    cfg.pszMainInstruction = L"Elige el idioma del plugin / Choose the plugin language";
    cfg.pszContent = L"El idioma nativo del plugin es el español.\n"
                     L"Podrás cambiarlo en cualquier momento desde los ajustes del filtro.\n\n"
                     L"The native language of the plugin is Spanish.\n"
                     L"You can change it any time from the filter settings.";
    cfg.cButtons = 2;
    cfg.pButtons = buttons;
    cfg.nDefaultButton = 101; /* nativo: espanol */

    int pressed = 0;
    HRESULT hr = task_dialog(&cfg, &pressed, nullptr, nullptr);
    if (FAILED(hr))
        return -1;
    return pressed;
}
#endif /* _WIN32 */

} // namespace

extern "C" void tf_i18n_init(void)
{
    if (g_initialized)
        return;
    g_initialized = true;

    char *es_path = obs_module_file("locale/es-ES.ini");
    if (es_path) {
        load_ini(es_path, g_text_es);
        bfree(es_path);
    }
    char *en_path = obs_module_file("locale/en-US.ini");
    if (en_path) {
        load_ini(en_path, g_text_en);
        bfree(en_path);
    }

    /* Idioma guardado; si no hay ninguno, primera carga (nativo: espanol). */
    std::string code;
    std::string path = config_file_path();
    if (!path.empty()) {
        std::ifstream f(path);
        if (f)
            std::getline(f, code);
        code = trim(code);
    }
    if (code == "en")
        g_lang.store(1);
    else if (code == "es")
        g_lang.store(0);
    else
        g_first_run = true;

    obs_log(LOG_INFO, "[i18n] Language: %s%s (es:%zu keys, en:%zu keys)", tf_i18n_code(),
            g_first_run ? " [first run]" : "", g_text_es.size(), g_text_en.size());
}

extern "C" int tf_i18n_first_run(void)
{
    return g_first_run ? 1 : 0;
}

extern "C" const char *tf_i18n_code(void)
{
    return g_lang.load() == 1 ? "en" : "es";
}

extern "C" void tf_i18n_set(const char *code)
{
    const bool is_en = code && std::strcmp(code, "en") == 0;
    const int previous = g_lang.exchange(is_en ? 1 : 0);
    g_first_run = false;
    save_language(is_en ? "en" : "es");
    if (previous != (is_en ? 1 : 0))
        obs_log(LOG_INFO, "[i18n] Language changed to: %s", tf_i18n_code());
}

extern "C" void tf_i18n_prompt_if_needed(void)
{
    if (!g_first_run)
        return;

    int choice = -1;
#ifdef _WIN32
    choice = show_language_dialog();
#endif /* _WIN32 */
    /* Si no hay dialogo disponible (o falla) se queda el idioma nativo. */
    tf_i18n_set(choice == 102 ? "en" : "es");
}

extern "C" const char *tf_text(const char *key)
{
    if (!key)
        return "";

    const bool is_en = g_lang.load() == 1;
    const auto &primary = is_en ? g_text_en : g_text_es;
    auto it = primary.find(key);
    if (it != primary.end())
        return it->second.c_str();

    const auto &fallback = is_en ? g_text_es : g_text_en;
    it = fallback.find(key);
    if (it != fallback.end())
        return it->second.c_str();

    return key;
}
