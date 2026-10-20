#pragma once

// NInferno-Setup.exe's words: English, Russian, Ukrainian, chosen by the Windows display language.

#include <windows.h>

namespace setup {

enum class Word {
    Title,
    Subtitle,
    Install,
    Cancel,
    Close,
    Open,
    Gpu,
    NoGpu,
    NoDriver,
    GpuUnsupported,
    DriverOld,
    StepGpu,
    StepWsl,
    StepDownload,
    StepInstall,
    StepShortcuts,
    Done,
    Failed,
    WslNeeded,
    WslRestart,
    WslAdmin,
    Downloading,
    Verifying,
    Unpacking,
    Installing,
    UninstallAsk,
    UninstallDone,
    UninstallAskNative,
    UpdateAvailable,
    UpdateRequired,
    UpToDate,
    Updating,
    TrayOpen,
    TrayUpdate,
    TrayQuit,
    Starting,
    StartFailed,
    Folder,
    DiskSpace,
    NewerSetup,
    LogHint,
};

inline LANGID ui_language() { return PRIMARYLANGID(GetUserDefaultUILanguage()); }

inline const wchar_t* word(Word w) {
    const LANGID lang = ui_language();
    const int column  = lang == LANG_RUSSIAN ? 1 : lang == LANG_UKRAINIAN ? 2 : 0;
    static const wchar_t* table[][3] = {
        {L"NInferno", L"NInferno", L"NInferno"},
        {L"Fast local language models on your NVIDIA GPU",
         L"Быстрые локальные языковые модели на вашей видеокарте NVIDIA",
         L"Швидкі локальні мовні моделі на вашій відеокарті NVIDIA"},
        {L"Install", L"Установить", L"Встановити"},
        {L"Cancel", L"Отмена", L"Скасувати"},
        {L"Close", L"Закрыть", L"Закрити"},
        {L"Open NInferno", L"Открыть NInferno", L"Відкрити NInferno"},
        {L"GPU: %s, %s, driver %s", L"Видеокарта: %s, %s, драйвер %s",
         L"Відеокарта: %s, %s, драйвер %s"},
        {L"No NVIDIA GPU found.", L"Видеокарта NVIDIA не найдена.",
         L"Відеокарту NVIDIA не знайдено."},
        {L"The NVIDIA driver is not installed: get it from nvidia.com/drivers, then run this "
         L"again.",
         L"Драйвер NVIDIA не установлен: скачайте его на nvidia.com/drivers и запустите установку "
         L"снова.",
         L"Драйвер NVIDIA не встановлено: завантажте його з nvidia.com/drivers і запустіть "
         L"встановлення знову."},
        {L"This GPU (compute capability %s) is not supported: NInferno needs compute capability "
         L"7.0 or later (V100, RTX 20 series and newer).",
         L"Эта видеокарта (compute capability %s) не поддерживается: NInferno нужна compute "
         L"capability 7.0 или новее (V100, RTX 20 и новее).",
         L"Ця відеокарта (compute capability %s) не підтримується: NInferno потрібна compute "
         L"capability 7.0 або новіша (V100, RTX 20 і новіші)."},
        {L"Driver %s is older than NInferno asks for (%d): update it at nvidia.com/drivers if it "
         L"does not start.",
         L"Драйвер %s старее нужного (%d): обновите его на nvidia.com/drivers, если NInferno не "
         L"запустится.",
         L"Драйвер %s старіший за потрібний (%d): оновіть його на nvidia.com/drivers, якщо "
         L"NInferno не запуститься."},
        {L"Checking the GPU", L"Проверка видеокарты", L"Перевірка відеокарти"},
        {L"Preparing the installation", L"Подготовка установки", L"Підготовка встановлення"},
        {L"Downloading NInferno", L"Загрузка NInferno", L"Завантаження NInferno"},
        {L"Installing", L"Установка", L"Встановлення"},
        {L"Start menu and desktop", L"Меню «Пуск» и рабочий стол", L"Меню «Пуск» і робочий стіл"},
        {L"NInferno %s is installed.", L"NInferno %s установлена.", L"NInferno %s встановлено."},
        {L"The installation did not finish: %s", L"Установка не завершилась: %s",
         L"Встановлення не завершилося: %s"},
        {L"NInferno needs the Windows Subsystem for Linux (WSL 2), a part of Windows. Turn it on "
         L"now? Windows asks for administrator rights.",
         L"NInferno нужна подсистема Windows для Linux (WSL 2) — часть Windows. Включить её "
         L"сейчас? Windows попросит права администратора.",
         L"NInferno потрібна підсистема Windows для Linux (WSL 2) — частина Windows. Увімкнути її "
         L"зараз? Windows попросить права адміністратора."},
        {L"Windows needs a restart to finish turning WSL on. The installation continues by itself "
         L"after the restart. Restart now?",
         L"Чтобы включить WSL, Windows нужно перезагрузить. После перезагрузки установка "
         L"продолжится сама. Перезагрузить сейчас?",
         L"Щоб увімкнути WSL, Windows потрібно перезавантажити. Після перезавантаження "
         L"встановлення продовжиться саме. Перезавантажити зараз?"},
        {L"WSL was not turned on (administrator rights were not given).",
         L"WSL не включена (права администратора не выданы).",
         L"WSL не увімкнено (права адміністратора не надано)."},
        {L"Downloading %s: %s of %s", L"Загрузка %s: %s из %s", L"Завантаження %s: %s з %s"},
        {L"Checking the download", L"Проверка загрузки", L"Перевірка завантаження"},
        {L"Unpacking", L"Распаковка", L"Розпакування"},
        {L"Installing NInferno %s", L"Установка NInferno %s", L"Встановлення NInferno %s"},
        {L"Remove NInferno and the models it downloaded (%s)?",
         L"Удалить NInferno и скачанные ею модели (%s)?",
         L"Видалити NInferno і завантажені нею моделі (%s)?"},
        {L"NInferno is removed.", L"NInferno удалена.", L"NInferno видалено."},
        {L"Remove NInferno? The models it downloaded stay in %s (shared with other Hugging Face "
         L"tools); delete that folder yourself to free the space.",
         L"Удалить NInferno? Скачанные модели останутся в %s (общая папка с другими инструментами "
         L"Hugging Face); удалите её сами, если нужно место.",
         L"Видалити NInferno? Завантажені моделі залишаться в %s (спільна тека з іншими "
         L"інструментами Hugging Face); видаліть її самі, якщо потрібне місце."},
        {L"NInferno %s is available (you have %s). Update now?",
         L"Вышла NInferno %s (у вас %s). Обновить сейчас?",
         L"Вийшла NInferno %s (у вас %s). Оновити зараз?"},
        {L"This version of NInferno (%s) is out of date. Please update to %s: it takes a minute "
         L"and keeps your models.",
         L"Эта версия NInferno (%s) устарела. Пожалуйста, обновитесь до %s: это займёт минуту, "
         L"модели сохранятся.",
         L"Ця версія NInferno (%s) застаріла. Будь ласка, оновіться до %s: це займе хвилину, "
         L"моделі збережуться."},
        {L"NInferno %s is the newest version.", L"NInferno %s — последняя версия.",
         L"NInferno %s — остання версія."},
        {L"Updating NInferno to %s", L"Обновление NInferno до %s", L"Оновлення NInferno до %s"},
        {L"Open NInferno", L"Открыть NInferno", L"Відкрити NInferno"},
        {L"Check for updates", L"Проверить обновления", L"Перевірити оновлення"},
        {L"Quit NInferno (stops the models)", L"Выйти из NInferno (модели остановятся)",
         L"Вийти з NInferno (моделі зупиняться)"},
        {L"Starting NInferno…", L"Запуск NInferno…", L"Запуск NInferno…"},
        {L"NInferno did not start: %s", L"NInferno не запустилась: %s",
         L"NInferno не запустилася: %s"},
        {L"Folder: %s", L"Папка: %s", L"Тека: %s"},
        {L"%s free on drive %s. NInferno and one model take about 30 GB.",
         L"Свободно %s на диске %s. NInferno с одной моделью займёт около 30 ГБ.",
         L"Вільно %s на диску %s. NInferno з однією моделлю займе близько 30 ГБ."},
        {L"Getting the newest installer (%s)", L"Загрузка нового установщика (%s)",
         L"Завантаження нового інсталятора (%s)"},
        {L"Details are in %s", L"Подробности в %s", L"Подробиці в %s"},
    };
    return table[static_cast<int>(w)][column];
}

} // namespace setup
