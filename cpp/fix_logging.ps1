$files = @(
    "src\main.cpp",
    "src\data\yahoo_client.cpp",
    "src\models\dataset.cpp",
    "src\models\trainer.cpp",
    "src\storage\excel_handler.cpp"
)

foreach ($file in $files) {
    $content = Get-Content $file -Raw
    if ($content -notmatch 'trading/util/logging\.hpp') {
        $newInclude = '#include "trading/util/logging.hpp"' + "`n"
        $content = $content -replace '(#include "trading/)', ($newInclude + '$1')
    }
    Set-Content $file $content -NoNewline
}
