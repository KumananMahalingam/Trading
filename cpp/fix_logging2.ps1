$files = @(
    "src\main.cpp",
    "src\data\yahoo_client.cpp",
    "src\models\dataset.cpp",
    "src\models\trainer.cpp",
    "src\storage\excel_handler.cpp"
)

foreach ($file in $files) {
    $content = Get-Content $file -Raw
    # Remove all existing logging.hpp includes
    $content = $content -replace '#include "trading/util/logging\.hpp"\r?\n', ''
    # Add exactly one at the very top
    $content = '#include "trading/util/logging.hpp"' + "`n" + $content
    Set-Content $file $content -NoNewline
}
