# 使用本机 Word 将正式报告 DOCX 导出为同内容 PDF。

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$inputPath = Join-Path $projectRoot 'reports/final/GMSC_建模总结报告.docx'
$outputPath = Join-Path $projectRoot 'reports/final/GMSC_建模总结报告.pdf'
if (-not (Test-Path -LiteralPath $inputPath)) {
    throw "缺少已生成的正式报告：$inputPath"
}
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $document = $word.Documents.Open($inputPath, $false, $true)
    try {
        $document.ExportAsFixedFormat($outputPath, 17)
    }
    finally {
        $document.Close($false)
    }
}
finally {
    $word.Quit()
}
Write-Output "已导出：$outputPath"
