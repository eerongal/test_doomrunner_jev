param(
    [int]$Port = 8097,
    [int]$Threads = 8,
    [string]$LlamaRoot = "C:\development\aistuff\llama.cpp",
    [string]$ModelName = "Qwen3.5-2B-UD-Q4_K_XL.gguf"
)

$server = Join-Path $LlamaRoot "build-cuda\bin\Release\llama-server.exe"
$model = Join-Path $LlamaRoot "models\$ModelName"

if (-not (Test-Path -LiteralPath $server)) {
    throw "llama-server.exe was not found: $server"
}
if (-not (Test-Path -LiteralPath $model)) {
    throw "Planner GGUF was not found: $model"
}

Write-Host "Starting CPU-only Level 2 planner on http://127.0.0.1:$Port using $Threads threads"
& $server --model $model --alias Qwen3.5-2B --port $Port --ctx-size 2048 --n-gpu-layers 0 --threads $Threads
