param(
    [string]$RepoRoot = "C:\Users\kimeu\Downloads\starlink_uemr-main\starlink_uemr-main",
    [string]$Background = "paper2/paper2-1/background_0_5_w001_1h.npz",
    [string]$Config = "paper2/paper2-1/nearfield_starlink_multiwindow_config.yaml",
    [string]$OutputRoot = "paper2/paper2-1/polarization_contrast_sweep_w001",
    [string[]]$Pols = @("ee", "nn"),
    [double[]]$Contrasts = @(0.0, 0.2, 0.35, 0.5),
    [int]$MaxSatsPerWindow = 50,
    [int]$MaxScanSatellites = 5000,
    [int]$PhaseNullTrials = 128
)

$ErrorActionPreference = "Stop"

$wslRepo = "/mnt/c/Users/kimeu/Downloads/starlink_uemr-main/starlink_uemr-main"
$python = "/home/kimeuhanyoon18/miniconda3/envs/starlink_env/bin/python"
$script = "paper2/paper2-1/run_nearfield_starlink_multiwindow_v4.py"

foreach ($pol in $Pols) {
    $out = "$OutputRoot/unpolarized_$pol"
    $cmd = "cd $wslRepo && $python $script --config $Config --background $Background --output-dir $out --start-window 0 --max-windows 1 --max-sats-per-window $MaxSatsPerWindow --max-scan-satellites $MaxScanSatellites --peak-alt-min-deg 35 --peak-alt-max-deg 70 --cap-selection-method predicted_peak_apparent_flux --pol $pol --polarization-mode jones_unpolarized --phase-null-trials $PhaseNullTrials"
    wsl bash -lc $cmd

    foreach ($contrast in $Contrasts) {
        $tag = ("{0:0.00}" -f $contrast).Replace(".", "p")
        $out = "$OutputRoot/anti_correlated_${pol}_c$tag"
        $cmd = "cd $wslRepo && $python $script --config $Config --background $Background --output-dir $out --start-window 0 --max-windows 1 --max-sats-per-window $MaxSatsPerWindow --max-scan-satellites $MaxScanSatellites --peak-alt-min-deg 35 --peak-alt-max-deg 70 --cap-selection-method predicted_peak_apparent_flux --pol $pol --polarization-mode jones_anti_correlated --polarization-contrast $contrast --phase-null-trials $PhaseNullTrials"
        wsl bash -lc $cmd
    }
}

$rows = Get-ChildItem -Path (Join-Path $RepoRoot $OutputRoot) -Recurse -Filter multi_window_summary.csv |
    ForEach-Object {
        $case = Split-Path (Split-Path $_.FullName -Parent) -Parent | Split-Path -Leaf
        Import-Csv $_.FullName | ForEach-Object {
            $_ | Add-Member -NotePropertyName case -NotePropertyValue $case -PassThru
        }
    }

$summaryPath = Join-Path $RepoRoot "$OutputRoot\sweep_summary.csv"
New-Item -ItemType Directory -Force -Path (Split-Path $summaryPath -Parent) | Out-Null
$rows | Export-Csv -NoTypeInformation -Path $summaryPath
Write-Host "Wrote $summaryPath"
