param([switch]$Apply)
$ErrorActionPreference = 'Stop'
$taskRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$taskPrefix = $taskRoot.TrimEnd('\') + '\'
$taskTargets = [Collections.Generic.List[string]]::new()

# Exact old artifact directories; never follow caller-provided paths.
foreach ($taskName in @('artifacts', 'artifacts_deep', 'artifacts_optimized',
    'artifacts_regional', 'artifacts_regional_simplex', 'artifacts_roster',
    'artifacts_v2', '.deps', 'artifacts_roster_2026\benchmark')) {
    $taskPath = Join-Path $taskRoot $taskName
    if (Test-Path -LiteralPath $taskPath) { $taskTargets.Add($taskPath) }
}
foreach ($taskName in @('seed42', 'seed43', 'seed44', 'snapshot.json')) {
    $taskPath = Join-Path $taskRoot ('artifacts_roster_2026_swiss\' + $taskName)
    if (Test-Path -LiteralPath $taskPath) { $taskTargets.Add($taskPath) }
}
foreach ($taskDir in @(Get-ChildItem -LiteralPath $taskRoot -Directory -Recurse -Force |
    Where-Object { $_.Name -eq '__pycache__' -and $_.FullName -notlike '*\.git\*' })) {
    $taskTargets.Add($taskDir.FullName)
}
$taskTargets = @($taskTargets | Sort-Object -Unique)
$taskTopTargets = @($taskTargets | Where-Object {
    $taskCandidate = $_
    -not ($taskTargets | Where-Object {
        $_ -ne $taskCandidate -and $taskCandidate.StartsWith($_.TrimEnd('\')+'\', [StringComparison]::OrdinalIgnoreCase)
    })
})

# Validate every resolved target before any deletion. Refuse junctions/symlinks.
$taskPlan = foreach ($taskTarget in $taskTopTargets) {
    $taskResolved = (Resolve-Path -LiteralPath $taskTarget).Path
    if (-not $taskResolved.StartsWith($taskPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Target leaves workspace: $taskResolved"
    }
    if ($taskResolved -like '*artifacts_roster_2026_swiss_weighted*') {
        throw "The selected final model is protected: $taskResolved"
    }
    $taskItem = Get-Item -LiteralPath $taskResolved -Force
    $taskChildren = if ($taskItem.PSIsContainer) { @(Get-ChildItem -LiteralPath $taskResolved -Force -Recurse) } else { @() }
    if (@(@($taskItem) + $taskChildren | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }).Count) {
        throw "Refusing recursive cleanup through a reparse point: $taskResolved"
    }
    $taskFiles = @(@($taskItem) + $taskChildren | Where-Object { -not $_.PSIsContainer })
    [PSCustomObject]@{ path=$taskResolved.Substring($taskPrefix.Length).Replace('\','/');
        files=$taskFiles.Count; bytes=[long](($taskFiles | Measure-Object Length -Sum).Sum) }
}
$taskPlan = @($taskPlan)
$taskTotal = [long](($taskPlan | Measure-Object bytes -Sum).Sum)
if (-not $Apply) {
    $taskPlan | Format-Table -AutoSize
    Write-Output ('Dry run: {0} targets, {1:N2} MiB. Use -Apply to delete.' -f $taskPlan.Count, ($taskTotal/1MB))
    return
}
if (-not $taskPlan.Count) { Write-Output 'Nothing to remove.'; return }
$taskManifest = Join-Path $taskRoot 'docs\cleanup_manifest.json'
if (Test-Path -LiteralPath $taskManifest) { throw 'Cleanup manifest exists; preserve it before applying a new cleanup.' }
$taskRecord = [ordered]@{ date='2026-10-29'; status='planned'; removed_files=($taskPlan | Measure-Object files -Sum).Sum;
    removed_bytes=$taskTotal; targets=$taskPlan;
    retained=@('artifacts_roster_2026_swiss_weighted: selected model',
        'artifacts_roster_2026: parent checkpoints and reproducibility features',
        'artifacts_roster_2026_swiss: lightweight historical comparison without weights',
        'data: original user data retained locally; raw tables ignored by Git') }
$taskRecord | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $taskManifest -Encoding utf8
foreach ($taskTarget in $taskTopTargets) { Remove-Item -LiteralPath $taskTarget -Recurse -Force }
$taskRecord.status = 'completed'
$taskRecord | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $taskManifest -Encoding utf8
Write-Output ('Removed {0} files, {1:N2} MiB. Final weights and pretraining parents preserved.' -f $taskRecord.removed_files, ($taskTotal/1MB))
