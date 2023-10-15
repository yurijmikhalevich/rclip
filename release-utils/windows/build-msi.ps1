param (
    [string]$Version
)
if (-not $Version) {
    throw "Version is not specified"
}

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$currentDir = Get-Location

$templatePath = Join-Path -Path $currentDir -ChildPath "release-utils\windows\template.aip"

$advinst = New-Object -ComObject AdvancedInstaller
$project = $advinst.LoadProject($templatePath)
$project.ProductDetails.Version = $Version
$project.ProductDetails.UpgradeCode.UpgradeCode = "{7C6C2996-8E43-4D30-8D67-1A347DCFEEBF}"

$project.InstallParameters.PackageType = "32bit"

$buildPath = Join-Path -Path $currentDir -ChildPath "dist\rclip"
$project.FilesComponent.AddFolderContentS("appdir", $buildPath)

$pathEnvVar = $project.Environment.NewVariable("PATH", "[APPDIR]")
$pathEnvVar.InstallOperationType = "CreateOrUpdate"
$pathEnvVar.RemoveOnUninstall = $true
$pathEnvVar.IsSystemVariable = $false
$pathEnvVar.UpdateOperationType = "Append"
$pathEnvVar.Separator = ";"

$msiBuildPath = Join-Path -Path $currentDir -ChildPath "build-msi"
if (-not (Test-Path -Path $msiBuildPath -PathType Container)) {
    New-Item -Path $msiBuildPath -ItemType Directory
}

$projectFile = Join-Path -Path $msiBuildPath -ChildPath "rclip.aip"
$project.SaveAs($projectFile)
$project.Build()
