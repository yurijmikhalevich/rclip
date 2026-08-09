# -*- mode: python ; coding: utf-8 -*-

import json
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import copy_metadata


block_cipher = None
project_root = Path(__file__).resolve().parents[2]
legal_dir = project_root / 'build/legal'
compliance_report_path = legal_dir / 'compliance-report.json'

if not compliance_report_path.is_file():
    raise RuntimeError('Missing build/legal/compliance-report.json; run `make build-windows` first.')

with compliance_report_path.open(encoding='utf-8') as stream:
    compliance_report = json.load(stream)

distribution_metadata = []
for component in compliance_report['components']:
    if component.get('type', 'python') == 'python':
        distribution_metadata += copy_metadata(component['name'])


a = Analysis(
    ['..\\..\\rclip\\main.py'],
    pathex=[],
    binaries=[],
    datas=[
        *collect_data_files('onnxruntime'),
        *distribution_metadata,
        (str(legal_dir), 'legal'),
    ],
    # rclip imports onnxruntime dynamically, so PyInstaller won't see it unless we
    # declare it explicitly.
    hiddenimports=['onnxruntime'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['pytest', 'pycodestyle', 'poet'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='rclip',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    contents_directory='.',
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='rclip',
)
