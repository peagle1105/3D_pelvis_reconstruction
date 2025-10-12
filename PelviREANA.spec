# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['PelviREANA.py'],
    pathex=[],
    binaries=[],
    datas=[('config', 'config'), ('static', 'static'), ('train_data', 'train_data'), ('temp_folder', 'temp_folder'), ('tools', 'tools')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='PelviREANA',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
