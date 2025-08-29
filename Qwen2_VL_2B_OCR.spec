# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['ocr_gui.py'],
    pathex=[],
    binaries=[],
    datas=[('C:\\Users\\pcr\\miniconda3\\envs\\qwen_ocr\\Lib\\site-packages\\torch\\lib', 'torch\\lib'), ('C:\\Users\\pcr\\miniconda3\\envs\\qwen_ocr\\Library\\bin', 'cudalibs'), ('models', 'models')],
    hiddenimports=['torch', 'transformers', 'PySide6', 'fitz'],
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
    [],
    exclude_binaries=True,
    name='Qwen2_VL_2B_OCR',
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
    icon=['ocr.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Qwen2_VL_2B_OCR',
)
