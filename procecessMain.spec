# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['procecessMain.py'],
             pathex=['net.py', 'profunc.py', 'Ui_mainwindow.py', 'F:\\pyqt5UI\\processLicenceForbs'],
             binaries=[],
             datas=[],
             hiddenimports=['net', 'profunc', 'Ui_mainwindow', 'sklearn.utils._cython_blas'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='procecessMain',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=False , icon='favicon.ico')
