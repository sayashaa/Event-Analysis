# japanese_font_setup.py
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def ensure_japanese_font():
    """
    Ensure Japanese font (Hiragino Sans W6 or Noto Sans CJK JP) is available for matplotlib.
    """
    font_paths = [
        "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",   # macOS bold
        "/System/Library/Fonts/ヒラギノ角ゴシック W5.ttc",
        "/Library/Fonts/NotoSansCJKjp-Bold.otf",
        "/Library/Fonts/NotoSansCJKjp-Regular.otf"
    ]
    font_path = next((p for p in font_paths if os.path.exists(p)), None)

    if font_path:
        fm.fontManager.addfont(font_path)
        prop = fm.FontProperties(fname=font_path)
        plt.rcParams["font.family"] = prop.get_name()
        plt.rcParams["axes.unicode_minus"] = False
        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"
        print(f"✅ 日本語フォント設定完了: {prop.get_name()}")
    else:
        print("⚠️ 日本語フォントが見つかりません。英数字フォントを使用します。")
