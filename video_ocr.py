import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
import numpy as np
import srt
import datetime
import threading
import os
import asyncio
import traceback
from PIL import Image, ImageTk

# --- Windows OCR 模块导入 ---
HAS_WIN_OCR = False
try:
    from winsdk.windows.media.ocr import OcrEngine
    from winsdk.windows.globalization import Language
    from winsdk.windows.graphics.imaging import SoftwareBitmap, BitmapPixelFormat
    import winsdk.windows.storage.streams as streams
    HAS_WIN_OCR = True
except ImportError:
    print("未安装 winsdk 库或不在 Windows 环境")

# ================= 核心算法类 =================
class VideoProcessor:
    def __init__(self, logger):
        self.ocr_engine = None
        self.logger = logger
        if HAS_WIN_OCR:
            try:
                lang = Language("ja-JP")
                if OcrEngine.is_language_supported(lang):
                    self.ocr_engine = OcrEngine.try_create_from_language(lang)
                    self.logger("✅ [系统] Windows OCR (日语) 就绪。")
                else:
                    self.logger("⚠️ [系统] OCR 初始化失败：您的 Windows 可能未安装日语语言包。")
            except Exception as e:
                self.logger(f"❌ [系统] OCR 初始化异常: {e}")

    async def _run_win_ocr(self, cv2_img):
        if not self.ocr_engine: return ""
        try:
            # ✅ 修正1: BGR → BGRA (必须4通道才能用 Bgra8 格式)
            bgra_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2BGRA)
            height, width = bgra_img.shape[:2]

            bytes_data = bgra_img.tobytes()
            data_writer = streams.DataWriter()
            data_writer.write_bytes(bytes_data)
            ibuffer = data_writer.detach_buffer()

            # ✅ 修正2: RG_B8 根本不存在，正确枚举是 Bgra8
            software_bitmap = SoftwareBitmap.create_copy_from_buffer(
                ibuffer,
                BitmapPixelFormat.BGRA8,  # ← 核心修正
                width,
                height
            )

            result = await self.ocr_engine.recognize_async(software_bitmap)
            return result.text.replace(" ", "")
        except Exception as e:
            self.logger(f"⚠️ [OCR内部错误] {e}")
            return ""

    def ocr_image(self, img):
        if not HAS_WIN_OCR: return ""
        try:
            return asyncio.run(self._run_win_ocr(img))
        except Exception:
            return ""

# ================= GUI 主程序 =================
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Subtitle Extractor V10.2 (OCR修正+白屏过滤)")
        self.root.geometry("1280x900")

        self.rect_d = [320, 465, 630, 100]  # 对话(绿)
        self.rect_c = [430, 170, 450, 90]   # 选项(蓝)
        self.rect_b = [100, 100, 150, 150]  # 背景(红) — 用于白屏检测

        self.video_path = ""
        self.cap = None
        self.total_frames = 0
        self.fps = 30
        self.is_processing = False

        self._setup_ui()
        self.processor = VideoProcessor(self.log)

    def log(self, message):
        self.txt_log.config(state=tk.NORMAL)
        self.txt_log.insert(tk.END, message + "\n")
        self.txt_log.see(tk.END)
        self.txt_log.config(state=tk.DISABLED)

    def _setup_ui(self):
        # 顶部
        f_top1 = tk.Frame(self.root, pady=5)
        f_top1.pack(fill=tk.X, padx=10)
        tk.Button(f_top1, text="📂 加载视频", command=self.load_video, font=("微软雅黑", 10)).pack(side=tk.LEFT)
        self.lbl_info = tk.Label(f_top1, text="未加载...", fg="blue")
        self.lbl_info.pack(side=tk.LEFT, padx=10)

        f_top2 = tk.Frame(self.root, pady=5)
        f_top2.pack(fill=tk.X, padx=10)

        self.var_mode = tk.StringVar(value="BLACK")
        tk.Label(f_top2, text="模式:", font=("微软雅黑", 10, "bold")).pack(side=tk.LEFT)
        tk.Radiobutton(f_top2, text="TWST (黑字)", variable=self.var_mode, value="BLACK").pack(side=tk.LEFT)
        tk.Radiobutton(f_top2, text="18TRIP (白字)", variable=self.var_mode, value="WHITE").pack(side=tk.LEFT, padx=10)

        self.var_ocr = tk.BooleanVar(value=False)
        cb_ocr = tk.Checkbutton(f_top2, text="启用 OCR", variable=self.var_ocr, font=("微软雅黑", 10, "bold"), fg="purple")
        cb_ocr.pack(side=tk.LEFT, padx=20)
        if not HAS_WIN_OCR: cb_ocr.config(state=tk.DISABLED, text="OCR不可用(缺winsdk)")

        self.btn_run = tk.Button(f_top2, text="▶️ 开始处理", command=self.start_task, bg="#ddffdd", font=("微软雅黑", 11, "bold"))
        self.btn_run.pack(side=tk.RIGHT)
        self.btn_stop = tk.Button(f_top2, text="🛑 停止", command=self.stop_task, bg="#ffdddd", font=("微软雅黑", 11), state=tk.DISABLED)
        self.btn_stop.pack(side=tk.RIGHT, padx=10)

        # 中间
        f_mid = tk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        f_mid.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.canvas_frame = tk.Frame(f_mid, bg="#222")
        f_mid.add(self.canvas_frame, stretch="always")
        self.canvas = tk.Canvas(self.canvas_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        f_log = tk.Frame(f_mid)
        f_mid.add(f_log, width=350)
        tk.Label(f_log, text="📜 运行日志").pack(anchor="w")
        self.txt_log = tk.Text(f_log, bg="#1e1e1e", fg="#00ff00", font=("Consolas", 9), state=tk.DISABLED)
        self.txt_log.pack(fill=tk.BOTH, expand=True)

        # 底部参数
        f_ctrl = tk.Frame(self.root, height=150)
        f_ctrl.pack(fill=tk.X, padx=10, pady=5)

        nb = ttk.Notebook(f_ctrl)
        nb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.create_tab(nb, "对话框(绿)", self.rect_d, 0)
        self.create_tab(nb, "选项框(蓝)", self.rect_c, 1)
        self.create_tab(nb, "背景(红)", self.rect_b, 2)

        f_sets = tk.LabelFrame(f_ctrl, text="设置", padx=5)
        f_sets.pack(side=tk.RIGHT, fill=tk.Y, padx=5)

        tk.Label(f_sets, text="防连读灵敏度:").pack(anchor="w")
        self.s_diff = tk.Scale(f_sets, from_=0.1, to=10.0, resolution=0.1, orient=tk.HORIZONTAL)
        self.s_diff.set(3.0)
        self.s_diff.pack(fill=tk.X)

        tk.Label(f_sets, text="文字阈值:").pack(anchor="w")
        self.s_bin = tk.Scale(f_sets, from_=50, to=255, orient=tk.HORIZONTAL, command=self.update_preview)
        self.s_bin.set(130)
        self.s_bin.pack(fill=tk.X)

        # ✅ 新增：白屏亮度阈值滑条
        tk.Label(f_sets, text="白屏过滤阈值(红框):").pack(anchor="w")
        self.s_white = tk.Scale(f_sets, from_=150, to=255, orient=tk.HORIZONTAL)
        self.s_white.set(220)
        self.s_white.pack(fill=tk.X)

        f_bot = tk.Frame(self.root)
        f_bot.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        self.s_time = tk.Scale(f_bot, from_=0, to=100, orient=tk.HORIZONTAL, showvalue=0, command=self.on_seek)
        self.s_time.pack(fill=tk.X)
        self.progress = ttk.Progressbar(f_bot, mode='determinate')
        self.progress.pack(fill=tk.X, pady=5)

    def create_tab(self, nb, title, rect_var, rid):
        f = tk.Frame(nb)
        nb.add(f, text=title)
        self.sliders = getattr(self, "sliders", {})
        if rid not in self.sliders: self.sliders[rid] = []
        labels = ["X", "Y", "W", "H"]
        for i in range(4):
            tk.Label(f, text=labels[i]).pack(side=tk.LEFT, padx=2)
            s = tk.Scale(f, from_=0, to=2000, orient=tk.HORIZONTAL, command=lambda v, x=i, r=rid: self.on_rect(v, x, r))
            s.set(rect_var[i])
            s.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.sliders[rid].append(s)

    def on_rect(self, val, idx, rid):
        val = int(float(val))
        if rid == 0: self.rect_d[idx] = val
        elif rid == 1: self.rect_c[idx] = val
        elif rid == 2: self.rect_b[idx] = val
        self.update_preview()

    def load_video(self):
        path = filedialog.askopenfilename()
        if not path: return
        self.video_path = path
        self.cap = cv2.VideoCapture(path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.s_time.config(to=self.total_frames)
        self.lbl_info.config(text=f"{os.path.basename(path)} ({w}x{h})")
        for slist in self.sliders.values():
            for s in slist: s.config(to=max(w, h))
        self.update_preview()

    def on_seek(self, val):
        self.update_preview()

    def update_preview(self, _=None):
        if not self.cap or self.is_processing: return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(self.s_time.get()))
        ret, frame = self.cap.read()
        if ret:
            x, y, w, h = self.rect_d
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)      # 绿
            xc, yc, wc, hc = self.rect_c
            cv2.rectangle(frame, (xc, yc), (xc+wc, yc+hc), (255, 255, 0), 2)  # 蓝
            xb, yb, wb, hb = self.rect_b
            cv2.rectangle(frame, (xb, yb), (xb+wb, yb+hb), (0, 0, 255), 2)   # 红

            roi = frame[y:y+h, x:x+w]
            if roi.size > 0:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                mode = cv2.THRESH_BINARY_INV if self.var_mode.get() == "BLACK" else cv2.THRESH_BINARY
                _, bin_img = cv2.threshold(gray, self.s_bin.get(), 255, mode)
                bin_c = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)
                frame[y:y+h, x:x+w] = cv2.addWeighted(roi, 0.3, bin_c, 0.7, 0)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
            if cw > 1: img.thumbnail((cw, ch))
            self.photo = ImageTk.PhotoImage(img)
            self.canvas.create_image(cw//2, ch//2, image=self.photo, anchor=tk.CENTER)

    def stop_task(self):
        self.is_processing = False
        self.log("⚠️ 停止...")

    def start_task(self):
        if not self.video_path: return
        self.is_processing = True
        self.btn_run.config(state=tk.DISABLED, text="处理中...")
        self.btn_stop.config(state=tk.NORMAL)
        self.log("\n🚀 === 开始任务 ===")
        threading.Thread(target=self.run_process, daemon=True).start()

    def is_white_flash(self, frame, rect, threshold):
        """
        ✅ 红框白屏检测
        判断背景区域平均亮度是否超过阈值，超过说明是演出白屏，应忽略该帧。
        """
        xb, yb, wb, hb = rect
        if wb <= 0 or hb <= 0: return False
        roi_b = frame[yb:yb+hb, xb:xb+wb]
        if roi_b.size == 0: return False
        gray_b = cv2.cvtColor(roi_b, cv2.COLOR_BGR2GRAY)
        mean_brightness = cv2.mean(gray_b)[0]
        return mean_brightness >= threshold

    def _save_dialogue(self, subs, d_start, d_end_idx, d_best_frame, sub_index, do_ocr):
        """対話セグメントを確定してリストに追加する共通関数"""
        dur = (d_end_idx - d_start) / self.fps
        if dur < 0.3:
            return sub_index  # 短すぎるものは無視（0.25→0.3に微調整）
        st = datetime.timedelta(seconds=d_start / self.fps)
        et = datetime.timedelta(seconds=d_end_idx / self.fps)
        content = ""
        if do_ocr and d_best_frame is not None:
            try:
                text = self.processor.ocr_image(d_best_frame)
                # ✅ 2文字未満のOCR結果はノイズと判断して無視
                if text and len(text.strip()) >= 2:
                    content = text
            except: pass
        if not content:
            content = f"Line {sub_index}"
        # ✅ 直前エントリと全く同じテキストなら重複追加しない
        if subs and subs[-1].content == content and content.startswith("Line "):
            return sub_index  # 「Line N」の連続重複はスキップ
        subs.append(srt.Subtitle(index=sub_index, start=st, end=et, content=content))
        self.log(f"✅ [L{sub_index}] 対話: {content[:20]}...")
        return sub_index + 1

    def _save_choice(self, subs, c_start, c_end_idx, sub_index, do_ocr, p_rect_c, p_bin):
        """選択肢セグメントを確定してリストに追加する共通関数"""
        dur_c = (c_end_idx - c_start) / self.fps
        if dur_c < 0.5:
            return sub_index
        st = datetime.timedelta(seconds=c_start / self.fps)
        et = datetime.timedelta(seconds=c_end_idx / self.fps)
        content = ""
        if do_ocr:
            xc, yc, wc, hc = p_rect_c
            cap2 = cv2.VideoCapture(self.video_path)
            cap2.set(cv2.CAP_PROP_POS_FRAMES, c_start + 5)
            ret_c, frame_c = cap2.read()
            cap2.release()
            if ret_c:
                roi_ocr_c = frame_c[yc:yc+hc, xc:xc+wc]
                text_c = self.processor.ocr_image(roi_ocr_c)
                if text_c: content = text_c
        if not content:
            content = f"[Choice] Line {sub_index}"
        else:
            content = f"[選項] {content}"
        subs.append(srt.Subtitle(index=sub_index, start=st, end=et, content=content))
        self.log(f"🔹 [L{sub_index}] {content[:30]}")
        return sub_index + 1

    def run_process(self):
        try:
            p_rect_d = list(self.rect_d)
            p_rect_c = list(self.rect_c)
            p_rect_b = list(self.rect_b)
            p_diff = self.s_diff.get() / 100.0
            p_bin = self.s_bin.get()
            p_white = self.s_white.get()
            do_ocr = self.var_ocr.get()
            is_black_text = (self.var_mode.get() == "BLACK")
            mode = cv2.THRESH_BINARY_INV if is_black_text else cv2.THRESH_BINARY

            cap = cv2.VideoCapture(self.video_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            subs = []

            # --- 対話ステート ---
            d_speaking = False
            d_start = 0
            d_peak = 0.0
            d_best_frame = None
            d_max_den = 0.0
            last_dil = None

            # --- 選択肢ステート ---
            c_active = False
            c_start = 0

            sub_index = 1
            kernel = np.ones((3, 3), np.uint8)
            xc, yc, wc, hc = p_rect_c

            idx = 0
            while self.is_processing:
                ret, frame = cap.read()
                if not ret: break

                if idx % 100 == 0:
                    prog = (idx / total) * 100
                    self.root.after(0, lambda v=prog: self.progress.config(value=v))

                # ===== 白屏検出: 全フレーム処理をスキップ =====
                if self.is_white_flash(frame, p_rect_b, p_white):
                    if d_speaking:
                        sub_index = self._save_dialogue(subs, d_start, idx, d_best_frame, sub_index, do_ocr)
                        d_speaking = False
                        last_dil = None
                    if c_active:
                        sub_index = self._save_choice(subs, c_start, idx, sub_index, do_ocr, p_rect_c, p_bin)
                        c_active = False
                    idx += 1
                    continue

                # ===== Step1: 先に選択肢を判定 =====
                is_choice_frame = False
                if wc > 0 and hc > 0:
                    roi_c = frame[yc:yc+hc, xc:xc+wc]
                    gray_c = cv2.cvtColor(roi_c, cv2.COLOR_BGR2GRAY)
                    _, bin_c_img = cv2.threshold(gray_c, p_bin, 255, mode)
                    den_c = cv2.countNonZero(bin_c_img) / (wc * hc)
                    is_choice_frame = (den_c > 0.1)

                # ===== Step2: 相互排他 — 選択肢フレームなら対話を強制終了 =====
                if is_choice_frame:
                    # 対話が進行中なら締める
                    if d_speaking:
                        sub_index = self._save_dialogue(subs, d_start, idx, d_best_frame, sub_index, do_ocr)
                        d_speaking = False
                        last_dil = None

                    # 選択肢ステートマシン
                    if not c_active:
                        c_active = True
                        c_start = idx
                    # 選択肢継続中は何もしない（終了は次のelseで処理）

                else:
                    # ===== 選択肢フレームでない → 選択肢が終わったか確認 =====
                    if c_active:
                        sub_index = self._save_choice(subs, c_start, idx, sub_index, do_ocr, p_rect_c, p_bin)
                        c_active = False
                        # ✅ 選択肢終了後は差分計算をリセット（選択肢前のフレームと比べてしまうのを防ぐ）
                        last_dil = None

                    # ===== Step3: 対話検出（選択肢フレームでないときだけ） =====
                    x, y, w, h = p_rect_d
                    if w > 0 and h > 0:
                        roi = frame[y:y+h, x:x+w]
                        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        _, binary = cv2.threshold(roi_gray, p_bin, 255, mode)
                        dilated = cv2.dilate(binary, kernel, iterations=1)
                        density = cv2.countNonZero(dilated) / (w * h)

                        diff_score = 0.0
                        if last_dil is not None:
                            diff_score = cv2.countNonZero(cv2.absdiff(dilated, last_dil)) / (w * h)
                        last_dil = dilated.copy()

                        if not d_speaking:
                            if density > 0.005:
                                d_speaking = True
                                d_start = idx
                                d_peak = density
                                d_max_den = density
                                d_best_frame = roi.copy()
                        else:
                            if density > d_peak: d_peak = density
                            if density > d_max_den + 0.001:
                                d_max_den = density
                                d_best_frame = roi.copy()

                            should_cut = False
                            if density < 0.002: should_cut = True
                            elif density < (d_peak * 0.4) and d_peak > 0.02: should_cut = True
                            elif diff_score > p_diff and (idx - d_start) / self.fps > 0.2: should_cut = True

                            if should_cut:
                                sub_index = self._save_dialogue(subs, d_start, idx, d_best_frame, sub_index, do_ocr)
                                if density > 0.005:
                                    d_speaking = True
                                    d_start = idx
                                    d_peak = density
                                    d_max_den = density
                                    d_best_frame = roi.copy()
                                else:
                                    d_speaking = False

                idx += 1

            # ===== ループ終了後、未確定セグメントを締める =====
            if d_speaking:
                self._save_dialogue(subs, d_start, idx, d_best_frame, sub_index, do_ocr)
            if c_active:
                self._save_choice(subs, c_start, idx, sub_index, do_ocr, p_rect_c, p_bin)

            cap.release()

            # 時系列ソート → 連番振り直し
            subs.sort(key=lambda x: x.start)
            for i, sub in enumerate(subs): sub.index = i + 1

            srt_path = os.path.splitext(self.video_path)[0] + ("_OCR.srt" if do_ocr else ".srt")
            with open(srt_path, "w", encoding="utf-8-sig") as f:
                f.write(srt.compose(subs))

            self.root.after(0, lambda: messagebox.showinfo("完成", f"文件已保存:\n{srt_path}"))

        except Exception as e:
            self.log(f"❌ 错误: {e}")
            print(traceback.format_exc())
        finally:
            self.is_processing = False
            self.root.after(0, lambda: self.btn_run.config(state=tk.NORMAL, text="▶️ 开始处理"))

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
