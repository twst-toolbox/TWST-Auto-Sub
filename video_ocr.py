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
            # ✅ 正确的 OCR 图像转换逻辑 (4通道 BGRA)
            bgra_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2BGRA)
            height, width = bgra_img.shape[:2]

            bytes_data = bgra_img.tobytes()
            data_writer = streams.DataWriter()
            data_writer.write_bytes(bytes_data)
            ibuffer = data_writer.detach_buffer()

            software_bitmap = SoftwareBitmap.create_copy_from_buffer(
                ibuffer,
                BitmapPixelFormat.BGRA8,
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
        self.root.title("Video Subtitle Extractor V10.5 (纯净双轨+防乱轴版)")
        self.root.geometry("1280x900")

        self.rect_d =[320, 465, 630, 100]  # 对话(绿)
        self.rect_c =[430, 170, 450, 90]   # 选项(蓝)
        self.rect_b =[100, 100, 150, 150]  # 背景(红)

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
        tk.Radiobutton(f_top2, text="TWST (黑字+米色底)", variable=self.var_mode, value="BLACK").pack(side=tk.LEFT)
        tk.Radiobutton(f_top2, text="18TRIP (白字检测)", variable=self.var_mode, value="WHITE").pack(side=tk.LEFT, padx=10)

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

        tk.Label(f_sets, text="文字/边缘 阈值:").pack(anchor="w")
        self.s_bin = tk.Scale(f_sets, from_=50, to=255, orient=tk.HORIZONTAL, command=self.update_preview)
        self.s_bin.set(130)
        self.s_bin.pack(fill=tk.X)

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
        if rid not in self.sliders:
            self.sliders[rid] = []
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
            for s in slist:
                s.config(to=max(w, h))
        self.update_preview()

    def on_seek(self, val):
        self.update_preview()

    def update_preview(self, _=None):
        if not self.cap or self.is_processing: return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(self.s_time.get()))
        ret, frame = self.cap.read()
        if ret:
            x, y, w, h = self.rect_d
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            xc, yc, wc, hc = self.rect_c
            cv2.rectangle(frame, (xc, yc), (xc+wc, yc+hc), (255, 255, 0), 2)
            xb, yb, wb, hb = self.rect_b
            cv2.rectangle(frame, (xb, yb), (xb+wb, yb+hb), (0, 0, 255), 2)

            # 预览二值化
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
        self.log("⚠️ 手动终止...")

    def start_task(self):
        if not self.video_path: return
        self.is_processing = True
        self.btn_run.config(state=tk.DISABLED, text="处理中...")
        self.btn_stop.config(state=tk.NORMAL)
        self.log("\n🚀 === 开始提取任务 ===")
        threading.Thread(target=self.run_process, daemon=True).start()

    def run_process(self):
        try:
            # 锁定参数
            p_rect_d = list(self.rect_d)
            p_rect_c = list(self.rect_c)
            p_rect_b = list(self.rect_b)
            p_diff = self.s_diff.get() / 100.0
            p_bin = self.s_bin.get()
            do_ocr = self.var_ocr.get()
            is_twst_mode = (self.var_mode.get() == "BLACK")
            
            # 米白色定义 (TWST专用)
            LOWER_COLOR = np.array([0, 0, 130])
            UPPER_COLOR = np.array([180, 100, 255])

            cap = cv2.VideoCapture(self.video_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            subs =[]

            # --- 对话轨道 (绿框) ---
            d_speaking = False
            d_start = 0
            d_peak = 0.0
            d_best_frame = None
            d_max_den = 0.0
            last_dil = None

            # --- 选项轨道 (蓝框) ---
            c_active = False
            c_start = 0
            c_peak = 0.0
            c_best_frame = None

            kernel = np.ones((3, 3), np.uint8)
            idx = 0

            while self.is_processing:
                ret, frame = cap.read()
                if not ret: break

                if idx % 100 == 0:
                    prog = (idx / total) * 100
                    self.root.after(0, lambda v=prog: self.progress.config(value=v))

                hsv_full = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                
                # ========================================================
                # 🟢 轨道A：对话检测 (绿框) - 绝对独立
                # ========================================================
                x, y, w, h = p_rect_d
                density_d = 0.0
                diff_score = 0.0
                
                if w > 0 and h > 0:
                    roi_d = frame[y:y+h, x:x+w]
                    
                    # 如果是 TWST 模式，用米色底框过滤（防白屏闪光）
                    if is_twst_mode:
                        roi_d_hsv = hsv_full[y:y+h, x:x+w]
                        ratio_d = cv2.countNonZero(cv2.inRange(roi_d_hsv, LOWER_COLOR, UPPER_COLOR)) / (w * h)
                        if ratio_d > 0.4:
                            roi_gray = cv2.cvtColor(roi_d, cv2.COLOR_BGR2GRAY)
                            _, binary = cv2.threshold(roi_gray, 150, 255, cv2.THRESH_BINARY_INV)
                            dilated = cv2.dilate(binary, kernel, iterations=1)
                            density_d = cv2.countNonZero(dilated) / (w * h)
                    # 18TRIP 模式：直接提白字
                    else:
                        roi_gray = cv2.cvtColor(roi_d, cv2.COLOR_BGR2GRAY)
                        _, binary = cv2.threshold(roi_gray, p_bin, 255, cv2.THRESH_BINARY)
                        dilated = cv2.dilate(binary, kernel, iterations=1)
                        density_d = cv2.countNonZero(dilated) / (w * h)

                    # 计算形状突变
                    if 'dilated' in locals():
                        if last_dil is not None:
                            diff_score = cv2.countNonZero(cv2.absdiff(dilated, last_dil)) / (w * h)
                        last_dil = dilated.copy()
                    else:
                        last_dil = None

                # 对话状态机
                if not d_speaking:
                    if density_d > 0.005:
                        d_speaking = True
                        d_start = idx
                        d_peak = density_d
                        d_max_den = density_d
                        d_best_frame = roi_d.copy() if 'roi_d' in locals() else None
                else:
                    if density_d > d_peak: d_peak = density_d
                    if density_d > d_max_den + 0.001:
                        d_max_den = density_d
                        d_best_frame = roi_d.copy() if 'roi_d' in locals() else None

                    should_cut = False
                    if density_d < 0.002: should_cut = True
                    elif density_d < (d_peak * 0.4) and d_peak > 0.02: should_cut = True
                    elif diff_score > p_diff and (idx - d_start) / self.fps > 0.2: should_cut = True

                    if should_cut:
                        dur = (idx - d_start) / self.fps
                        if dur > 0.25:
                            st = datetime.timedelta(seconds=d_start / self.fps)
                            et = datetime.timedelta(seconds=idx / self.fps)
                            content = "Line [Dialog]" # 默认占位符
                            
                            if do_ocr and d_best_frame is not None:
                                try:
                                    text = self.processor.ocr_image(d_best_frame)
                                    if text and len(text.strip()) >= 2:
                                        content = text.strip()
                                except: pass
                            
                            # 注意：这里暂不设定 sub.index，等最后排序时统一分配
                            subs.append(srt.Subtitle(index=0, start=st, end=et, content=content))
                            self.log(f"✅ 对话抓取: {content[:15]}...")

                        if density_d > 0.005:
                            d_speaking = True
                            d_start = idx
                            d_peak = density_d
                            d_max_den = density_d
                            d_best_frame = roi_d.copy() if 'roi_d' in locals() else None
                        else:
                            d_speaking = False

                # ========================================================
                # 🔵 轨道B：选项检测 (蓝框) - 绝对独立
                # ========================================================
                xc, yc, wc, hc = p_rect_c
                xb, yb, wb, hb = p_rect_b
                is_choice = False
                c_score = 0.0
                
                if wc > 0 and hc > 0:
                    roi_c = frame[yc:yc+hc, xc:xc+wc]
                    
                    if is_twst_mode:
                        # TWST: 找米色底框 (抗干扰能力极强)
                        roi_c_hsv = hsv_full[yc:yc+hc, xc:xc+wc]
                        ratio_c = cv2.countNonZero(cv2.inRange(roi_c_hsv, LOWER_COLOR, UPPER_COLOR)) / (wc * hc)
                        
                        ratio_b = 0
                        if wb > 0 and hb > 0:
                            roi_b_hsv = hsv_full[yb:yb+hb, xb:xb+wb]
                            ratio_b = cv2.countNonZero(cv2.inRange(roi_b_hsv, LOWER_COLOR, UPPER_COLOR)) / (wb * hb)
                        
                        # 选项框很亮，且比背景红框明显亮
                        if (ratio_c > 0.5) and (ratio_c > ratio_b + 0.2):
                            is_choice = True
                            c_score = ratio_c
                    else:
                        # 18TRIP: 找白字
                        gray_c = cv2.cvtColor(roi_c, cv2.COLOR_BGR2GRAY)
                        _, bin_c_img = cv2.threshold(gray_c, p_bin, 255, cv2.THRESH_BINARY)
                        c_score = cv2.countNonZero(bin_c_img) / (wc * hc)
                        if c_score > 0.02: # 白字占比不需要太高
                            is_choice = True

                    # 选项状态机
                    if not c_active:
                        if is_choice:
                            c_active = True
                            c_start = idx
                            c_peak = c_score
                            c_best_frame = roi_c.copy()
                    else:
                        if is_choice:
                            if c_score > c_peak:
                                c_peak = c_score
                                c_best_frame = roi_c.copy()
                        else:
                            c_active = False
                            dur_c = (idx - c_start) / self.fps
                            if dur_c > 0.5: # 选项通常停留较久
                                st = datetime.timedelta(seconds=c_start / self.fps)
                                et = datetime.timedelta(seconds=idx / self.fps)
                                content = "Line [Choice]" # 默认占位符
                                
                                if do_ocr and c_best_frame is not None:
                                    try:
                                        # OCR 选项截图
                                        text_c = self.processor.ocr_image(c_best_frame)
                                        if text_c and len(text_c.strip()) >= 2:
                                            # ✅ 成功识别后加上 [Choice] 尾缀
                                            content = f"{text_c.strip()} [Choice]"
                                    except: pass

                                subs.append(srt.Subtitle(index=0, start=st, end=et, content=content))
                                self.log(f"🔹 选项抓取: {content[:20]}...")

                idx += 1

            cap.release()

            # ✅ 统一排序并编排序号，避免日志和字幕文件中数字错乱
            subs.sort(key=lambda x: x.start)
            for i, sub in enumerate(subs):
                # 如果没有开启 OCR，或者是占位符，在这里补上最终的序号
                if sub.content == "Line [Dialog]":
                    sub.content = f"Line {i+1}"
                elif sub.content == "Line [Choice]":
                    sub.content = f"Line {i+1} [Choice]"
                
                sub.index = i + 1

            # 导出带 BOM 的 UTF-8，完美适配 Aegisub
            srt_path = os.path.splitext(self.video_path)[0] + ("_OCR.srt" if do_ocr else ".srt")
            with open(srt_path, "w", encoding="utf-8-sig") as f:
                f.write(srt.compose(subs))

            self.root.after(0, lambda: messagebox.showinfo("完成", f"任务成功！文件已保存至:\n{srt_path}"))

        except Exception as e:
            self.log(f"❌ [致命错误] {str(e)}")
            print(traceback.format_exc())
        finally:
            self.is_processing = False
            self.root.after(0, lambda: self.btn_run.config(state=tk.NORMAL, text="▶️ 开始处理"))

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
