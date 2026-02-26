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
                    self.logger("⚠️ [系统] OCR 初始化失败：未安装日语语言包。")
            except Exception as e:
                self.logger(f"❌ [系统] OCR 初始化异常: {e}")

    async def _run_win_ocr(self, cv2_img):
        if not self.ocr_engine:
            return ""
        try:
            # BGR 转 BGRA (必须4通道才能用 BGRA8 格式)
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
            return ""

    def ocr_image(self, img):
        if not HAS_WIN_OCR:
            return ""
        try:
            return asyncio.run(self._run_win_ocr(img))
        except Exception:
            return ""

# ================= GUI 主程序 =================
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Subtitle Extractor V12 (选项静止侦测版)")
        self.root.geometry("1280x900")

        self.rect_d =[320, 465, 630, 100]  
        self.rect_c =[430, 170, 450, 90]   
        self.rect_b = [100, 100, 150, 150]  

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
        if not HAS_WIN_OCR:
            cb_ocr.config(state=tk.DISABLED, text="OCR不可用")

        self.btn_run = tk.Button(f_top2, text="▶️ 开始处理", command=self.start_task, bg="#ddffdd", font=("微软雅黑", 11, "bold"))
        self.btn_run.pack(side=tk.RIGHT)
        self.btn_stop = tk.Button(f_top2, text="🛑 停止", command=self.stop_task, bg="#ffdddd", font=("微软雅黑", 11), state=tk.DISABLED)
        self.btn_stop.pack(side=tk.RIGHT, padx=10)

        f_mid = tk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        f_mid.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.canvas_frame = tk.Frame(f_mid, bg="#222")
        f_mid.add(self.canvas_frame, stretch="always")
        self.canvas = tk.Canvas(self.canvas_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        f_log = tk.Frame(f_mid)
        f_mid.add(f_log, width=380)
        tk.Label(f_log, text="📜 运行日志").pack(anchor="w")
        self.txt_log = tk.Text(f_log, bg="#1e1e1e", fg="#00ff00", font=("Consolas", 9), state=tk.DISABLED)
        self.txt_log.pack(fill=tk.BOTH, expand=True)

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
            self.sliders[rid] =[]
        labels = ["X", "Y", "W", "H"]
        for i in range(4):
            tk.Label(f, text=labels[i]).pack(side=tk.LEFT, padx=2)
            s = tk.Scale(f, from_=0, to=2000, orient=tk.HORIZONTAL, command=lambda v, x=i, r=rid: self.on_rect(v, x, r))
            s.set(rect_var[i])
            s.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.sliders[rid].append(s)

    def on_rect(self, val, idx, rid):
        val = int(float(val))
        if rid == 0:
            self.rect_d[idx] = val
        elif rid == 1:
            self.rect_c[idx] = val
        elif rid == 2:
            self.rect_b[idx] = val
        self.update_preview()

    def load_video(self):
        path = filedialog.askopenfilename()
        if not path:
            return
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
        if not self.cap or self.is_processing:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(self.s_time.get()))
        ret, frame = self.cap.read()
        if ret:
            x, y, w, h = self.rect_d
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            xc, yc, wc, hc = self.rect_c
            cv2.rectangle(frame, (xc, yc), (xc+wc, yc+hc), (255, 255, 0), 2)
            xb, yb, wb, hb = self.rect_b
            cv2.rectangle(frame, (xb, yb), (xb+wb, yb+hb), (0, 0, 255), 2)

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
            if cw > 1:
                img.thumbnail((cw, ch))
            self.photo = ImageTk.PhotoImage(img)
            self.canvas.create_image(cw//2, ch//2, image=self.photo, anchor=tk.CENTER)

    def stop_task(self):
        self.is_processing = False
        self.log("⚠️ 收到停止指令，正在安全退出...")

    def start_task(self):
        if not self.video_path:
            return
        self.is_processing = True
        self.btn_run.config(state=tk.DISABLED, text="处理中...")
        self.btn_stop.config(state=tk.NORMAL)
        self.log("\n🚀 === V12 开始提取 ===")
        threading.Thread(target=self.run_process, daemon=True).start()

    def run_process(self):
        try:
            p_rect_d = list(self.rect_d)
            p_rect_c = list(self.rect_c)
            p_rect_b = list(self.rect_b)
            p_diff = self.s_diff.get() / 100.0
            p_bin = self.s_bin.get()
            do_ocr = self.var_ocr.get()
            is_twst_mode = (self.var_mode.get() == "BLACK")
            
            LOWER_COLOR = np.array([0, 0, 100])
            UPPER_COLOR = np.array([180, 100, 255])

            cap = cv2.VideoCapture(self.video_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            subs =[]

            # --- 对话轨道变量 ---
            d_speaking = False
            d_start = 0
            d_peak = 0.0
            d_best_frame = None
            d_max_den = 0.0
            last_dil_d = None

            # --- 选项轨道变量 (V12 绝对静止检测器) ---
            c_active = False
            c_start = 0
            c_empty_frames = 0
            
            # 快门机制相关
            c_locked = False        # 是否已经拍到了完美静止的相片
            c_best_frame = None     # 锁死的最完美截图
            c_fallback_frame = None # 保底截图（防止手速太快没来得及静止）
            c_max_den = 0.0         # 用于保底
            c_still_frames = 0      # 静止计数器
            last_dil_c = None

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
                # 🟢 轨道A：对话 (保留智能缝合逻辑)
                # ========================================================
                x, y, w, h = p_rect_d
                density_d = 0.0
                diff_score_d = 0.0
                dilated_d = None
                
                if w > 0 and h > 0:
                    roi_d = frame[y:y+h, x:x+w]
                    if is_twst_mode:
                        roi_d_hsv = hsv_full[y:y+h, x:x+w]
                        ratio_d = cv2.countNonZero(cv2.inRange(roi_d_hsv, LOWER_COLOR, UPPER_COLOR)) / (w * h)
                        if ratio_d > 0.4:
                            roi_gray = cv2.cvtColor(roi_d, cv2.COLOR_BGR2GRAY)
                            _, binary = cv2.threshold(roi_gray, 150, 255, cv2.THRESH_BINARY_INV)
                            dilated_d = cv2.dilate(binary, kernel, iterations=1)
                            density_d = cv2.countNonZero(dilated_d) / (w * h)
                    else:
                        roi_gray = cv2.cvtColor(roi_d, cv2.COLOR_BGR2GRAY)
                        _, binary = cv2.threshold(roi_gray, p_bin, 255, cv2.THRESH_BINARY)
                        dilated_d = cv2.dilate(binary, kernel, iterations=1)
                        density_d = cv2.countNonZero(dilated_d) / (w * h)

                    if dilated_d is not None:
                        if last_dil_d is not None:
                            diff_score_d = cv2.countNonZero(cv2.absdiff(dilated_d, last_dil_d)) / (w * h)
                        last_dil_d = dilated_d.copy()
                    else:
                        last_dil_d = None

                if not d_speaking:
                    if density_d > 0.005:
                        d_speaking = True
                        d_start = idx
                        d_peak = density_d
                        d_max_den = density_d
                        d_best_frame = roi_d.copy()
                else:
                    if density_d > d_peak: d_peak = density_d
                    if density_d > d_max_den + 0.001:
                        d_max_den = density_d
                        d_best_frame = roi_d.copy()

                    should_cut = False
                    if density_d < 0.003: should_cut = True
                    elif density_d < (d_peak * 0.4) and d_peak > 0.02: should_cut = True
                    elif diff_score_d > p_diff and (idx - d_start) / self.fps > 0.2: should_cut = True

                    if should_cut:
                        dur = (idx - d_start) / self.fps
                        if dur > 0.25:
                            st = datetime.timedelta(seconds=d_start / self.fps)
                            et = datetime.timedelta(seconds=idx / self.fps)
                            content = "Line [Dialog]"
                            
                            if do_ocr and d_best_frame is not None:
                                try:
                                    text = self.processor.ocr_image(d_best_frame)
                                    if text and len(text.strip()) >= 2: content = text.strip()
                                except: pass

                            # 智能缝合防重影
                            is_merged = False
                            if len(subs) > 0:
                                last_sub = subs[-1]
                                if content != "Line [Dialog]" and content == last_sub.content:
                                    last_sub.end = et
                                    is_merged = True
                                    self.log(f"🔄 缝合对话碎片: {content[:10]}...")

                            if not is_merged:
                                subs.append(srt.Subtitle(index=0, start=st, end=et, content=content))
                                self.log(f"✅ 对话: {content[:15]}...")

                        if density_d > 0.005:
                            d_speaking = True
                            d_start = idx
                            d_peak = density_d
                            d_max_den = density_d
                            d_best_frame = roi_d.copy()
                        else:
                            d_speaking = False

                # ========================================================
                # 🔵 轨道B：选项 (V12核心: 绝对静止侦测)
                # ========================================================
                xc, yc, wc, hc = p_rect_c
                xb, yb, wb, hb = p_rect_b
                is_choice = False
                density_c = 0.0
                diff_score_c = 0.0
                dilated_c = None
                
                if wc > 0 and hc > 0:
                    roi_c = frame[yc:yc+hc, xc:xc+wc]
                    
                    if is_twst_mode:
                        # 依旧使用米色底框定位
                        roi_c_hsv = hsv_full[yc:yc+hc, xc:xc+wc]
                        ratio_c = cv2.countNonZero(cv2.inRange(roi_c_hsv, LOWER_COLOR, UPPER_COLOR)) / (wc * hc)
                        
                        ratio_b = 0
                        if wb > 0 and hb > 0:
                            roi_b_hsv = hsv_full[yb:yb+hb, xb:xb+wb]
                            ratio_b = cv2.countNonZero(cv2.inRange(roi_b_hsv, LOWER_COLOR, UPPER_COLOR)) / (wb * hb)
                        
                        if (ratio_c > 0.4) and (ratio_c > ratio_b + 0.1):
                            gray_c = cv2.cvtColor(roi_c, cv2.COLOR_BGR2GRAY)
                            _, bin_c = cv2.threshold(gray_c, 150, 255, cv2.THRESH_BINARY_INV)
                            dilated_c = cv2.dilate(bin_c, kernel, iterations=1)
                            density_c = cv2.countNonZero(dilated_c) / (wc * hc)
                            if density_c > 0.005: 
                                is_choice = True
                    else:
                        gray_c = cv2.cvtColor(roi_c, cv2.COLOR_BGR2GRAY)
                        _, bin_c = cv2.threshold(gray_c, p_bin, 255, cv2.THRESH_BINARY)
                        dilated_c = cv2.dilate(bin_c, kernel, iterations=1)
                        density_c = cv2.countNonZero(dilated_c) / (wc * hc)
                        if density_c > 0.01:
                            is_choice = True

                    if dilated_c is not None:
                        if last_dil_c is not None:
                            diff_score_c = cv2.countNonZero(cv2.absdiff(dilated_c, last_dil_c)) / (wc * hc)
                        last_dil_c = dilated_c.copy()
                    else:
                        last_dil_c = None

                # 选项状态机
                if not c_active:
                    if is_choice:
                        # 选项刚弹出来
                        c_active = True
                        c_start = idx
                        c_empty_frames = 0
                        c_locked = False
                        c_best_frame = None
                        c_still_frames = 0
                        c_max_den = density_c
                        c_fallback_frame = roi_c.copy() # 第一帧保底
                else:
                    if is_choice:
                        c_empty_frames = 0 
                        
                        # 随时更新保底最高密度帧
                        if density_c > c_max_den:
                            c_max_den = density_c
                            if not c_locked:
                                c_fallback_frame = roi_c.copy()

                        # 📷 快门逻辑：寻找绝对静止的那一刻
                        if not c_locked:
                            # 判定条件：误差小于 0.1% 视为静止 (屏蔽mp4压缩的微小像素抖动)
                            if diff_score_c < 0.001:
                                c_still_frames += 1
                                # 连续 8 帧 (约 0.25 秒) 画面纹丝不动
                                if c_still_frames >= 8:
                                    c_best_frame = roi_c.copy() # 咔嚓！上锁！
                                    c_locked = True
                                    self.log(f"📸[选项快门] 发现完美静止画面，锁定！")
                            else:
                                # 只要动了一下（动画还没放完/玩家点了），重新倒数
                                c_still_frames = 0
                    else:
                        # 选项消失了
                        c_empty_frames += 1
                        if c_empty_frames > 15: # 容忍 0.5 秒的消失动画
                            c_active = False
                            real_end_idx = idx - 15
                            dur_c = (real_end_idx - c_start) / self.fps
                            
                            if dur_c > 0.5:
                                st_c = datetime.timedelta(seconds=c_start / self.fps)
                                et_c = datetime.timedelta(seconds=real_end_idx / self.fps)
                                content_c = "Line [Choice]"
                                
                                # 取图策略：如果成功锁定了静止帧就用静止帧，否则说明玩家手速太快，用保底最高密度帧
                                target_frame = c_best_frame if c_locked else c_fallback_frame
                                
                                if do_ocr and target_frame is not None:
                                    try:
                                        text_c = self.processor.ocr_image(target_frame)
                                        if text_c and len(text_c.strip()) >= 2:
                                            content_c = f"{text_c.strip()} [Choice]"
                                    except: pass

                                # 选项缝合
                                is_merged_c = False
                                if len(subs) > 0:
                                    last_sub = subs[-1]
                                    if content_c != "Line [Choice]" and content_c == last_sub.content:
                                        last_sub.end = et_c
                                        is_merged_c = True
                                        self.log(f"🔄 缝合选项碎片: {content_c[:10]}...")

                                if not is_merged_c:
                                    subs.append(srt.Subtitle(index=0, start=st_c, end=et_c, content=content_c))
                                    self.log(f"🔹 选项结算: {content_c[:15]}...")

                idx += 1

            cap.release()

            # --- 最后整理序号 ---
            subs.sort(key=lambda x: x.start)
            for i, sub in enumerate(subs):
                if sub.content == "Line [Dialog]": sub.content = f"Line {i+1}"
                elif sub.content == "Line [Choice]": sub.content = f"Line {i+1} [Choice]"
                sub.index = i + 1

            srt_path = os.path.splitext(self.video_path)[0] + ("_OCR.srt" if do_ocr else ".srt")
            with open(srt_path, "w", encoding="utf-8-sig") as f:
                f.write(srt.compose(subs))

            self.root.after(0, lambda: messagebox.showinfo("完成", f"任务成功！\n文件已保存至:\n{srt_path}"))

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
