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
import sys
from PIL import Image, ImageTk

# 尝试导入 Windows OCR
try:
    from winsdk.windows.media.ocr import OcrEngine
    from winsdk.windows.globalization import Language
    from winsdk.windows.graphics.imaging import SoftwareBitmap
    import winsdk.windows.storage.streams as streams
    HAS_WIN_OCR = True
except ImportError:
    HAS_WIN_OCR = False

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
                    self.logger("✅ [系统] Windows OCR (日语) 初始化成功。")
                else:
                    self.logger("⚠️ [系统] Windows 不支持日语 OCR，请在系统设置中添加日语语言包。")
            except Exception as e:
                self.logger(f"❌ [系统] OCR 初始化异常: {e}")

    async def _run_win_ocr(self, cv2_img):
        if not self.ocr_engine: return ""
        try:
            rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
            height, width, _ = rgb_img.shape
            bytes_data = rgb_img.tobytes()
            data_writer = streams.DataWriter()
            data_writer.write_bytes(bytes_data)
            ibuffer = data_writer.detach_buffer()
            software_bitmap = SoftwareBitmap.create_copy_from_buffer(
                ibuffer, winsdk.windows.graphics.imaging.BitmapPixelFormat.RG_B8, width, height
            )
            result = await self.ocr_engine.recognize_async(software_bitmap)
            return result.text.replace(" ", "")
        except Exception as e:
            self.logger(f"❌ [OCR 错误] {e}")
            return ""

    def ocr_image(self, img):
        if not HAS_WIN_OCR: return ""
        return asyncio.run(self._run_win_ocr(img))

# ================= GUI 主程序 =================
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Subtitle Extractor V10 (防崩溃可视化版)")
        self.root.geometry("1280x850")
        
        self.rect_d = [320, 465, 630, 100] 
        self.rect_c = [430, 170, 450, 90]  
        self.rect_b = [100, 100, 150, 150] 
        
        self.video_path = ""
        self.cap = None
        self.total_frames = 0
        self.fps = 30
        self.is_processing = False
        
        self._setup_ui()
        self.processor = VideoProcessor(self.log)
        
    def log(self, message):
        """实时输出日志到界面"""
        self.txt_log.config(state=tk.NORMAL)
        self.txt_log.insert(tk.END, message + "\n")
        self.txt_log.see(tk.END)
        self.txt_log.config(state=tk.DISABLED)
        
    def _setup_ui(self):
        # --- 顶部第一行：文件选择 ---
        f_top1 = tk.Frame(self.root, pady=5)
        f_top1.pack(fill=tk.X, padx=10)
        tk.Button(f_top1, text="📂 加载视频", command=self.load_video, font=("微软雅黑", 10)).pack(side=tk.LEFT)
        self.lbl_info = tk.Label(f_top1, text="未加载任何视频...", fg="blue")
        self.lbl_info.pack(side=tk.LEFT, padx=10)
        
        # --- 顶部第二行：功能开关与操作 ---
        f_top2 = tk.Frame(self.root, pady=5)
        f_top2.pack(fill=tk.X, padx=10)
        
        self.var_mode = tk.StringVar(value="BLACK") # 默认TWST黑字
        tk.Label(f_top2, text="文字颜色模式:", font=("微软雅黑", 10, "bold")).pack(side=tk.LEFT)
        tk.Radiobutton(f_top2, text="TWST (提取黑字)", variable=self.var_mode, value="BLACK").pack(side=tk.LEFT)
        tk.Radiobutton(f_top2, text="18TRIP (提取白字)", variable=self.var_mode, value="WHITE").pack(side=tk.LEFT, padx=10)
        
        self.var_ocr = tk.BooleanVar(value=False)
        cb_ocr = tk.Checkbutton(f_top2, text="启用 OCR 识别文本", variable=self.var_ocr, font=("微软雅黑", 10, "bold"), fg="purple")
        cb_ocr.pack(side=tk.LEFT, padx=20)
        if not HAS_WIN_OCR: cb_ocr.config(state=tk.DISABLED, text="OCR 库缺失")
        
        self.btn_run = tk.Button(f_top2, text="▶️ 开始处理", command=self.start_task, bg="#ddffdd", font=("微软雅黑", 11, "bold"))
        self.btn_run.pack(side=tk.RIGHT)
        self.btn_stop = tk.Button(f_top2, text="🛑 强行停止", command=self.stop_task, bg="#ffdddd", font=("微软雅黑", 11), state=tk.DISABLED)
        self.btn_stop.pack(side=tk.RIGHT, padx=10)

        # --- 中间部分：左预览，右日志 ---
        f_mid = tk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        f_mid.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 左侧：视频画面
        self.canvas_frame = tk.Frame(f_mid, bg="#222")
        f_mid.add(self.canvas_frame, stretch="always")
        self.canvas = tk.Canvas(self.canvas_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 右侧：日志输出
        f_log = tk.Frame(f_mid)
        f_mid.add(f_log, width=300)
        tk.Label(f_log, text="📜 运行日志").pack(anchor="w")
        self.txt_log = tk.Text(f_log, bg="#1e1e1e", fg="#00ff00", font=("Consolas", 9), state=tk.DISABLED)
        self.txt_log.pack(fill=tk.BOTH, expand=True)

        # --- 下半部分：参数控制 ---
        f_ctrl = tk.Frame(self.root, height=150)
        f_ctrl.pack(fill=tk.X, padx=10, pady=5)
        
        nb = ttk.Notebook(f_ctrl)
        nb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.create_tab(nb, "对话框(绿)", self.rect_d, 0)
        self.create_tab(nb, "选项框(蓝)", self.rect_c, 1)
        self.create_tab(nb, "背景(红)", self.rect_b, 2)
        
        f_sets = tk.LabelFrame(f_ctrl, text="敏感度调校", padx=5)
        f_sets.pack(side=tk.RIGHT, fill=tk.Y, padx=5)
        
        tk.Label(f_sets, text="切分灵敏度 (防连读):").pack(anchor="w")
        self.s_diff = tk.Scale(f_sets, from_=0.1, to=10.0, resolution=0.1, orient=tk.HORIZONTAL)
        self.s_diff.set(3.0)
        self.s_diff.pack(fill=tk.X)
        
        tk.Label(f_sets, text="文字亮度/灰度 阈值:").pack(anchor="w")
        self.s_bin = tk.Scale(f_sets, from_=50, to=255, orient=tk.HORIZONTAL, command=self.update_preview)
        self.s_bin.set(130)
        self.s_bin.pack(fill=tk.X)

        # --- 最底部：进度条固定 ---
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
        labels = ["X", "Y", "Width", "Height"]
        for i in range(4):
            tk.Label(f, text=labels[i]).pack(side=tk.LEFT, padx=5)
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
        self.lbl_info.config(text=f"📄 {os.path.basename(path)} ({w}x{h}, {self.total_frames}帧)")
        
        for slist in self.sliders.values():
            for s in slist: s.config(to=max(w, h))
        self.update_preview()
        self.log(f"🎬 视频加载成功，时长约 {self.total_frames/self.fps/60:.1f} 分钟。")

    def on_seek(self, val):
        self.update_preview()

    def update_preview(self, _=None):
        if not self.cap or self.is_processing: return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(self.s_time.get()))
        ret, frame = self.cap.read()
        if ret:
            x,y,w,h = self.rect_d
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            x,y,w,h = self.rect_c
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,255,0), 2)
            x,y,w,h = self.rect_b
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)
            
            # 预览二值化
            roi = frame[y:y+h, x:x+w]
            if roi.size > 0:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                # 根据模式选择二值化方式
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
        self.log("⚠️ 收到停止指令，正在安全退出...")

    def start_task(self):
        if not self.video_path: return
        self.is_processing = True
        self.btn_run.config(state=tk.DISABLED, text="正在处理...")
        self.btn_stop.config(state=tk.NORMAL)
        self.log("\n🚀 === 开始新的提取任务 ===")
        self.log(f"模式: {self.var_mode.get()}字提取 | OCR: {self.var_ocr.get()}")
        threading.Thread(target=self.run_process, daemon=True).start()

    def run_process(self):
        try:
            p_rect_d = list(self.rect_d)
            p_rect_c = list(self.rect_c)
            p_rect_b = list(self.rect_b)
            p_diff = self.s_diff.get() / 100.0
            p_bin = self.s_bin.get()
            do_ocr = self.var_ocr.get()
            is_black_text = (self.var_mode.get() == "BLACK")
            
            cap = cv2.VideoCapture(self.video_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            subs = []
            
            d_speaking = False
            d_start = 0
            d_peak = 0.0
            d_best_frame = None
            d_max_den = 0.0
            sub_index = 1
            last_dil = None
            kernel = np.ones((3,3), np.uint8)
            
            idx = 0
            while self.is_processing:
                ret, frame = cap.read()
                if not ret: break
                
                # UI 更新频率降低，防止卡顿
                if idx % 100 == 0:
                    prog = (idx / total) * 100
                    self.root.after(0, lambda v=prog: self.progress.config(value=v))
                    
                x,y,w,h = p_rect_d
                if w==0 or h==0: 
                    idx+=1; continue
                
                roi = frame[y:y+h, x:x+w]
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                
                # 【修复核心】根据用户选择提取黑字或白字
                mode = cv2.THRESH_BINARY_INV if is_black_text else cv2.THRESH_BINARY
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
                    elif diff_score > p_diff and (idx - d_start)/self.fps > 0.2: should_cut = True
                    
                    if should_cut:
                        dur = (idx - d_start) / self.fps
                        if dur > 0.3: # 过滤极短杂讯
                            st = datetime.timedelta(seconds=d_start/self.fps)
                            et = datetime.timedelta(seconds=idx/self.fps)
                            content = f"Line {sub_index}"
                            
                            if do_ocr and d_best_frame is not None:
                                # 捕获OCR异常，防止闪退
                                try:
                                    text = self.processor.ocr_image(d_best_frame)
                                    if text: content = text
                                except Exception as ocr_err:
                                    self.log(f"⚠️ [Line {sub_index}] OCR 失败，已跳过。")
                            
                            subs.append(srt.Subtitle(index=sub_index, start=st, end=et, content=content))
                            self.log(f"✅ 抓取 [Line {sub_index}]: {st.total_seconds():.1f}s -> {et.total_seconds():.1f}s")
                            sub_index += 1
                        
                        if density > 0.005:
                            d_speaking = True
                            d_start = idx
                            d_peak = density
                            d_max_den = density
                            d_best_frame = roi.copy()
                        else:
                            d_speaking = False
                idx += 1
                
            cap.release()
            
            # 【Aegisub修复核心】编码设为 utf-8-sig
            srt_path = os.path.splitext(self.video_path)[0] + ("_OCR.srt" if do_ocr else ".srt")
            if subs:
                with open(srt_path, "w", encoding="utf-8-sig") as f:
                    f.write(srt.compose(subs))
                self.log(f"🎉 成功！共提取 {len(subs)} 条字幕。\n保存至: {srt_path}")
                self.root.after(0, lambda: messagebox.showinfo("完成", "任务结束，文件已保存！"))
            else:
                self.log("⚠️ 任务结束，但没有提取到任何有效字幕。请检查阈值和黑白字模式。")
                self.root.after(0, lambda: messagebox.showwarning("空结果", "未能提取到字幕。"))
                
        except Exception as e:
            trace = traceback.format_exc()
            self.log(f"❌ [致命错误] 运行崩溃:\n{trace}")
            self.root.after(0, lambda: messagebox.showerror("程序崩溃", str(e)))
        finally:
            self.is_processing = False
            self.root.after(0, lambda: self.btn_run.config(state=tk.NORMAL, text="▶️ 开始作业"))
            self.root.after(0, lambda: self.btn_stop.config(state=tk.DISABLED))

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
