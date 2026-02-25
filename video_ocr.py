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

# --- Windows OCR 模块硬件级导入 ---
HAS_WIN_OCR = False
try:
    from winsdk.windows.media.ocr import OcrEngine
    from winsdk.windows.globalization import Language
    from winsdk.windows.graphics.imaging import SoftwareBitmap, BitmapPixelFormat
    import winsdk.windows.storage.streams as streams
    HAS_WIN_OCR = True
except Exception:
    pass

# ================= 核心算法类 =================
class VideoProcessor:
    def __init__(self, logger):
        self.ocr_engine = None
        self.logger = logger
        self.init_ocr()

    def init_ocr(self):
        if HAS_WIN_OCR:
            try:
                lang = Language("ja-JP")
                if OcrEngine.is_language_supported(lang):
                    self.ocr_engine = OcrEngine.try_create_from_language(lang)
                    self.logger("✅ [系统] Windows OCR 引擎已就绪 (ja-JP)。")
                else:
                    self.logger("⚠️ [系统] 系统未找到日语 OCR 语言包。")
            except Exception as e:
                self.logger(f"❌ [系统] OCR 初始化异常: {e}")

    async def _run_win_ocr(self, cv2_img):
        if not self.ocr_engine: return ""
        try:
            # 确保图像有效
            if cv2_img is None or cv2_img.size == 0: return ""
            rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
            height, width, _ = rgb_img.shape
            bytes_data = rgb_img.tobytes()
            
            data_writer = streams.DataWriter()
            data_writer.write_bytes(bytes_data)
            ibuffer = data_writer.detach_buffer()
            
            software_bitmap = SoftwareBitmap.create_copy_from_buffer(
                ibuffer, BitmapPixelFormat.RG_B8, width, height
            )
            result = await self.ocr_engine.recognize_async(software_bitmap)
            return result.text.replace(" ", "").strip()
        except Exception as e:
            return f"[Error: {str(e)[:20]}]"

    def ocr_image(self, img):
        if not HAS_WIN_OCR or self.ocr_engine is None: return ""
        try:
            # 创建新的事件循环防止异步冲突
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            res = loop.run_until_complete(self._run_win_ocr(img))
            loop.close()
            return res
        except:
            return ""

# ================= GUI 主程序 =================
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Subtitle Extractor V10.2 (稳定版)")
        self.root.geometry("1280x900")
        
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
        self.txt_log.config(state=tk.NORMAL)
        self.txt_log.insert(tk.END, message + "\n")
        self.txt_log.see(tk.END)
        self.txt_log.config(state=tk.DISABLED)
        
    def _setup_ui(self):
        # 顶部
        f_top = tk.Frame(self.root, pady=10)
        f_top.pack(fill=tk.X, padx=10)
        
        tk.Button(f_top, text="📂 加载视频", command=self.load_video).pack(side=tk.LEFT)
        self.lbl_info = tk.Label(f_top, text="未加载", fg="gray")
        self.lbl_info.pack(side=tk.LEFT, padx=10)
        
        self.btn_run = tk.Button(f_top, text="▶️ 开始处理", command=self.start_task, bg="#ddffdd", font=("Arial", 10, "bold"))
        self.btn_run.pack(side=tk.RIGHT, padx=5)
        self.btn_stop = tk.Button(f_top, text="🛑 停止", command=self.stop_task, bg="#ffdddd", state=tk.DISABLED)
        self.btn_stop.pack(side=tk.RIGHT, padx=5)
        
        # 功能行
        f_func = tk.Frame(self.root, pady=5)
        f_func.pack(fill=tk.X, padx=10)
        
        self.var_mode = tk.StringVar(value="BLACK") 
        tk.Radiobutton(f_func, text="TWST模式(黑字)", variable=self.var_mode, value="BLACK").pack(side=tk.LEFT)
        tk.Radiobutton(f_func, text="18TRIP模式(白字)", variable=self.var_mode, value="WHITE").pack(side=tk.LEFT, padx=10)
        
        self.var_ocr = tk.BooleanVar(value=True)
        tk.Checkbutton(f_func, text="启用 OCR", variable=self.var_ocr).pack(side=tk.LEFT, padx=10)
        
        tk.Button(f_func, text="🔍 测试单帧 OCR", command=self.test_ocr, bg="#eee").pack(side=tk.LEFT, padx=20)

        # 中间：预览 + 日志
        f_mid = tk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        f_mid.pack(fill=tk.BOTH, expand=True, padx=10)
        
        self.canvas = tk.Canvas(f_mid, bg="black")
        f_mid.add(self.canvas, stretch="always")
        
        f_log_container = tk.Frame(f_mid)
        f_mid.add(f_log_container, width=400)
        self.txt_log = tk.Text(f_log_container, bg="#1e1e1e", fg="#00ff00", font=("Consolas", 9))
        self.txt_log.pack(fill=tk.BOTH, expand=True)

        # 底部滑块
        f_ctrl = tk.Frame(self.root, height=180)
        f_ctrl.pack(fill=tk.X, padx=10, pady=5)
        
        nb = ttk.Notebook(f_ctrl)
        nb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.create_tab(nb, "对话框(绿)", self.rect_d, 0)
        self.create_tab(nb, "选项框(蓝)", self.rect_c, 1)
        self.create_tab(nb, "背景参考(红)", self.rect_b, 2)
        
        f_sets = tk.LabelFrame(f_ctrl, text="设置", padx=5)
        f_sets.pack(side=tk.RIGHT, fill=tk.Y, padx=5)
        self.s_diff = tk.Scale(f_sets, from_=0.1, to=10.0, resolution=0.1, label="切分灵敏度", orient=tk.HORIZONTAL)
        self.s_diff.set(3.0)
        self.s_diff.pack()
        self.s_bin = tk.Scale(f_sets, from_=50, to=255, label="文字亮度阈值", orient=tk.HORIZONTAL, command=self.update_preview)
        self.s_bin.set(130)
        self.s_bin.pack()

        # 进度条
        self.s_time = tk.Scale(self.root, from_=0, to=100, orient=tk.HORIZONTAL, showvalue=0, command=self.on_seek)
        self.s_time.pack(fill=tk.X, padx=10)
        self.progress = ttk.Progressbar(self.root, mode='determinate')
        self.progress.pack(fill=tk.X, padx=10, pady=5)

    def create_tab(self, nb, title, rect_var, rid):
        f = tk.Frame(nb)
        nb.add(f, text=title)
        self.sliders = getattr(self, "sliders", {})
        if rid not in self.sliders: self.sliders[rid] = []
        labels = ["X", "Y", "W", "H"]
        for i in range(4):
            tk.Label(f, text=labels[i]).pack(side=tk.LEFT)
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

    def test_ocr(self):
        if not self.cap: return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(self.s_time.get()))
        ret, frame = self.cap.read()
        if ret:
            x,y,w,h = self.rect_d
            roi = frame[y:y+h, x:x+w]
            self.log("🧪 正在测试当前帧 OCR...")
            res = self.processor.ocr_image(roi)
            if res: self.log(f"结果: {res}")
            else: self.log("结果: (无识别内容，请确保语言包已安装且阈值正确)")

    def load_video(self):
        path = filedialog.askopenfilename()
        if not path: return
        self.video_path = path
        self.cap = cv2.VideoCapture(path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        w, h = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.s_time.config(to=self.total_frames)
        self.lbl_info.config(text=f"{os.path.basename(path)} ({w}x{h})")
        for slist in self.sliders.values():
            for s in slist: s.config(to=max(w, h))
        self.update_preview()

    def on_seek(self, val): self.update_preview()

    def update_preview(self, _=None):
        if not self.cap or self.is_processing: return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(self.s_time.get()))
        ret, frame = self.cap.read()
        if ret:
            # 绘制预览框
            xd, yd, wd, hd = self.rect_d
            cv2.rectangle(frame, (xd, yd), (xd+wd, yd+hd), (0, 255, 0), 2)
            xc, yc, wc, hc = self.rect_c
            cv2.rectangle(frame, (xc, yc), (xc+wc, yc+hc), (255, 255, 0), 2)
            xb, yb, wb, hb = self.rect_b
            cv2.rectangle(frame, (xb, yb), (xb+wb, yb+hb), (0, 0, 255), 2)
            
            roi = frame[yd:yd+hd, xd:xd+wd]
            if roi.size > 0:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                mode = cv2.THRESH_BINARY_INV if self.var_mode.get() == "BLACK" else cv2.THRESH_BINARY
                _, bin_img = cv2.threshold(gray, self.s_bin.get(), 255, mode)
                bin_c = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)
                frame[yd:yd+hd, xd:xd+wd] = cv2.addWeighted(roi, 0.3, bin_c, 0.7, 0)

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
            if cw > 1: img.thumbnail((cw, ch))
            self.photo = ImageTk.PhotoImage(img)
            self.canvas.create_image(cw//2, ch//2, image=self.photo, anchor=tk.CENTER)

    def start_task(self):
        if not self.video_path: return
        self.is_processing = True
        self.btn_run.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        threading.Thread(target=self.run_process, daemon=True).start()

    def stop_task(self): 
        self.is_processing = False
        self.log("⏹️ 正在停止...")

    def run_process(self):
        try:
            # 快照
            pd, pc, pb = list(self.rect_d), list(self.rect_c), list(self.rect_b)
            p_diff = self.s_diff.get() / 100.0
            p_bin = self.s_bin.get()
            is_black = (self.var_mode.get() == "BLACK")
            do_ocr = self.var_ocr.get()
            
            cap = cv2.VideoCapture(self.video_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            subs = []
            
            # --- 分离状态机 ---
            d_state = {"active": False, "start": 0, "peak": 0.0, "best_frame": None, "max_den": 0.0, "last_dil": None}
            c_state = {"active": False, "start": 0}
            
            sub_index = 1
            kernel = np.ones((3,3), np.uint8)
            LOWER_COLOR = np.array([0, 0, 130]) 
            UPPER_COLOR = np.array([180, 100, 255])
            
            idx = 0
            while self.is_processing:
                ret, frame = cap.read()
                if not ret: break
                if idx % 100 == 0:
                    self.root.after(0, lambda v=(idx/total)*100: self.progress.config(value=v))

                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                
                # 1. 对话框检测 (绿)
                xd, yd, wd, hd = pd
                roi_d = frame[yd:yd+hd, xd:xd+wd]
                ratio_d = cv2.countNonZero(cv2.inRange(hsv[yd:yd+hd, xd:xd+wd], LOWER_COLOR, UPPER_COLOR)) / (wd * hd)
                
                den_d = 0.0
                diff_d = 0.0
                if ratio_d > 0.4:
                    gray_d = cv2.cvtColor(roi_d, cv2.COLOR_BGR2GRAY)
                    mode_d = cv2.THRESH_BINARY_INV if is_black else cv2.THRESH_BINARY
                    _, bin_d = cv2.threshold(gray_d, p_bin, 255, mode_d)
                    dil_d = cv2.dilate(bin_d, kernel, iterations=1)
                    den_d = cv2.countNonZero(dil_d) / (wd * hd)
                    if d_state["last_dil"] is not None:
                        diff_d = cv2.countNonZero(cv2.absdiff(dil_d, d_state["last_dil"])) / (wd * hd)
                    d_state["last_dil"] = dil_d.copy()
                else:
                    d_state["last_dil"] = None

                # 2. 选项检测 (蓝)
                xc, yc, wc, hc = pc
                xb, yb, wb, hb = pb
                ratio_c = cv2.countNonZero(cv2.inRange(hsv[yc:yc+hc, xc:xc+wc], LOWER_COLOR, UPPER_COLOR)) / (wc * hc)
                ratio_b = cv2.countNonZero(cv2.inRange(hsv[yb:yb+hb, xb:xb+wb], LOWER_COLOR, UPPER_COLOR)) / (wb * hb)
                is_c = (ratio_c > 0.6) and (ratio_c > ratio_b + 0.3)

                # --- 状态处理：对话 ---
                if not d_state["active"]:
                    if den_d > 0.005:
                        d_state.update({"active": True, "start": idx, "peak": den_d, "max_den": den_d, "best_frame": roi_d.copy()})
                else:
                    if den_d > d_state["peak"]: d_state["peak"] = den_d
                    if den_d > d_state["max_den"] + 0.001:
                        d_state["max_den"] = den_d
                        d_state["best_frame"]
