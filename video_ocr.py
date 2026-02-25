import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
import numpy as np
import srt
import datetime
import threading
import os
import asyncio
from PIL import Image, ImageTk

# 尝试导入 Windows OCR 库
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
    def __init__(self):
        self.ocr_engine = None
        if HAS_WIN_OCR:
            # 初始化日语 OCR 引擎
            lang = Language("ja-JP")
            if OcrEngine.is_language_supported(lang):
                self.ocr_engine = OcrEngine.try_create_from_language(lang)

    async def _run_win_ocr(self, cv2_img):
        """调用 Windows 原生 OCR"""
        if not self.ocr_engine: return ""
        try:
            # OpenCV (BGR) -> RGB
            rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
            height, width, _ = rgb_img.shape
            
            # 转换为 Windows SoftwareBitmap
            # 这里需要一点字节流操作
            bytes_data = rgb_img.tobytes()
            data_writer = streams.DataWriter()
            data_writer.write_bytes(bytes_data)
            
            ibuffer = data_writer.detach_buffer()
            software_bitmap = SoftwareBitmap.create_copy_from_buffer(
                ibuffer, 
                winsdk.windows.graphics.imaging.BitmapPixelFormat.RG_B8,
                width, 
                height
            )
            
            # 识别
            result = await self.ocr_engine.recognize_async(software_bitmap)
            return result.text.replace(" ", "") # 去除空格
        except Exception as e:
            print(f"OCR Error: {e}")
            return ""

    def ocr_image(self, img):
        """同步包装异步OCR"""
        if not HAS_WIN_OCR: return "OCR_NOT_INSTALLED"
        return asyncio.run(self._run_win_ocr(img))

# ================= GUI 主程序 =================

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("TWST 自动化综合作业工具 V9.0 (WinOCR集成版)")
        self.root.geometry("1200x900")
        
        # 参数
        self.rect_d = [320, 465, 630, 100] 
        self.rect_c = [430, 170, 450, 90]  
        self.rect_b = [100, 100, 150, 150] 
        self.diff_thresh = 3.0
        self.bin_thresh = 130
        
        self.processor = VideoProcessor()
        self.video_path = ""
        self.is_processing = False
        
        self._setup_ui()
        
    def _setup_ui(self):
        # 顶部
        f_top = tk.Frame(self.root, pady=10)
        f_top.pack(fill=tk.X)
        tk.Button(f_top, text="📂 加载视频", command=self.load_video, font=("微软雅黑", 10)).pack(side=tk.LEFT, padx=10)
        self.lbl_info = tk.Label(f_top, text="未加载", fg="gray")
        self.lbl_info.pack(side=tk.LEFT)
        
        # 核心开关
        self.var_ocr = tk.BooleanVar(value=False)
        cb_ocr = tk.Checkbutton(f_top, text="启用 Windows OCR (识别日语)", variable=self.var_ocr, font=("微软雅黑", 10, "bold"), fg="blue")
        cb_ocr.pack(side=tk.RIGHT, padx=10)
        if not HAS_WIN_OCR:
            cb_ocr.config(state=tk.DISABLED, text="未检测到 Windows OCR 库")
            
        self.btn_run = tk.Button(f_top, text="▶️ 开始作业", command=self.start_task, bg="#ddffdd", font=("微软雅黑", 12))
        self.btn_run.pack(side=tk.RIGHT, padx=10)

        # 预览区
        f_mid = tk.Frame(self.root, bg="#222")
        f_mid.pack(fill=tk.BOTH, expand=True, padx=10)
        self.canvas = tk.Canvas(f_mid, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 控制区
        f_ctrl = tk.Frame(self.root, height=200)
        f_ctrl.pack(fill=tk.X, padx=10, pady=5)
        
        # 选项卡
        nb = ttk.Notebook(f_ctrl)
        nb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.create_tab(nb, "对话框 (绿)", self.rect_d, 0)
        self.create_tab(nb, "选项框 (蓝)", self.rect_c, 1)
        self.create_tab(nb, "背景 (红)", self.rect_b, 2)
        
        # 阈值设置
        f_sets = tk.LabelFrame(f_ctrl, text="敏感度设置", padx=5)
        f_sets.pack(side=tk.RIGHT, fill=tk.Y, padx=5)
        
        tk.Label(f_sets, text="防连读灵敏度:").pack(anchor="w")
        self.s_diff = tk.Scale(f_sets, from_=0.1, to=10.0, resolution=0.1, orient=tk.HORIZONTAL, command=self.update_preview)
        self.s_diff.set(self.diff_thresh)
        self.s_diff.pack(fill=tk.X)
        
        tk.Label(f_sets, text="文字亮度阈值:").pack(anchor="w")
        self.s_bin = tk.Scale(f_sets, from_=50, to=255, orient=tk.HORIZONTAL, command=self.update_preview)
        self.s_bin.set(self.bin_thresh)
        self.s_bin.pack(fill=tk.X)

        # 底部
        f_bot = tk.Frame(self.root)
        f_bot.pack(fill=tk.X, padx=10, pady=5)
        self.s_time = tk.Scale(f_bot, from_=0, to=100, orient=tk.HORIZONTAL, showvalue=0, command=self.on_seek)
        self.s_time.pack(fill=tk.X)
        self.progress = ttk.Progressbar(self.root, mode='determinate')
        self.progress.pack(fill=tk.X)

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
        self.lbl_info.config(text=os.path.basename(path))
        
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
            # 绘制框
            x,y,w,h = self.rect_d
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            x,y,w,h = self.rect_c
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,255,0), 2)
            x,y,w,h = self.rect_b
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)
            
            # 显示二值化效果 (辅助调试)
            # 这里仅展示绿框区域的二值化情况
            roi = frame[y:y+h, x:x+w]
            if roi.size > 0:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                _, bin = cv2.threshold(gray, self.s_bin.get(), 255, cv2.THRESH_BINARY)
                bin_c = cv2.cvtColor(bin, cv2.COLOR_GRAY2BGR)
                frame[y:y+h, x:x+w] = cv2.addWeighted(roi, 0.5, bin_c, 0.5, 0)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
            if cw > 1: img.thumbnail((cw, ch))
            self.photo = ImageTk.PhotoImage(img)
            self.canvas.create_image(cw//2, ch//2, image=self.photo, anchor=tk.CENTER)

    def start_task(self):
        if not self.video_path: return
        self.is_processing = True
        self.btn_run.config(state=tk.DISABLED, text="处理中...")
        threading.Thread(target=self.run_process, daemon=True).start()

    def run_process(self):
        try:
            # 快照参数 (锁定当前滑块设置，防止运行中误触)
            p_rect_d = list(self.rect_d)
            p_rect_c = list(self.rect_c)
            p_rect_b = list(self.rect_b)
            p_diff = self.s_diff.get() / 100.0
            p_bin = self.s_bin.get()
            do_ocr = self.var_ocr.get()
            
            cap = cv2.VideoCapture(self.video_path)
            subs = []
            
            # --- 状态机变量 ---
            d_speaking = False
            d_start = 0
            d_peak = 0.0
            
            # 【关键新增】记录这句话“最完美”的一帧
            d_best_frame = None
            d_max_density_in_sentence = 0.0
            
            last_dil = None
            kernel = np.ones((3,3), np.uint8)
            
            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                if idx % 50 == 0:
                    prog = (idx / self.total_frames) * 100
                    self.root.after(0, lambda v=prog: self.progress.config(value=v))

                # 1. 绿框处理
                x,y,w,h = p_rect_d
                if w==0 or h==0: continue
                
                # 裁剪出对话框区域
                roi = frame[y:y+h, x:x+w]
                
                # 图像处理
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                # TWST/18Trip 白字黑字逻辑不同，这里假设用通用的阈值处理
                # 如果是18Trip(白字)，用 THRESH_BINARY；如果是TWST(黑字)，用 THRESH_BINARY_INV
                # 这里默认用 INV (TWST模式)，如果你主要跑18T，可以在界面加个开关切换模式
                # 暂且用 INV (黑字模式) 演示，或者你可以根据亮度自动判断
                # 为了通用，这里假设你已经调整好了滑块能把字抠出来
                _, binary = cv2.threshold(roi_gray, p_bin, 255, cv2.THRESH_BINARY_INV)
                
                dilated = cv2.dilate(binary, kernel, iterations=1)
                density = cv2.countNonZero(dilated) / (w * h)
                
                # 突变检测
                diff_score = 0.0
                if last_dil is not None:
                    diff = cv2.absdiff(dilated, last_dil)
                    diff_score = cv2.countNonZero(diff) / (w * h)
                last_dil = dilated.copy()
                
                # 2. 状态判断
                if not d_speaking:
                    # === 开始说话 ===
                    if density > 0.005:
                        d_speaking = True
                        d_start = idx
                        d_peak = density
                        
                        # 初始化最佳帧：刚开始说话，当前帧就是最佳
                        d_max_density_in_sentence = density
                        d_best_frame = roi.copy() 
                else:
                    # === 说话中 ===
                    if density > d_peak: d_peak = density
                    
                    # 【核心逻辑】更新最佳帧
                    # 如果当前帧的字数比之前记录的还多，或者差不多多但画面更稳定
                    # 我们就认为当前帧是更好的OCR素材
                    # (加 0.001 的缓冲是为了防止微小抖动频繁更新)
                    if density > d_max_density_in_sentence + 0.001:
                        d_max_density_in_sentence = density
                        d_best_frame = roi.copy() # 必须用 copy() 存入内存
                    
                    # === 结束判定 ===
                    should_cut = False
                    
                    # 条件1: 没字了
                    if density < 0.003: should_cut = True
                    # 条件2: 字突然变少 (峰值回落)
                    elif density < (d_peak * 0.4) and d_peak > 0.02: should_cut = True
                    # 条件3: 字的形状突变 (防连读)
                    elif diff_score > p_diff and (idx - d_start)/self.fps > 0.2: should_cut = True
                    
                    if should_cut:
                        dur = (idx - d_start) / self.fps
                        if dur > 0.2:
                            st = datetime.timedelta(seconds=d_start/self.fps)
                            et = datetime.timedelta(seconds=idx/self.fps)
                            
                            content = f"Line {len(subs)+1}"
                            
                            # === 触发 OCR (使用缓存的最佳帧) ===
                            if do_ocr and d_best_frame is not None:
                                # 注意：OCR很耗时，这里是单线程会卡顿界面
                                # 但为了数据准确，必须等OCR完成
                                text = self.processor.ocr_image(d_best_frame)
                                if text.strip(): content = text.strip()
                            
                            subs.append(srt.Subtitle(index=len(subs)+1, start=st, end=et, content=content))
                        
                        # 连读处理：如果切断时屏幕上还有字，说明是连读
                        if density > 0.005:
                            d_speaking = True
                            d_start = idx
                            d_peak = density
                            # 重置最佳帧为当前新句子的第一帧
                            d_max_density_in_sentence = density
                            d_best_frame = roi.copy()
                        else:
                            d_speaking = False
            
            # 保存 SRT
            base_name = os.path.splitext(self.video_path)[0]
            suffix = "_OCR.srt" if do_ocr else ".srt"
            srt_path = base_name + suffix
            
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(srt.compose(subs))
            
            self.root.after(0, lambda: messagebox.showinfo("完成", f"已生成字幕: {srt_path}"))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("错误", str(e)))
        finally:
            self.is_processing = False
            self.root.after(0, lambda: self.btn_run.config(state=tk.NORMAL, text="▶️ 开始作业"))

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
