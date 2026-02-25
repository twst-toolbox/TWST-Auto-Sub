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
    pass

# ================= 核心算法类 =================
class VideoProcessor:
    def __init__(self, logger):
        self.ocr_engine = None
        self.logger = logger
        if HAS_WIN_OCR:
            # 启动时强制自检
            threading.Thread(target=self.self_test_ocr, daemon=True).start()

    def self_test_ocr(self):
        """启动时自检OCR引擎是否能正常工作"""
        try:
            lang = Language("ja-JP")
            if OcrEngine.is_language_supported(lang):
                self.ocr_engine = OcrEngine.try_create_from_language(lang)
                self.logger("✅ [系统] Windows OCR (日语) 引擎已找到。")
                
                # 创建一个包含“あ”字的测试图片
                test_img = np.zeros((100, 100, 3), dtype=np.uint8)
                cv2.putText(test_img, "a", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3)
                
                test_result = self.ocr_image(test_img)
                if test_result and "a" in test_result.lower():
                     self.logger("✅ [系统] OCR 自检通过，可以正常识别。")
                else:
                    self.logger("⚠️ [系统] OCR 自检失败，引擎可能无法正确工作。")
                    self.ocr_engine = None # 禁用
            else:
                self.logger("❌ [系统] 未安装日语 OCR 语言包。")
        except Exception as e:
            self.logger(f"❌ [系统] OCR 初始化失败: {e}")
            self.ocr_engine = None

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
                ibuffer, BitmapPixelFormat.RG_B8, width, height
            )
            result = await self.ocr_engine.recognize_async(software_bitmap)
            return result.text.replace(" ", "").replace("\n", " ")
        except:
            return ""

    def ocr_image(self, img):
        if not HAS_WIN_OCR: return ""
        try:
            return asyncio.run(self._run_win_ocr(img))
        except:
            return ""

# ================= GUI 主程序 =================
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Subtitle Extractor V10.2 (时间轴修正)")
        self.root.geometry("1280x900")
        self.rect_d, self.rect_c, self.rect_b = [320, 465, 630, 100], [430, 170, 450, 90], [100, 100, 150, 150]
        self.video_path, self.cap, self.total_frames, self.fps, self.is_processing = "", None, 0, 30, False
        self._setup_ui()
        self.processor = VideoProcessor(self.log)
        
    def log(self, message):
        self.txt_log.config(state=tk.NORMAL)
        self.txt_log.insert(tk.END, message + "\n")
        self.txt_log.see(tk.END)
        self.txt_log.config(state=tk.DISABLED)
        
    def _setup_ui(self):
        # 顶部
        f_top1 = tk.Frame(self.root, pady=5); f_top1.pack(fill=tk.X, padx=10)
        tk.Button(f_top1, text="📂 加载视频", command=self.load_video, font=("微软雅黑", 10)).pack(side=tk.LEFT)
        self.lbl_info = tk.Label(f_top1, text="未加载...", fg="blue"); self.lbl_info.pack(side=tk.LEFT, padx=10)
        
        f_top2 = tk.Frame(self.root, pady=5); f_top2.pack(fill=tk.X, padx=10)
        self.var_mode = tk.StringVar(value="BLACK"); tk.Label(f_top2, text="模式:", font=("微软雅黑", 10, "bold")).pack(side=tk.LEFT)
        tk.Radiobutton(f_top2, text="TWST (黑字)", variable=self.var_mode, value="BLACK").pack(side=tk.LEFT)
        tk.Radiobutton(f_top2, text="18TRIP (白字)", variable=self.var_mode, value="WHITE").pack(side=tk.LEFT, padx=10)
        self.var_ocr = tk.BooleanVar(value=False); cb_ocr = tk.Checkbutton(f_top2, text="启用 OCR", variable=self.var_ocr, font=("微软雅黑", 10, "bold"), fg="purple"); cb_ocr.pack(side=tk.LEFT, padx=20)
        if not HAS_WIN_OCR: cb_ocr.config(state=tk.DISABLED, text="OCR不可用(缺winsdk)")
        self.btn_run = tk.Button(f_top2, text="▶️ 开始处理", command=self.start_task, bg="#ddffdd", font=("微软雅黑", 11, "bold")); self.btn_run.pack(side=tk.RIGHT)
        self.btn_stop = tk.Button(f_top2, text="🛑 停止", command=self.stop_task, bg="#ffdddd", font=("微软雅黑", 11), state=tk.DISABLED); self.btn_stop.pack(side=tk.RIGHT, padx=10)

        # 中间
        f_mid = tk.PanedWindow(self.root, orient=tk.HORIZONTAL); f_mid.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.canvas_frame = tk.Frame(f_mid, bg="#222"); f_mid.add(self.canvas_frame, stretch="always")
        self.canvas = tk.Canvas(self.canvas_frame, bg="black"); self.canvas.pack(fill=tk.BOTH, expand=True)
        f_log = tk.Frame(f_mid); f_mid.add(f_log, width=350); tk.Label(f_log, text="📜 运行日志").pack(anchor="w")
        self.txt_log = tk.Text(f_log, bg="#1e1e1e", fg="#00ff00", font=("Consolas", 9), state=tk.DISABLED); self.txt_log.pack(fill=tk.BOTH, expand=True)

        # 底部
        f_ctrl = tk.Frame(self.root, height=150); f_ctrl.pack(fill=tk.X, padx=10, pady=5)
        nb = ttk.Notebook(f_ctrl); nb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.create_tab(nb, "对话框(绿)", self.rect_d, 0); self.create_tab(nb, "选项框(蓝)", self.rect_c, 1); self.create_tab(nb, "背景(红)", self.rect_b, 2)
        f_sets = tk.LabelFrame(f_ctrl, text="设置", padx=5); f_sets.pack(side=tk.RIGHT, fill=tk.Y, padx=5)
        tk.Label(f_sets, text="防连读灵敏度:").pack(anchor="w"); self.s_diff = tk.Scale(f_sets, from_=0.1, to=10.0, resolution=0.1, orient=tk.HORIZONTAL); self.s_diff.set(3.0); self.s_diff.pack(fill=tk.X)
        tk.Label(f_sets, text="文字阈值:").pack(anchor="w"); self.s_bin = tk.Scale(f_sets, from_=50, to=255, orient=tk.HORIZONTAL, command=self.update_preview); self.s_bin.set(130); self.s_bin.pack(fill=tk.X)
        
        f_bot = tk.Frame(self.root); f_bot.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        self.s_time = tk.Scale(f_bot, from_=0, to=100, orient=tk.HORIZONTAL, showvalue=0, command=self.on_seek); self.s_time.pack(fill=tk.X)
        self.progress = ttk.Progressbar(f_bot, mode='determinate'); self.progress.pack(fill=tk.X, pady=5)

    # 省略了部分UI设置函数，因为它们没有变化
    def create_tab(self, nb, title, rect_var, rid):
        f = tk.Frame(nb); nb.add(f, text=title); self.sliders = getattr(self, "sliders", {});
        if rid not in self.sliders: self.sliders[rid] = []; labels = ["X", "Y", "W", "H"];
        for i in range(4):
            tk.Label(f, text=labels[i]).pack(side=tk.LEFT, padx=2); s = tk.Scale(f, from_=0, to=2000, orient=tk.HORIZONTAL, command=lambda v, x=i, r=rid: self.on_rect(v, x, r));
            s.set(rect_var[i]); s.pack(side=tk.LEFT, fill=tk.X, expand=True); self.sliders[rid].append(s);
    def on_rect(self, val, idx, rid):
        val = int(float(val));
        if rid == 0: self.rect_d[idx] = val; elif rid == 1: self.rect_c[idx] = val; elif rid == 2: self.rect_b[idx] = val;
        self.update_preview();
    def load_video(self):
        path = filedialog.askopenfilename();
        if not path: return; self.video_path = path; self.cap = cv2.VideoCapture(path);
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)); self.fps = self.cap.get(cv2.CAP_PROP_FPS);
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT));
        self.s_time.config(to=self.total_frames); self.lbl_info.config(text=f"{os.path.basename(path)} ({w}x{h})");
        for slist in self.sliders.values():
            for s in slist: s.config(to=max(w, h));
        self.update_preview(); self.log("🎬 视频加载成功。");
    def on_seek(self, val): self.update_preview();
    def update_preview(self, _=None):
        if not self.cap or self.is_processing: return; self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(self.s_time.get())); ret, frame = self.cap.read();
        if ret:
            x,y,w,h = self.rect_d; cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2);
            xc,yc,wc,hc = self.rect_c; cv2.rectangle(frame, (xc,yc), (xc+wc,yc+hc), (255,255,0), 2);
            xb,yb,wb,hb = self.rect_b; cv2.rectangle(frame, (xb,yb), (xb+wb,yb+hb), (0,0,255), 2);
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); img = Image.fromarray(frame);
            cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height();
            if cw > 1: img.thumbnail((cw, ch)); self.photo = ImageTk.PhotoImage(img);
            self.canvas.create_image(cw//2, ch//2, image=self.photo, anchor=tk.CENTER);
    def stop_task(self): self.is_processing = False; self.log("⚠️ 停止...");
    def start_task(self):
        if not self.video_path: return; self.is_processing = True;
        self.btn_run.config(state=tk.DISABLED, text="处理中..."); self.btn_stop.config(state=tk.NORMAL);
        self.log("\n🚀 === 开始任务 ===");
        threading.Thread(target=self.run_process, daemon=True).start();
    
    # 【核心修复区】
    def run_process(self):
        try:
            p_rect_d = list(self.rect_d); p_rect_c = list(self.rect_c); p_rect_b = list(self.rect_b);
            p_diff = self.s_diff.get() / 100.0; p_bin = self.s_bin.get();
            do_ocr = self.var_ocr.get(); is_black_text = (self.var_mode.get() == "BLACK");
            
            cap = cv2.VideoCapture(self.video_path); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT));
            
            # --- 🌟 修复 1：统一的收纳盒 ---
            detected_events = [] # 格式: [ (start_frame, end_frame, type, ocr_frame), ... ]
            
            d_speaking = False; d_start = 0; d_peak = 0.0; d_best_frame = None; d_max_den = 0.0;
            c_active = False; c_start = 0;
            last_dil = None; kernel = np.ones((3,3), np.uint8); idx = 0;
            
            while self.is_processing:
                ret, frame = cap.read();
                if not ret: break;
                
                if idx % 100 == 0: self.root.after(0, lambda v=(idx/total)*100: self.progress.config(value=v));

                # --- 对话逻辑 ---
                x,y,w,h = p_rect_d; density_d = 0.0;
                if w > 0 and h > 0:
                    roi = frame[y:y+h, x:x+w]; roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY);
                    mode = cv2.THRESH_BINARY_INV if is_black_text else cv2.THRESH_BINARY;
                    _, binary = cv2.threshold(roi_gray, p_bin, 255, mode);
                    dilated = cv2.dilate(binary, kernel, iterations=1);
                    density_d = cv2.countNonZero(dilated) / (w * h);
                    
                    diff_score = 0.0;
                    if last_dil is not None: diff_score = cv2.countNonZero(cv2.absdiff(dilated, last_dil)) / (w * h);
                    last_dil = dilated.copy();
                    
                    if not d_speaking:
                        if density_d > 0.005: d_speaking, d_start, d_peak, d_max_den, d_best_frame = True, idx, density_d, density_d, roi.copy();
                    else:
                        if density_d > d_peak: d_peak = density_d;
                        if density_d > d_max_den + 0.001: d_max_den, d_best_frame = density_d, roi.copy();
                        
                        should_cut = False;
                        if density_d < 0.002: should_cut = True;
                        elif density_d < (d_peak * 0.4) and d_peak > 0.02: should_cut = True;
                        elif diff_score > p_diff and (idx - d_start)/self.fps > 0.2: should_cut = True;
                        
                        if should_cut:
                            if (idx - d_start) / self.fps > 0.25:
                                detected_events.append( (d_start, idx, "dialogue", d_best_frame.copy() if d_best_frame is not None else None) )
                            if density_d > 0.005:
                                d_speaking, d_start, d_peak, d_max_den, d_best_frame = True, idx, density_d, density_d, roi.copy();
                            else: d_speaking = False;
                
                # --- 选项逻辑 ---
                xc,yc,wc,hc = p_rect_c; is_choice = False;
                if wc > 0 and hc > 0:
                    roi_c_gray = cv2.cvtColor(frame[yc:yc+hc, xc:xc+wc], cv2.COLOR_BGR2GRAY)
                    _, bin_c = cv2.threshold(roi_c_gray, p_bin, 255, mode)
                    is_choice = (cv2.countNonZero(bin_c) / (wc * hc) > 0.1)

                if not c_active:
                    if is_choice: c_active, c_start = True, idx;
                else:
                    if not is_choice:
                        c_active = False;
                        if (idx - c_start) / self.fps > 0.5:
                            detected_events.append( (c_start, idx, "choice", None) )
                
                idx += 1
                
            cap.release()
            
            # --- 🌟 修复 2：统一排序和编号 ---
            self.log("🔄 正在排序时间轴...")
            detected_events.sort(key=lambda x: x[0])
            subs = []
            
            for i, (start_f, end_f, type, ocr_frame) in enumerate(detected_events):
                sub_index = i + 1
                start_time = datetime.timedelta(seconds=start_f/self.fps)
                end_time = datetime.timedelta(seconds=end_f/self.fps)
                
                content = f"Line {sub_index}"
                # 🌟 修复 3：统一添加 [Choice] 标记
                if type == "choice": content += " [Choice]"
                
                if do_ocr and ocr_frame is not None:
                    ocr_text = self.processor.ocr_image(ocr_frame)
                    if ocr_text: content = ocr_text
                
                subs.append(srt.Subtitle(index=sub_index, start=start_time, end=end_time, content=content))
                self.log(f"-> 生成 [L{sub_index}] {type}: {content[:20]}...")

            # 保存
            srt_path = os.path.splitext(self.video_path)[0] + ("_OCR.srt" if do_ocr else ".srt")
            if subs:
                with open(srt_path, "w", encoding="utf-8-sig") as f: f.write(srt.compose(subs));
                self.log(f"🎉 成功！保存至: {srt_path}")
                self.root.after(0, lambda: messagebox.showinfo("完成", "文件已保存！"))
            else:
                self.log("⚠️ 未提取到字幕。")
        except Exception as e:
            self.log(f"❌ 错误: {traceback.format_exc()}")
        finally:
            self.is_processing = False
            self.root.after(0, lambda: self.btn_run.config(state=tk.NORMAL, text="▶️ 开始作业"))
            self.root.after(0, lambda: self.btn_stop.config(state=tk.DISABLED))

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
