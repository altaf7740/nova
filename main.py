import os, sys, json, time, shutil, hashlib, tempfile, threading, subprocess, random
from pathlib import Path
import numpy as np
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import sounddevice as sd
from scipy.io.wavfile import write
from llama_index.readers.file import PDFReader
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.prompts import PromptTemplate
from llama_index.core.postprocessor import SimilarityPostprocessor
from dotenv import load_dotenv
load_dotenv()

PIPER_MODEL=os.getenv("PIPER_MODEL_PATH")

# Configure modern theme first (this doesn't require root window)
ctk.set_appearance_mode(os.getenv("THEME","system"))
ctk.set_default_color_theme("blue")

class NovaSplash(ctk.CTkToplevel):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Define fonts after window creation
        self.title_font = ctk.CTkFont(family="Segoe UI", size=42, weight="bold")
        self.subtitle_font = ctk.CTkFont(family="Segoe UI", size=18)
        self.log_font = ctk.CTkFont(family="Segoe UI", size=14)
        
        self.overrideredirect(True)
        self.attributes("-topmost", True)
        self.w, self.h = 800, 500
        self.cx = (self.winfo_screenwidth() // 2) - (self.w // 2)
        self.cy = (self.winfo_screenheight() // 2) - (self.h // 2)
        self.geometry(f"{self.w}x{self.h}+{self.cx}+{self.cy}")
        
        # Modern gradient background
        self.bg = ctk.CTkFrame(self, fg_color=["#0A0F2D", "#1A1F3D"])
        self.bg.pack(fill="both", expand=True)
        
        self.canvas = tk.Canvas(self.bg, bg="#0A0F2D", highlightthickness=0)
        self.canvas.place(relx=0, rely=0, relwidth=1, relheight=1)
        
        # Modern title with gradient effect
        self.title = ctk.CTkLabel(self.bg, text="NOVA",
            font=self.title_font, 
            text_color="#67D7FF")
        self.title.place(relx=0.5, rely=0.18, anchor="center")
        
        self.sub = ctk.CTkLabel(self.bg, text="Neural Optimized Virtual Assistant",
            font=self.subtitle_font, 
            text_color="#9BCBFF")
        self.sub.place(relx=0.5, rely=0.26, anchor="center")
        
        # Modern progress bar
        self.pb = ctk.CTkProgressBar(self.bg, width=540, height=6, 
                                   progress_color="#67D7FF")
        self.pb.place(relx=0.5, rely=0.85, anchor="center")
        self.pb.set(0)
        
        # Modern log box
        self.log = ctk.CTkTextbox(self.bg, width=540, height=140,
            fg_color="#151A35", text_color="#8AB4F8", font=self.log_font,
            border_width=1, border_color="#2A2F4A")
        self.log.place(relx=0.5, rely=0.62, anchor="center")
        self.log.insert("end", "[BOOT] Initializing core systems...\n")
        
        self.ring_angle = 0
        self.particles = []
        self.matrix_cols = []
        self._init_particles()
        self._init_matrix()
        self.after(10, self._animate)

    def _init_particles(self):
        for _ in range(90):
            x = random.randint(0, self.w)
            y = random.randint(0, self.h)
            vx = (random.random() - 0.5) * 0.6
            vy = (random.random() - 0.5) * 0.6
            r = random.randint(1, 3)
            dot = self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="#1B8FFF", outline="")
            self.particles.append([x, y, vx, vy, r, dot])

    def _init_matrix(self):
        cols = 60
        size = self.w // cols
        for i in range(cols):
            x = i * size + size // 2
            y = random.randint(-400, 0)
            speed = random.randint(2, 8)
            self.matrix_cols.append([x, y, speed, size])
        self.matrix_chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def _draw_matrix(self):
        updated = []
        for x, y, s, size in self.matrix_cols:
            y += s
            if y > self.h:
                y = random.randint(-300, -50)
            ch = random.choice(self.matrix_chars)
            self.canvas.create_text(x, y, text=ch, fill="#0C2E6B", font=("Consolas", 12))
            self.canvas.create_text(x, y-14, text=ch, fill="#124AA8", font=("Consolas", 11))
            updated.append([x, y, s, size])
        self.matrix_cols = updated

    def _move_particles(self):
        for i, (x, y, vx, vy, r, dot) in enumerate(self.particles):
            x += vx
            y += vy
            if x < 0 or x > self.w: vx *= -1
            if y < 0 or y > self.h: vy *= -1
            self.particles[i] = [x, y, vx, vy, r, dot]
            self.canvas.coords(dot, x-r, y-r, x+r, y+r)

    def _draw_ring(self):
        cx, cy = self.w // 2, self.h // 2 - 30
        r1, r2 = 120, 150
        a = self.ring_angle
        for i in range(0, 360, 30):
            start1 = a + i
            start2 = start1 + 220
            self.canvas.create_arc(cx-r2, cy-r2, cx+r2, cy+r2,
                                   start=start1, extent=14,
                                   style="arc", outline="#1976D2", width=2)
            self.canvas.create_arc(cx-r1, cy-r1, cx+r1, cy+r1,
                                   start=start2, extent=10,
                                   style="arc", outline="#33B5FF", width=3)
        self.ring_angle = (self.ring_angle + 4) % 360
        self.canvas.create_oval(cx-2, cy-2, cx+2, cy+2, fill="#33B5FF", outline="")

    def _logs(self, step):
        msgs = [
            "[OK] Loading embeddings...",
            "[OK] Preparing vector index...",
            "[OK] Activating Hindi protocol...",
            "[OK] Securing guardrails...",
            "[OK] Audio I/O initialized...",
            "[OK] UI subsystems ready...",
            "[OK] Boot sequence complete."
        ]
        if step < len(msgs):
            self.log.insert("end", msgs[step] + "\n")
            self.log.see("end")

    def _animate(self, prog=0, step=0):
        self.canvas.delete("anim")
        self._draw_matrix()
        self._move_particles()
        self._draw_ring()
        self.title.configure(text_color=["#67D7FF","#9BE1FF","#67D7FF","#33B5FF"][step % 4])
        if prog < 1:
            self.pb.set(prog)
            self._logs(step)
            self.after(20, lambda: self._animate(prog + 0.02, step + 1))
        else:
            self.after(20, self.destroy)

APP_HOME=Path(__file__).resolve().parent / ".novadocs"
PDF_DIR=APP_HOME/"pdf_files"
INDEX_DIR=APP_HOME/"storage"
PDF_DIR.mkdir(parents=True,exist_ok=True)
INDEX_DIR.mkdir(parents=True,exist_ok=True)

WHISPER_MODEL_PATH=os.getenv("WHISPER_MODEL_PATH")
LLM_MODEL=os.getenv("LLM_MODEL")
EMBED_MODEL="nomic-embed-text"
SAMPLERATE=int(os.getenv("SAMPLERATE"))
TEMPERATURE=float(os.getenv("TEMPERATURE",0))

QA_PROMPT = PromptTemplate(
    "### भाषा निर्देश (अवश्य पालन करें):\n"
    "1. अंतिम उत्तर केवल और केवल हिंदी में होना चाहिए\n"
    "2. किसी भी अंग्रेजी शब्द, वाक्य या तकनीकी टर्म का प्रयोग वर्जित है\n"
    "3. अंग्रेजी शब्दों को हिंदी लिप्यंतरण में बदलें (जैसे: college=कॉलेज, computer=कंप्यूटर, system=सिस्टम)\n"
    "4. उत्तर पूर्णतः हिंदी लिपि (देवनागरी) में हो\n\n"
    
    "### कार्य निर्देश:\n"
    "आपको केवल दिए गए संदर्भ के आधार पर उत्तर देना है।\n"
    "यदि संदर्भ पर्याप्त न हो, तो केवल यही लिखें:\n"
    "\"यह प्रश्न उपलब्ध दस्तावेज़ों के दायरे से बाहर है।\"\n\n"
    
    "### उत्तर शैली:\n"
    "- पेशेवर हिंदी\n"
    "- सरल और स्पष्ट\n"
    "- संक्षिप्त\n"
    "- 100% हिंदी\n\n"
    
    "संदर्भ:\n{context_str}\n\n"
    "प्रश्न:\n{query_str}\n\n"
    "उत्तर (केवल हिंदी में):"
)

REFINE_PROMPT = PromptTemplate(
    "### भाषा निर्देश:\n"
    "अंतिम उत्तर 100% हिंदी में होना चाहिए। कोई अंग्रेजी शब्द नहीं।\n"
    "सभी अंग्रेजी शब्दों को हिंदी लिप्यंतरण में परिवर्तित करें।\n\n"
    
    "मौजूदा उत्तर:\n{existing_answer}\n\n"
    "नया संदर्भ:\n{context_msg}\n\n"
    "हिंदी में संशोधित उत्तर (कोई अंग्रेजी नहीं):"
)

Settings.llm = Ollama(
    model=LLM_MODEL, 
    temperature=TEMPERATURE,
    system=(
        "आप एक हिंदी विशेषज्ञ सहायक हैं। आपका कार्य केवल और केवल हिंदी में उत्तर देना है। "
        "निम्नलिखित नियमों का सख्ती से पालन करें:\n"
        "1. सभी उत्तर 100% शुद्ध हिंदी में हों\n"
        "2. अंग्रेजी के शब्दों को हिंदी लिप्यंतरण में बदलें (जैसे 'college' -> 'कॉलेज')\n"
        "3. तकनीकी शब्दों को भी हिंदी में लिखें या लिप्यंतरित करें\n"
        "4. किसी भी स्थिति में अंग्रेजी के अक्षर, शब्द या वाक्यांश न लिखें\n"
        "5. यदि कोई शब्द हिंदी में नहीं है, तो उसका उच्चारण के आधार पर हिंदी लिप्यंतरण करें\n"
        "6. उत्तर सरल, स्पष्ट और व्यावसायिक हिंदी में दें"
    )
)
Settings.embed_model=OllamaEmbedding(model_name=EMBED_MODEL)

app=None

def build_or_load_index():
    def pdf_signature():
        items=[]
        pdfs=sorted(PDF_DIR.glob("*.pdf"))
        if not pdfs:return None
        for p in pdfs:items.append(f"{p.name}:{p.stat().st_mtime}:{p.stat().st_size}")
        return hashlib.md5("".join(items).encode()).hexdigest()
    sig=pdf_signature()
    sig_path=INDEX_DIR/"signature.json"
    if not sig:
        app.set_status("Ready. Add a PDF to begin.");app.refresh_pdf_list();return None
    if INDEX_DIR.exists() and sig_path.exists():
        try:
            saved=json.load(open(sig_path))
            if saved.get("sig")==sig:
                ctx=StorageContext.from_defaults(persist_dir=str(INDEX_DIR))
                return load_index_from_storage(ctx)
        except Exception:
            pass
    app.set_status("Processing PDFs...")
    reader=PDFReader()
    docs=[d for f in PDF_DIR.glob("*.pdf") for d in reader.load_data(file=f)]
    if not docs:
        app.set_status("No PDFs found.");return None
    index=VectorStoreIndex.from_documents(docs)
    index.storage_context.persist(persist_dir=str(INDEX_DIR))
    json.dump({"sig":sig},open(sig_path,"w"))
    return index

def transcribe_audio(audio_np):
    p=tempfile.NamedTemporaryFile(delete=False,suffix=".wav").name
    write(p,SAMPLERATE, audio_np.astype(np.int16))
    app.set_status("Transcribing...")
    cmd=["whisper-cli","-m",WHISPER_MODEL_PATH,"-f",p,"--language","auto","--threads","6","--max-context","0","--output-txt"]
    r=subprocess.run(cmd,capture_output=True,text=True)
    os.remove(p)
    if r.returncode!=0:
        app.set_status("Whisper error");return ""
    t=r.stdout.strip().split("]")[-1].strip()
    app.set_user_hint(f"You said: {t}")
    return t

def tts(text):
    if not text:
        return
    try:
        out = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        cmd = ["piper", "--model", PIPER_MODEL, "--output_file", out]
        px = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        px.communicate(input=text.encode())

        from scipy.io import wavfile
        sr, audio = wavfile.read(out)

        sd.play(audio, sr)
        sd.wait()
        os.remove(out)
    except Exception as e:
        print("TTS error:", e)


class NovaApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Define fonts after window creation
        self.large_font = ctk.CTkFont(family="Segoe UI", size=16)
        self.medium_font = ctk.CTkFont(family="Segoe UI", size=14)
        self.small_font = ctk.CTkFont(family="Segoe UI", size=12)
        self.heading_font = ctk.CTkFont(family="Segoe UI", size=24, weight="bold")
        self.subheading_font = ctk.CTkFont(family="Segoe UI", size=18, weight="bold")
        
        self.title("Nova | Taikisha India")
        self.geometry("1400x900")
        
        # Set protocol for window close
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Configure modern grid layout
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(1, weight=1)
        
        # Modern header with gradient
        self.header = ctk.CTkFrame(self, corner_radius=0, fg_color=["#1C3A6B", "#2A4A7B"])
        self.header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=0, pady=0)
        self.header.grid_columnconfigure(0, weight=1)
        
        self.status = ctk.CTkLabel(self.header, text="Initializing...", 
                                 anchor="w", font=self.large_font, text_color="white")
        self.status.grid(row=0, column=0, padx=20, pady=15, sticky="ew")
        
        # Modern card-style body
        self.body_left = ctk.CTkFrame(self, corner_radius=12, fg_color=["#F8F9FA", "#2A2A2A"])
        self.body_left.grid(row=1, column=0, sticky="nsew", padx=(16, 8), pady=16)
        self.body_left.grid_columnconfigure(0, weight=1)
        self.body_left.grid_rowconfigure(0, weight=1)
        
        # Modern answer area with larger font
        self.answer = ctk.CTkTextbox(self.body_left, wrap="word", font=self.large_font,
                                   fg_color=["#FFFFFF", "#1E1E1E"], border_width=1,
                                   border_color=["#E0E0E0", "#404040"])
        self.answer.grid(row=0, column=0, sticky="nsew", padx=16, pady=16)
        
        # User hint with modern styling
        self.user_hint = ctk.CTkLabel(self.body_left, text="", anchor="w", 
                                    font=self.medium_font, text_color=["#666666", "#AAAAAA"])
        self.user_hint.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 8))
        
        # Modern input row
        self.input_row = ctk.CTkFrame(self.body_left, fg_color="transparent")
        self.input_row.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 16))
        self.input_row.grid_columnconfigure(0, weight=1)
        
        self.entry = ctk.CTkEntry(self.input_row, placeholder_text="Type your question here...",
                                font=self.large_font, height=40)
        self.entry.grid(row=0, column=0, sticky="ew", padx=(0, 12))
        
        self.ask_btn = ctk.CTkButton(self.input_row, text="Ask", font=self.large_font,
                                   height=40, width=80, command=self.ask_text)
        self.ask_btn.grid(row=0, column=1, padx=(0, 12))
        
        self.mic_btn = ctk.CTkButton(self.input_row, text="🎤 Hold to Talk", 
                                   font=self.large_font, height=40, width=120,
                                   fg_color=["#4CAF50", "#45a049"], hover_color=["#43A047", "#3D8B40"])
        self.mic_btn.grid(row=0, column=2)
        self.mic_btn.bind("<ButtonPress-1>", self.start_listen)
        self.mic_btn.bind("<ButtonRelease-1>", self.stop_listen)
        
        # Modern right panel
        self.body_right = ctk.CTkFrame(self, corner_radius=12, fg_color=["#F8F9FA", "#2A2A2A"])
        self.body_right.grid(row=1, column=1, sticky="nsew", padx=(8, 16), pady=16)
        self.body_right.grid_rowconfigure(1, weight=1)
        self.body_right.grid_columnconfigure(0, weight=1)
        
        # Knowledge base header
        self.kb_head = ctk.CTkFrame(self.body_right, fg_color="transparent")
        self.kb_head.grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 8))
        
        self.kb_title = ctk.CTkLabel(self.kb_head, text="📚 Knowledge Base", 
                                   font=self.subheading_font)
        self.kb_title.pack(side="left", padx=8)
        
        self.add_btn = ctk.CTkButton(self.kb_head, text="+ Add PDFs", 
                                   font=self.medium_font, height=35,
                                   fg_color=["#2196F3", "#1976D2"], hover_color=["#1976D2", "#1565C0"])
        self.add_btn.pack(side="right", padx=8)
        self.add_btn.configure(command=self.add_pdfs)
        
        # Modern scrollable list
        self.kb_list = ctk.CTkScrollableFrame(self.body_right, fg_color="transparent")
        self.kb_list.grid(row=1, column=0, sticky="nsew", padx=16, pady=8)
        
        self.ptt = None
        self.query_engine = None
        self.is_indexing = False
        self.is_listening = False
        self.audio_buf = []
        self.stream = None
        
        # Bind Enter key and setup
        self.bind("<Return>", lambda e: self.ask_text())
        threading.Thread(target=self.init_backend, daemon=True).start()

    def on_closing(self):
        """Handle proper application exit"""
        try:
            if self.is_listening and self.stream:
                self.stream.stop()
                self.stream.close()
            # Destroy all windows and exit
            self.quit()
            self.destroy()
            os._exit(0)  # Force exit to prevent hanging threads
        except Exception:
            os._exit(0)

    def init_backend(self):
        self.toggle(False)
        self.is_indexing = True
        idx = build_or_load_index()
        if idx:
            post = [SimilarityPostprocessor(similarity_cutoff=0.10)]
            self.query_engine = idx.as_query_engine(
                similarity_top_k=4,
                streaming=True,
                text_qa_template=QA_PROMPT,
                refine_template=REFINE_PROMPT,
                node_postprocessors=post
            )
        else:
            self.query_engine = None
        self.is_indexing = False
        self.toggle(True)
        self.refresh_pdf_list()
        self.set_status("Ready. Hold to talk or type.")

    def refresh_pdf_list(self):
        for w in self.kb_list.winfo_children():
            w.destroy()
            
        items = sorted(PDF_DIR.glob("*.pdf"))
        if not items:
            no_files_label = ctk.CTkLabel(self.kb_list, text="No PDFs loaded", 
                                        font=self.medium_font, text_color=["#666666", "#AAAAAA"])
            no_files_label.pack(anchor="w", padx=8, pady=12)
            return
            
        for p in items:
            row = ctk.CTkFrame(self.kb_list, corner_radius=8, 
                             fg_color=["#FFFFFF", "#3A3A3A"])
            row.pack(fill="x", padx=8, pady=6)
            
            # File name with ellipsis for long names
            name_label = ctk.CTkLabel(row, text=p.name, font=self.medium_font,
                                    anchor="w")
            name_label.pack(side="left", padx=12, pady=8, fill="x", expand=True)
            
            # Modern action buttons
            open_btn = ctk.CTkButton(row, text="Open", width=70, font=self.small_font,
                                   height=28, fg_color=["#4CAF50", "#45a049"])
            open_btn.pack(side="right", padx=6, pady=6)
            open_btn.configure(command=lambda x=p: self.open_pdf(x))
            
            remove_btn = ctk.CTkButton(row, text="Remove", width=80, font=self.small_font,
                                     height=28, fg_color="#B71C1C", hover_color="#8E0000")
            remove_btn.pack(side="right", padx=6, pady=6)
            remove_btn.configure(command=lambda x=p: self.remove_pdf(x))

    def add_pdfs(self):
        files = filedialog.askopenfilenames(
            title="Select PDF Files", 
            filetypes=[("PDF Files", "*.pdf"), ("All Files", "*.*")]
        )
        if not files:
            return
            
        added_count = 0
        for file_path in files:
            try:
                dest_path = PDF_DIR / Path(file_path).name
                shutil.copy2(file_path, dest_path)
                added_count += 1
            except Exception as e:
                print(f"Error copying {file_path}: {e}")
                
        if added_count > 0:
            messagebox.showinfo("Nova", f"Successfully added {added_count} PDF(s). Reindexing...")
            threading.Thread(target=self.init_backend, daemon=True).start()

    def remove_pdf(self, pdf_path):
        if messagebox.askyesno("Confirm Removal", f"Are you sure you want to remove '{pdf_path.name}'?"):
            try:
                os.remove(pdf_path)
                threading.Thread(target=self.init_backend, daemon=True).start()
            except Exception as e:
                messagebox.showerror("Error", f"Could not remove file: {e}")

    def open_pdf(self, pdf_path):
        try:
            if sys.platform == "win32":
                os.startfile(pdf_path)
            elif sys.platform == "darwin":
                subprocess.run(["open", pdf_path], check=True)
            else:
                subprocess.run(["xdg-open", pdf_path], check=True)
        except Exception as e:
            messagebox.showerror("Error", f"Could not open PDF: {e}")

    def ask_text(self):
        if self.is_indexing or not self.query_engine:
            return
            
        question = self.entry.get().strip()
        if not question:
            return
            
        self.entry.delete(0, "end")
        self.set_user_hint(f"You asked: {question}")
        self.clear_answer()
        self.set_status("Processing your question...")
        
        threading.Thread(target=self.answer_worker, args=(question,), daemon=True).start()

    def answer_worker(self, question):
        try:
            response = self.query_engine.query(question)
            
            if hasattr(response, "source_nodes") and not response.source_nodes:
                message = "यह प्रश्न उपलब्ध दस्तावेज़ों के दायरे से बाहर है।"
                self.append_answer(message)
                self.set_status("Ready")
                threading.Thread(target=tts, args=(message,), daemon=True).start()
                return
                
            full_response = ""
            for chunk in response.response_gen:
                full_response += chunk
                self.append_answer(chunk)
                
            self.set_status("Ready")
            threading.Thread(target=tts, args=(full_response,), daemon=True).start()
            
        except Exception as e:
            print(f"Error in answer_worker: {e}")
            self.set_status("Error processing question")

    def start_listen(self, event=None):
        if self.is_indexing or self.is_listening:
            return
            
        self.is_listening = True
        self.audio_buf = []
        self.show_ptt()
        self.set_status("Listening... Speak now")
        
        def audio_callback(indata, frames, time, status):
            if status:
                print(f"Audio status: {status}")
            self.audio_buf.append(indata.copy())
            
        try:
            self.stream = sd.InputStream(
                samplerate=SAMPLERATE,
                channels=1,
                callback=audio_callback,
                dtype='int16'
            )
            self.stream.start()
        except Exception as e:
            print(f"Error starting audio stream: {e}")
            self.set_status("Audio error")
            self.is_listening = False

    def stop_listen(self, event=None):
        if not self.is_listening:
            return
            
        self.is_listening = False
        self.hide_ptt()
        
        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
        except Exception as e:
            print(f"Error stopping audio stream: {e}")
            
        if not self.audio_buf:
            self.set_status("No audio recorded")
            return
            
        # Process audio
        audio_data = np.concatenate(self.audio_buf, axis=0)
        threading.Thread(target=self.voice_flow, args=(audio_data,), daemon=True).start()

    def voice_flow(self, audio_data):
        question = transcribe_audio(audio_data.squeeze())
        if not question:
            self.set_status("Could not transcribe audio")
            return
            
        self.clear_answer()
        self.set_status("Processing your question...")
        self.answer_worker(question)

    def show_ptt(self):
        if self.ptt:
            return
            
        self.ptt = ctk.CTkToplevel(self)
        self.ptt.overrideredirect(True)
        self.ptt.attributes("-alpha", 0.95, "-topmost", True)
        
        w, h = 260, 160
        x = self.winfo_x() + self.winfo_width() // 2 - w // 2
        y = self.winfo_y() + self.winfo_height() // 2 - h // 2
        self.ptt.geometry(f"{w}x{h}+{x}+{y}")
        self.ptt.configure(fg_color=["#1C3A6B", "#2A4A7B"])
        
        # Modern PTT overlay
        ctk.CTkLabel(self.ptt, text="🎤", 
                   font=ctk.CTkFont(size=48, weight="bold")).pack(pady=(20, 0))
        ctk.CTkLabel(self.ptt, text="Listening...", 
                   font=self.large_font, text_color="white").pack(pady=12)

    def hide_ptt(self):
        if self.ptt:
            self.ptt.destroy()
            self.ptt = None

    def set_status(self, text):
        self.after(0, lambda: self.status.configure(text=text))

    def set_user_hint(self, text):
        self.after(0, lambda: self.user_hint.configure(text=text))

    def append_answer(self, text):
        self.after(0, lambda: self._append_text(text))

    def _append_text(self, text):
        self.answer.insert("end", text)
        self.answer.see("end")

    def clear_answer(self):
        self.after(0, lambda: self.answer.delete("1.0", "end"))

    def toggle(self, enabled=True):
        state = "normal" if enabled else "disabled"
        self.entry.configure(state=state)
        self.ask_btn.configure(state=state)
        self.mic_btn.configure(state=state)
        self.add_btn.configure(state=state)


if __name__ == "__main__":
    try:
        # Create and configure root window
        root = ctk.CTk()
        root.withdraw()
        
        # Show splash screen
        splash = NovaSplash(root)
        
        def open_main_app():
            global app
            try:
                app = NovaApp()
                # Ensure proper cleanup when main app starts
                root.after(100, lambda: app.focus())
                app.mainloop()
            except Exception as e:
                print(f"Error starting main app: {e}")
                try:
                    root.quit()
                    root.destroy()
                except:
                    pass
                os._exit(1)
        
        def wait_for_splash():
            if splash.winfo_exists():
                root.after(100, wait_for_splash)
            else:
                open_main_app()
        
        wait_for_splash()
        root.mainloop()
        
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
        try:
            root.quit()
            root.destroy()
        except:
            pass
        os._exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")
        try:
            root.quit()
            root.destroy()
        except:
            pass
        os._exit(1)