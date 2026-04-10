import tkinter as tk
from tkinter import messagebox, ttk
import os
import cv2
from PIL import Image, ImageTk
import threading
from face_recognition_system import EmployeeFaceRecognitionSystem
from database import (save_employee, get_all_employees,
                      get_next_employee_id, log_auth_attempt,
                      get_auth_logs)


class FaceRecognitionApp:
    SKY    = "#87CEEB"
    L_BLUE = "#B0E0E6"
    D_BLUE = "#4682B4"
    WHITE  = "#FFFFFF"
    RED    = "#E74C3C"
    GREEN  = "#2ECC71"
    TEXT   = "#2C3E50"

    def __init__(self, root: tk.Tk, user: dict = None):
        self.root = root
        self.user = user or {"username": "guest", "role": "staff"}
        self.root.title("Employee Face Recognition System")
        self.root.geometry("1000x600")
        self.root.resizable(True, True)
        self.root.configure(bg=self.WHITE)

        self.apply_theme()

        self.face_system = EmployeeFaceRecognitionSystem()
        self._sync_employees_from_db()

        self.is_camera_running = False
        self.camera_thread     = None

        self._build_ui()

    # ── sync DB employees into face_system ──────────────────────────────────
    def _sync_employees_from_db(self):
        """Load employees from MySQL into the face recognition system."""
        db_employees = get_all_employees()
        if db_employees:
            self.face_system.employee_ids = db_employees
            self.face_system.employee_names = set(db_employees.values())
            self.face_system.id_counter = max(db_employees.keys()) + 1

    # ── theme ────────────────────────────────────────────────────────────────
    def apply_theme(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except:
            pass
        style.configure("Main.TFrame",   background=self.WHITE)
        style.configure("TLabel",        background=self.WHITE, foreground="black")
        style.configure("Header.TLabel", background=self.WHITE, foreground=self.D_BLUE)
        style.configure("Status.TLabel", background=self.SKY,   foreground="black",
                        relief=tk.SUNKEN, padding=3)
        style.configure("Camera.TLabel", background=self.WHITE)
        style.configure("Action.TButton", background=self.SKY,  foreground="black")
        style.map("Action.TButton",
                  background=[('active', self.L_BLUE), ('pressed', self.D_BLUE)],
                  foreground=[('pressed', self.WHITE),  ('active', 'black')])
        style.configure("Logout.TButton", background="#FFB6C1")
        style.map("Logout.TButton",
                  background=[('active', "#FF6B6B"), ('pressed', "#FF0000")],
                  foreground=[('pressed', self.WHITE),  ('active', 'black')])
        style.configure("Camera.TLabelframe",       background=self.WHITE)
        style.configure("Camera.TLabelframe.Label", foreground=self.D_BLUE,
                        background=self.WHITE, font=("Arial", 10, "bold"))
        self.root.configure(background=self.WHITE)

    # ── UI ───────────────────────────────────────────────────────────────────
    def _build_ui(self):
        main_frame = ttk.Frame(self.root, padding=20, style="Main.TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # ── top bar: title + logout ──────────────────────────────────────────
        top_bar = ttk.Frame(main_frame, style="Main.TFrame")
        top_bar.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(top_bar, text="Employee Face Recognition System",
                  font=("Arial", 18, "bold"),
                  style="Header.TLabel").pack(side=tk.LEFT)

        # logout button — top right
        self.logout_btn = ttk.Button(
            top_bar, text="🚪 Logout",
            style="Logout.TButton", width=12,
            command=self._logout)
        self.logout_btn.pack(side=tk.RIGHT, padx=(10, 0))

        # user info label next to logout
        ttk.Label(top_bar,
                  text=f"👤 {self.user['username'].title()} ({self.user['role']})",
                  font=("Arial", 9), style="TLabel").pack(side=tk.RIGHT)

        # ── action buttons ───────────────────────────────────────────────────
        btn_frame = ttk.Frame(main_frame, style="Main.TFrame")
        btn_frame.pack(pady=8)

        self.register_btn = ttk.Button(
            btn_frame, text="Register New Employee", width=25,
            command=self.register_employee, style="Action.TButton")
        self.register_btn.grid(row=0, column=0, padx=10, pady=6)

        self.train_btn = ttk.Button(
            btn_frame, text="Train Model", width=25,
            command=self.train_model, style="Action.TButton")
        self.train_btn.grid(row=0, column=1, padx=10, pady=6)

        self.auth_btn = ttk.Button(
            btn_frame, text="Authenticate Employee", width=25,
            command=self.authenticate_employee, style="Action.TButton")
        self.auth_btn.grid(row=1, column=0, padx=10, pady=6)

        self.logs_btn = ttk.Button(
            btn_frame, text="View Auth Logs", width=25,
            command=self.view_logs, style="Action.TButton")
        self.logs_btn.grid(row=1, column=1, padx=10, pady=6)

        # ── split content area ───────────────────────────────────────────────
        content = ttk.Frame(main_frame, style="Main.TFrame")
        content.pack(fill=tk.BOTH, expand=True, pady=8)

        # left: camera
        left = ttk.Frame(content, style="Main.TFrame")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        cam_frame = ttk.LabelFrame(left, text="Camera Feed",
                                   padding=10, style="Camera.TLabelframe")
        cam_frame.pack(fill=tk.BOTH, expand=True)
        self.camera_label = ttk.Label(cam_frame, style="Camera.TLabel")
        self.camera_label.pack(fill=tk.BOTH, expand=True)

        # right: info panel
        right = ttk.Frame(content, style="Main.TFrame", width=250)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        right.pack_propagate(False)

        info_frame = ttk.LabelFrame(right, text="Authentication Info",
                                    padding=10, style="Camera.TLabelframe")
        info_frame.pack(fill=tk.BOTH, expand=True)

        self.auth_status_var     = tk.StringVar(value="Waiting...")
        self.auth_name_var       = tk.StringVar(value="N/A")
        self.auth_confidence_var = tk.StringVar(value="N/A")

        for label, var in [("Status:",     self.auth_status_var),
                            ("Employee:",   self.auth_name_var),
                            ("Confidence:", self.auth_confidence_var)]:
            ttk.Label(info_frame, text=label,
                      font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(5, 0))
            ttk.Label(info_frame, textvariable=var).pack(anchor=tk.W, pady=(0, 8))

        ttk.Label(info_frame, text="Employee Photo:",
                  font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(5, 0))
        self.auth_photo_label = ttk.Label(info_frame, style="Camera.TLabel",
                                          borderwidth=2, relief="solid")
        self.auth_photo_label.pack(pady=5)

        # registration helpers
        self.registration_frame = ttk.Frame(cam_frame, style="Main.TFrame")
        self.employee_name_var  = tk.StringVar()
        self.capture_count_var  = tk.StringVar(value="0/20")

        # ── status bar ───────────────────────────────────────────────────────
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self.root, textvariable=self.status_var,
                  relief=tk.SUNKEN, anchor=tk.W,
                  style="Status.TLabel").pack(side=tk.BOTTOM, fill=tk.X)

    # ── button state helper ──────────────────────────────────────────────────
    def set_buttons_state(self, state):
        for btn in (self.register_btn, self.train_btn,
                    self.auth_btn, self.logs_btn, self.logout_btn):
            btn.configure(state=state)

    # ── LOGOUT ───────────────────────────────────────────────────────────────
    def _logout(self):
        if self.is_camera_running:
            messagebox.showwarning(
                "Camera Active",
                "Please stop the current operation before logging out.")
            return

        if not messagebox.askyesno(
                "Logout",
                f"Logout as {self.user['username'].title()}?"):
            return

        # Clear everything and go back to login
        for widget in self.root.winfo_children():
            widget.destroy()

        self.root.geometry("480x560")
        self.root.resizable(False, False)
        self.root.title("Employee Face Recognition System — Login")

        # Re-center
        self.root.update_idletasks()
        w, h = 480, 560
        x = (self.root.winfo_screenwidth()  // 2) - (w // 2)
        y = (self.root.winfo_screenheight() // 2) - (h // 2)
        self.root.geometry(f"{w}x{h}+{x}+{y}")

        from login import LoginApp
        LoginApp(self.root)

    # ── VIEW LOGS ────────────────────────────────────────────────────────────
    def view_logs(self):
        """Open a window showing recent authentication logs from MySQL."""
        logs = get_auth_logs(limit=50)

        log_win = tk.Toplevel(self.root)
        log_win.title("Authentication Logs")
        log_win.geometry("680x400")
        log_win.configure(bg=self.WHITE)
        log_win.transient(self.root)

        tk.Label(log_win, text="Recent Authentication Logs",
                 font=("Arial", 14, "bold"),
                 fg=self.D_BLUE, bg=self.WHITE).pack(pady=10)

        # table
        cols = ("Employee", "Confidence", "Status", "Time")
        tree = ttk.Treeview(log_win, columns=cols, show="headings", height=15)
        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, width=160)

        for log in logs:
            tree.insert("", tk.END, values=(
                log.get("employee_name", "N/A"),
                f"{log['confidence']:.1f}%" if log.get("confidence") else "N/A",
                log.get("status", "N/A"),
                str(log.get("attempted_at", ""))
            ))

        scrollbar = ttk.Scrollbar(log_win, orient=tk.VERTICAL,
                                  command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 10))

        ttk.Button(log_win, text="Close",
                   command=log_win.destroy,
                   style="Logout.TButton").pack(pady=8)

    # ── REGISTER ─────────────────────────────────────────────────────────────
    def register_employee(self):
        if self.is_camera_running:
            return

        reg_win = tk.Toplevel(self.root)
        reg_win.title("Register New Employee")
        reg_win.geometry("300x150")
        reg_win.resizable(False, False)
        reg_win.transient(self.root)
        reg_win.grab_set()
        reg_win.configure(bg=self.WHITE)

        frame = ttk.Frame(reg_win, style="Main.TFrame")
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Label(frame, text="Enter Employee Name:").pack(pady=10)
        name_entry = ttk.Entry(frame, textvariable=self.employee_name_var,
                               width=30)
        name_entry.pack(pady=5)
        name_entry.focus()

        btn_row = ttk.Frame(frame, style="Main.TFrame")
        btn_row.pack(pady=10)

        def start():
            name = self.employee_name_var.get().strip()
            if not name:
                messagebox.showerror("Error", "Please enter an employee name")
                return
            reg_win.destroy()
            self.status_var.set(f"Registering: {name}")
            self.auth_status_var.set("Registering new employee")
            self.auth_name_var.set(name)
            self.auth_confidence_var.set("N/A")
            self.auth_photo_label.configure(image='')
            self.set_buttons_state('disabled')
            self.camera_thread = threading.Thread(
                target=self.run_registration, daemon=True)
            self.camera_thread.start()

        ttk.Button(btn_row, text="Start", command=start,
                   style="Action.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_row, text="Cancel", command=reg_win.destroy,
                   style="Logout.TButton").pack(side=tk.LEFT, padx=5)

    def run_registration(self):
        name        = self.employee_name_var.get()
        employee_id = get_next_employee_id()

        self.is_camera_running = True
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Failed to open camera")
            self.is_camera_running = False
            self.root.after(0, lambda: self.set_buttons_state('normal'))
            return

        emp_dir = os.path.join(
            self.face_system.data_path, 'training', str(employee_id))
        os.makedirs(emp_dir, exist_ok=True)

        self.face_system.employee_ids[employee_id] = name
        self.face_system.employee_names.add(name)
        self.face_system.id_counter = employee_id + 1

        count = 0
        while count < 20 and self.is_camera_running:
            ret, frame = cap.read()
            if not ret:
                break
            gray, face = self.face_system.detect_face(frame)
            display   = frame.copy()
            if face is not None:
                x, y, w, h = face
                cv2.rectangle(display, (x, y), (x+w, y+h), (135, 206, 235), 2)
                face_img = gray[y:y+h, x:x+w]
                cv2.imwrite(os.path.join(emp_dir, f"{count}.jpg"), face_img)

                crop = frame[y:y+h, x:x+w]
                pil  = Image.fromarray(
                    cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                pil  = pil.resize((150, 150), Image.LANCZOS)
                photo = ImageTk.PhotoImage(image=pil)
                self.auth_photo_label.configure(image=photo)
                self.auth_photo_label.image = photo

                count += 1
                self.capture_count_var.set(f"{count}/20")

                for _ in range(5):
                    if not self.is_camera_running:
                        break
                    ret2, f2 = cap.read()
                    if ret2:
                        self.update_camera_feed(f2)
                    self.root.update()

            self.update_camera_feed(display)

        cap.release()
        self.is_camera_running = False

        # Save to MySQL
        save_employee(employee_id, name)
        self.face_system.save_employee_mapping()

        # Auto-train
        self.status_var.set(f"Training model for {name}...")
        self.face_system.train_model()

        self.status_var.set(f"✅ Registered & trained: {name}")
        self.camera_label.configure(image='')
        self.root.after(0, lambda: self.set_buttons_state('normal'))
        messagebox.showinfo(
            "Registration Complete",
            f"✅ {name} registered with {count} face samples.\n"
            "Model has been automatically retrained.")

    # ── TRAIN ────────────────────────────────────────────────────────────────
    def train_model(self):
        if self.is_camera_running:
            messagebox.showwarning("Busy", "Camera is in use. Please wait.")
            return
        self.status_var.set("Training model...")
        self.auth_status_var.set("Training...")
        self.set_buttons_state('disabled')

        def _train():
            success = self.face_system.train_model()
            self.root.after(0, lambda: self.set_buttons_state('normal'))
            if success:
                self.root.after(0, lambda: self.status_var.set(
                    "✅ Model trained successfully"))
                self.root.after(0, lambda: self.auth_status_var.set(
                    "Ready for authentication"))
                self.root.after(0, lambda: messagebox.showinfo(
                    "Done", "Model trained successfully."))
            else:
                self.root.after(0, lambda: self.status_var.set(
                    "❌ Training failed"))
                self.root.after(0, lambda: messagebox.showerror(
                    "Failed",
                    "Training failed. Register at least one employee first."))

        threading.Thread(target=_train, daemon=True).start()

    # ── AUTHENTICATE ─────────────────────────────────────────────────────────
    def authenticate_employee(self):
        if self.is_camera_running:
            messagebox.showwarning("Busy", "Camera is in use. Please wait.")
            return
        if (not self.face_system.model_trained
                and not self.face_system.load_model()):
            messagebox.showerror(
                "Error", "No trained model. Register & train first.")
            return

        self.status_var.set("Starting authentication...")
        self.auth_status_var.set("Scanning...")
        self.auth_name_var.set("N/A")
        self.auth_confidence_var.set("N/A")
        self.auth_photo_label.configure(image='')
        self.set_buttons_state('disabled')

        self.camera_thread = threading.Thread(
            target=self.run_authentication, daemon=True)
        self.camera_thread.start()

    def run_authentication(self):
        self.is_camera_running = True
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Failed to open camera")
            self.is_camera_running = False
            self.root.after(0, lambda: self.set_buttons_state('normal'))
            return

        start    = cv2.getTickCount()
        timeout  = 10
        auth_result = None

        while self.is_camera_running:
            ret, frame = cap.read()
            if not ret:
                break

            elapsed   = (cv2.getTickCount() - start) / cv2.getTickFrequency()
            remaining = max(0, timeout - elapsed)

            if elapsed > timeout:
                break

            gray, face = self.face_system.detect_face(frame)
            display    = frame.copy()
            cv2.putText(display, f"Time: {remaining:.1f}s",
                        (10, display.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (70, 130, 180), 2)

            if face is not None:
                x, y, w, h = face
                face_img = gray[y:y+h, x:x+w]
                label, confidence = \
                    self.face_system.face_recognizer.predict(face_img)

                if confidence < self.face_system.face_recognizer.getThreshold():
                    name = self.face_system.employee_ids.get(
                        label, f"Unknown_{label}")
                    conf_pct = 100 - confidence

                    self.auth_status_var.set("✅ Authenticated")
                    self.auth_name_var.set(name)
                    self.auth_confidence_var.set(f"{conf_pct:.1f}%")

                    crop  = frame[y:y+h, x:x+w]
                    pil   = Image.fromarray(
                        cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    pil   = pil.resize((150, 150), Image.LANCZOS)
                    photo = ImageTk.PhotoImage(image=pil)
                    self.auth_photo_label.configure(image=photo)
                    self.auth_photo_label.image = photo

                    cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 3)
                    cv2.putText(display, name, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    self.update_camera_feed(display)

                    # Log to MySQL
                    log_auth_attempt("success", label, name, conf_pct)

                    auth_result = {"name": name, "confidence": conf_pct}
                    break  # stop immediately on match

                else:
                    cv2.rectangle(display, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    self.auth_status_var.set("Scanning — not recognised")
                    self.auth_name_var.set("Unknown")
                    self.auth_confidence_var.set("N/A")
            else:
                self.auth_status_var.set("No face detected")
                self.auth_name_var.set("N/A")
                self.auth_confidence_var.set("N/A")

            self.update_camera_feed(display)

        cap.release()
        self.is_camera_running = False
        self.camera_label.configure(image='')
        self.root.after(0, lambda: self.set_buttons_state('normal'))

        if auth_result:
            self.status_var.set(f"✅ Authenticated: {auth_result['name']}")
            self.root.after(0, lambda: messagebox.showinfo(
                "Authentication Successful",
                f"✅ Employee Recognised!\n\n"
                f"Name:       {auth_result['name']}\n"
                f"Confidence: {auth_result['confidence']:.1f}%"))
        else:
            # Log failed attempt
            log_auth_attempt("failed")
            self.status_var.set("❌ Authentication failed")
            self.auth_status_var.set("❌ Not recognised")
            self.root.after(0, lambda: messagebox.showwarning(
                "Authentication Failed",
                "No registered employee was recognised.\n"
                "Please try again or register first."))

    # ── camera feed ──────────────────────────────────────────────────────────
    def update_camera_feed(self, frame):
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img  = Image.fromarray(rgb)
        w, h = self.camera_label.winfo_width(), self.camera_label.winfo_height()
        if w > 1 and h > 1:
            iw, ih = img.size
            ratio  = min(w / iw, h / ih)
            img    = img.resize((int(iw * ratio), int(ih * ratio)),
                                Image.LANCZOS)
        photo = ImageTk.PhotoImage(image=img)
        self.camera_label.configure(image=photo)
        self.camera_label.image = photo


if __name__ == "__main__":
    root = tk.Tk()
    FaceRecognitionApp(root)
    root.mainloop()