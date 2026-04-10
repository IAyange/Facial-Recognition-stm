import tkinter as tk
from tkinter import messagebox
from database import verify_login, setup_database


class LoginApp:
    SKY    = "#87CEEB"
    D_BLUE = "#4682B4"
    WHITE  = "#FFFFFF"
    GRAY   = "#F0F4F8"
    RED    = "#E74C3C"
    GREEN  = "#2ECC71"
    TEXT   = "#2C3E50"

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Employee Face Recognition System — Login")
        self.root.geometry("480x560")
        self.root.resizable(False, False)
        self.root.configure(bg=self.WHITE)
        self._show_password = False
        self._attempts      = 0
        self._max_attempts  = 5
        self._build_ui()
        self._center_window()

    def _build_ui(self):
        tk.Frame(self.root, bg=self.D_BLUE, height=8).pack(fill=tk.X)

        card = tk.Frame(self.root, bg=self.WHITE, padx=50, pady=30)
        card.pack(fill=tk.BOTH, expand=True)

        # icon
        icon_frame = tk.Frame(card, bg=self.SKY, width=80, height=80)
        icon_frame.pack(pady=(0, 15))
        icon_frame.pack_propagate(False)
        tk.Label(icon_frame, text="👁", font=("Segoe UI Emoji", 36),
                 bg=self.SKY).place(relx=0.5, rely=0.5, anchor="center")

        tk.Label(card, text="Welcome Back",
                 font=("Georgia", 22, "bold"),
                 fg=self.D_BLUE, bg=self.WHITE).pack()
        tk.Label(card, text="Employee Face Recognition System",
                 font=("Segoe UI", 10), fg="#7F8C8D",
                 bg=self.WHITE).pack(pady=(2, 20))

        # username
        self._field_label(card, "Username")
        self.username_var = tk.StringVar()
        user_entry = self._styled_entry(card, self.username_var)

        # password
        self._field_label(card, "Password")
        pw_frame = tk.Frame(card, bg=self.WHITE)
        pw_frame.pack(fill=tk.X, pady=(4, 0))
        self.password_var = tk.StringVar()
        self.pw_entry = tk.Entry(
            pw_frame, textvariable=self.password_var,
            show="●", font=("Segoe UI", 12), relief=tk.FLAT,
            bg=self.GRAY, fg=self.TEXT, insertbackground=self.D_BLUE)
        self.pw_entry.pack(side=tk.LEFT, fill=tk.X, expand=True,
                           ipady=10, ipadx=8)
        self.eye_btn = tk.Button(
            pw_frame, text="🙈", font=("Segoe UI Emoji", 12),
            bg=self.GRAY, relief=tk.FLAT, cursor="hand2",
            command=self._toggle_password)
        self.eye_btn.pack(side=tk.RIGHT, ipadx=6)

        # error
        self.error_var = tk.StringVar()
        tk.Label(card, textvariable=self.error_var,
                 font=("Segoe UI", 9), fg=self.RED,
                 bg=self.WHITE).pack(pady=(6, 0))

        # login button
        self.login_btn = tk.Button(
            card, text="Login  →",
            font=("Segoe UI", 12, "bold"),
            bg=self.D_BLUE, fg=self.WHITE,
            activebackground=self.SKY, activeforeground=self.TEXT,
            relief=tk.FLAT, cursor="hand2", pady=12,
            command=self._attempt_login)
        self.login_btn.pack(fill=tk.X, pady=(15, 8))
        self.login_btn.bind("<Enter>",
            lambda e: self.login_btn.config(bg=self.SKY, fg=self.TEXT))
        self.login_btn.bind("<Leave>",
            lambda e: self.login_btn.config(bg=self.D_BLUE, fg=self.WHITE))

        self.root.bind("<Return>", lambda e: self._attempt_login())

        # tk.Label(card, text="🟢 Connected to database",
        #          font=("Segoe UI", 8), fg="#27AE60",
        #          bg=self.WHITE).pack()
        tk.Label(card, text="Authorised personnel only",
                 font=("Segoe UI", 8), fg="#BDC3C7",
                 bg=self.WHITE).pack(side=tk.BOTTOM)
        tk.Frame(self.root, bg=self.SKY, height=6).pack(
            fill=tk.X, side=tk.BOTTOM)

        user_entry.focus()

    def _field_label(self, p, text):
        tk.Label(p, text=text, font=("Segoe UI", 10, "bold"),
                 fg=self.TEXT, bg=self.WHITE,
                 anchor="w").pack(fill=tk.X, pady=(8, 0))

    def _styled_entry(self, p, var):
        e = tk.Entry(p, textvariable=var, font=("Segoe UI", 12),
                     relief=tk.FLAT, bg=self.GRAY, fg=self.TEXT,
                     insertbackground=self.D_BLUE)
        e.pack(fill=tk.X, ipady=10, ipadx=8, pady=(4, 0))
        return e

    def _toggle_password(self):
        self._show_password = not self._show_password
        self.pw_entry.config(show="" if self._show_password else "●")
        self.eye_btn.config(text="👁" if self._show_password else "🙈")

    def _center_window(self, w=480, h=560):
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth()  // 2) - (w // 2)
        y = (self.root.winfo_screenheight() // 2) - (h // 2)
        self.root.geometry(f"{w}x{h}+{x}+{y}")

    def _attempt_login(self):
        username = self.username_var.get().strip()
        password = self.password_var.get()

        if not username or not password:
            self._shake_error("Please enter both username and password.")
            return
        if self._attempts >= self._max_attempts:
            self._shake_error("Too many failed attempts. Restart the app.")
            self.login_btn.config(state=tk.DISABLED)
            return

        user = verify_login(username, password)
        if user:
            self.login_btn.config(
                text=f"✅ Welcome, {user['username'].title()}!",
                state=tk.DISABLED, bg=self.GREEN)
            self.root.update()
            # 150ms delay — just enough to see the welcome message
            self.root.after(150, lambda: self._switch_to_main(user))
        else:
            self._attempts += 1
            remaining = self._max_attempts - self._attempts
            self._shake_error(
                f"Invalid credentials. {remaining} attempt(s) remaining.")
            self.password_var.set("")
            self.pw_entry.focus()

    def _switch_to_main(self, user: dict):
        try:
            from face_recognition_gui import FaceRecognitionApp

            # Clear all login widgets
            for w in self.root.winfo_children():
                w.destroy()

            # Resize & re-center to main app size instantly
            self.root.resizable(True, True)
            self.root.title("Employee Face Recognition System")
            self._center_window(1000, 600)

            # Build main app in same window — zero gap
            app = FaceRecognitionApp(self.root, user=user)
            app.status_var.set(
                f"  ✅  Logged in as: {user['username'].title()} "
                f"({user['role']})   |   Ready")

        except Exception as e:
            messagebox.showerror("Launch Error",
                f"Could not load main application:\n{e}")

    def _shake_error(self, msg):
        self.error_var.set(msg)
        x0, y0 = self.root.winfo_x(), self.root.winfo_y()
        for dx in (8, -8, 6, -6, 4, -4, 0):
            self.root.geometry(f"+{x0+dx}+{y0}")
            self.root.update()
            self.root.after(25)


if __name__ == "__main__":
    setup_database()
    root = tk.Tk()
    LoginApp(root)
    root.mainloop()