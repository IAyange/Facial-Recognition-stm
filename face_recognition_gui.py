import tkinter as tk
from tkinter import messagebox, ttk
import os
import cv2
from PIL import Image, ImageTk
import threading
from face_recognition_system import EmployeeFaceRecognitionSystem

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Employee Face Recognition System")
        self.root.geometry("1000x600")  # Wider window to accommodate side panel
        self.root.resizable(True, True)
        
        # Define theme colors
        self.sky_blue = "#87CEEB"  # Sky blue color
        self.light_blue = "#B0E0E6"  # Powder blue for alternating elements
        self.dark_blue = "#4682B4"  # Steel blue for accents
        self.white = "#FFFFFF"  # White
        
        # Apply custom theme
        self.apply_theme()
        
        self.face_system = EmployeeFaceRecognitionSystem()
        self.is_camera_running = False
        self.camera_thread = None
        
        # Create main frame with background color
        main_frame = ttk.Frame(self.root, padding=20, style="Main.TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header with custom style
        header = ttk.Label(main_frame, text="Employee Face Recognition System", 
                         font=("Arial", 18, "bold"), style="Header.TLabel")
        header.pack(pady=10)
        
        # Buttons frame
        btn_frame = ttk.Frame(main_frame, style="Main.TFrame")
        btn_frame.pack(pady=10)
        
        # Create buttons with custom style
        register_btn = ttk.Button(btn_frame, text="Register New Employee", width=25,
                               command=self.register_employee, style="Action.TButton")
        register_btn.grid(row=0, column=0, padx=10, pady=10)
        
        train_btn = ttk.Button(btn_frame, text="Train Model", width=25,
                           command=self.train_model, style="Action.TButton")
        train_btn.grid(row=0, column=1, padx=10, pady=10)
        
        auth_btn = ttk.Button(btn_frame, text="Authenticate Employee", width=25,
                          command=self.authenticate_employee, style="Action.TButton")
        auth_btn.grid(row=1, column=0, padx=10, pady=10)
        
        quit_btn = ttk.Button(btn_frame, text="Exit", width=25,
                          command=self.exit_app, style="Exit.TButton")
        quit_btn.grid(row=1, column=1, padx=10, pady=10)
        
        # Create split layout for camera and info display
        self.main_content_frame = ttk.Frame(main_frame, style="Main.TFrame")
        self.main_content_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Left frame for camera
        self.left_frame = ttk.Frame(self.main_content_frame, style="Main.TFrame")
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Camera frame with custom border color
        self.camera_frame = ttk.LabelFrame(self.left_frame, text="Camera Feed", 
                                        padding=10, style="Camera.TLabelframe")
        self.camera_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Style for the camera frame label text
        self.camera_label = ttk.Label(self.camera_frame, style="Camera.TLabel")
        self.camera_label.pack(fill=tk.BOTH, expand=True)
        
        # Right frame for info display
        self.right_frame = ttk.Frame(self.main_content_frame, style="Main.TFrame", width=250)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        self.right_frame.pack_propagate(False)  # Prevent frame from shrinking
        
        # Authentication info frame
        self.auth_info_frame = ttk.LabelFrame(self.right_frame, text="Authentication Info", 
                                           padding=10, style="Camera.TLabelframe")
        self.auth_info_frame.pack(fill=tk.BOTH, expand=True)
        
        # Authentication status variables and labels
        self.auth_status_var = tk.StringVar(value="Waiting...")
        self.auth_name_var = tk.StringVar(value="N/A")
        self.auth_confidence_var = tk.StringVar(value="N/A")
        
        ttk.Label(self.auth_info_frame, text="Status:", 
               font=("Arial", 10, "bold"), style="TLabel").pack(anchor=tk.W, pady=(5, 0))
        ttk.Label(self.auth_info_frame, textvariable=self.auth_status_var, 
               style="TLabel").pack(anchor=tk.W, pady=(0, 10))
        
        ttk.Label(self.auth_info_frame, text="Employee:", 
               font=("Arial", 10, "bold"), style="TLabel").pack(anchor=tk.W, pady=(5, 0))
        ttk.Label(self.auth_info_frame, textvariable=self.auth_name_var, 
               style="TLabel").pack(anchor=tk.W, pady=(0, 10))
        
        ttk.Label(self.auth_info_frame, text="Confidence:", 
               font=("Arial", 10, "bold"), style="TLabel").pack(anchor=tk.W, pady=(5, 0))
        ttk.Label(self.auth_info_frame, textvariable=self.auth_confidence_var, 
               style="TLabel").pack(anchor=tk.W, pady=(0, 10))
        
        # Add a photo placeholder for recognized employee
        ttk.Label(self.auth_info_frame, text="Employee Photo:", 
                font=("Arial", 10, "bold"), style="TLabel").pack(anchor=tk.W, pady=(10, 5))
        self.auth_photo_label = ttk.Label(self.auth_info_frame, style="Camera.TLabel", 
                                        borderwidth=2, relief="solid")
        self.auth_photo_label.pack(pady=5)
        
        # Registration variables
        self.registration_frame = ttk.Frame(self.camera_frame, style="Main.TFrame")
        self.employee_name_var = tk.StringVar()
        self.capture_count_var = tk.StringVar()
        self.capture_count_var.set("0/20")
        
        # Status bar with custom style
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, 
                             relief=tk.SUNKEN, anchor=tk.W, style="Status.TLabel")
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def apply_theme(self):
        """Apply custom sky blue and white theme to the application"""
        style = ttk.Style()
        
        # Try to use a modern base theme first
        try:
            style.theme_use("clam")
        except:
            pass
        
        # Configure frame styles
        style.configure("Main.TFrame", background=self.white)
        
        # Configure label styles
        style.configure("TLabel", background=self.white, foreground="black")
        style.configure("Header.TLabel", background=self.white, foreground=self.dark_blue)
        style.configure("Status.TLabel", background=self.sky_blue, foreground="black", 
                       relief=tk.SUNKEN, padding=3)
        style.configure("Camera.TLabel", background=self.white)
        
        # Configure button styles
        style.configure("TButton", background=self.sky_blue, foreground="black")
        style.configure("Action.TButton", background=self.sky_blue, foreground="black")
        style.map("Action.TButton",
                background=[('active', self.light_blue), ('pressed', self.dark_blue)],
                foreground=[('pressed', self.white), ('active', 'black')])
        
        style.configure("Exit.TButton", background="#FFB6C1")  # Light red for exit button
        style.map("Exit.TButton",
                background=[('active', "#FF6B6B"), ('pressed', "#FF0000")],
                foreground=[('pressed', self.white), ('active', 'black')])
        
        # Configure labelframe styles
        style.configure("Camera.TLabelframe", background=self.white)
        style.configure("Camera.TLabelframe.Label", foreground=self.dark_blue, 
                       background=self.white, font=("Arial", 10, "bold"))
        
        # Configure entry style
        style.configure("TEntry", fieldbackground=self.white)
        
        # Set the main window background
        self.root.configure(background=self.white)
    
    def register_employee(self):
        """Open registration dialog and start employee registration process"""
        if self.is_camera_running:
            return
            
        reg_window = tk.Toplevel(self.root)
        reg_window.title("Register New Employee")
        reg_window.geometry("300x150")
        reg_window.resizable(False, False)
        reg_window.transient(self.root)
        reg_window.grab_set()
        
        # Apply theme to dialog
        reg_window.configure(background=self.white)
        
        dialog_frame = ttk.Frame(reg_window, style="Main.TFrame")
        dialog_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Label(dialog_frame, text="Enter Employee Name:", style="TLabel").pack(pady=10)
        name_entry = ttk.Entry(dialog_frame, textvariable=self.employee_name_var, width=30)
        name_entry.pack(pady=5)
        name_entry.focus()
        
        btn_frame = ttk.Frame(dialog_frame, style="Main.TFrame")
        btn_frame.pack(pady=10)
        
        def start_registration():
            name = self.employee_name_var.get().strip()
            if not name:
                messagebox.showerror("Error", "Please enter an employee name")
                return
                
            reg_window.destroy()
            self.status_var.set(f"Registering employee: {name}")
            self.registration_frame.pack(fill=tk.X, pady=5)
            
            # Reset authentication info
            self.auth_status_var.set("Registering new employee")
            self.auth_name_var.set(name)
            self.auth_confidence_var.set("N/A")
            self.auth_photo_label.configure(image='')
            
            # Add labels for registration progress
            if not hasattr(self, 'name_label'):
                self.name_label = ttk.Label(self.registration_frame, textvariable=self.employee_name_var)
                self.name_label.pack(side=tk.LEFT, padx=5)
                
                self.count_label = ttk.Label(self.registration_frame, textvariable=self.capture_count_var)
                self.count_label.pack(side=tk.RIGHT, padx=5)
            
            # Start camera in separate thread for registration
            self.camera_thread = threading.Thread(target=self.run_registration)
            self.camera_thread.daemon = True
            self.camera_thread.start()
        
        ttk.Button(btn_frame, text="Start Registration", command=start_registration, 
                 style="Action.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=reg_window.destroy, 
                 style="Exit.TButton").pack(side=tk.LEFT, padx=5)
    
    def run_registration(self):
        """Run the employee registration process in a separate thread"""
        employee_name = self.employee_name_var.get()
        
        # Get camera preview
        self.is_camera_running = True
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Failed to open camera")
            self.is_camera_running = False
            self.registration_frame.pack_forget()
            return
        
        # Create directory for this employee
        employee_id = self.face_system.id_counter
        employee_dir = os.path.join(self.face_system.data_path, 'training', str(employee_id))
        if not os.path.exists(employee_dir):
            os.makedirs(employee_dir)
        
        # Register the employee in the system's database
        self.face_system.employee_ids[employee_id] = employee_name
        self.face_system.id_counter += 1
        
        count = 0
        last_saved_face = None
        
        while count < 20 and self.is_camera_running:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect face
            gray, face = self.face_system.detect_face(frame)
            
            display_frame = frame.copy()
            
            if face is not None:
                x, y, w, h = face
                
                # Draw rectangle around the face (using theme color)
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (135, 206, 235), 2)  # Sky blue color in BGR
                
                # Take a photo every few frames
                if count < 20:
                    # Save the face image
                    face_img = gray[y:y+h, x:x+w]
                    face_filename = os.path.join(employee_dir, f"{count}.jpg")
                    cv2.imwrite(face_filename, face_img)
                    
                    # Save for display in side panel
                    last_saved_face = frame[y:y+h, x:x+w]
                    
                    count += 1
                    self.capture_count_var.set(f"{count}/20")
                    
                    # Update the photo in the side panel
                    if last_saved_face is not None:
                        face_rgb = cv2.cvtColor(last_saved_face, cv2.COLOR_BGR2RGB)
                        face_pil = Image.fromarray(face_rgb)
                        face_pil = face_pil.resize((150, 150), Image.LANCZOS)
                        face_photo = ImageTk.PhotoImage(image=face_pil)
                        self.auth_photo_label.configure(image=face_photo)
                        self.auth_photo_label.image = face_photo
                    
                    # Add slight delay to allow for different face angles
                    for _ in range(5):
                        if not self.is_camera_running:
                            break
                        ret, frame = cap.read()
                        if ret:
                            self.update_camera_feed(frame)
                        self.root.update()
                
            # Display camera feed
            self.update_camera_feed(display_frame)
        
        cap.release()
        self.is_camera_running = False
        
        # Save the employee mapping
        self.face_system.save_employee_mapping()
        
        # Update UI
        self.status_var.set(f"Registration completed for {employee_name}")
        self.registration_frame.pack_forget()
        
        # Clear camera feed
        self.camera_label.configure(image='')
        
        messagebox.showinfo("Registration Complete", 
                          f"Successfully registered {employee_name} with {count} face samples.\n\n"
                          "Remember to train the model after registering employees.")
    
    def train_model(self):
        """Train the face recognition model"""
        if self.is_camera_running:
            return
            
        self.status_var.set("Training model...")
        
        # Reset authentication info
        self.auth_status_var.set("Training model...")
        self.auth_name_var.set("N/A")
        self.auth_confidence_var.set("N/A")
        
        # Disable UI during training
        for widget in self.root.winfo_children():
            if isinstance(widget, ttk.Frame):
                for child in widget.winfo_children():
                    if isinstance(child, ttk.Button):
                        child.configure(state='disabled')
        
        # Train in a separate thread to keep UI responsive
        def train_thread():
            success = self.face_system.train_model()
            
            # Re-enable UI
            for widget in self.root.winfo_children():
                if isinstance(widget, ttk.Frame):
                    for child in widget.winfo_children():
                        if isinstance(child, ttk.Button):
                            child.configure(state='normal')
            
            if success:
                self.status_var.set("Model trained successfully")
                self.auth_status_var.set("Ready for authentication")
                messagebox.showinfo("Training Complete", "Face recognition model trained successfully.")
            else:
                self.status_var.set("Training failed")
                self.auth_status_var.set("Training failed")
                messagebox.showerror("Training Failed", 
                                  "Failed to train the model. Make sure you've registered at least one employee.")
        
        train_thread = threading.Thread(target=train_thread)
        train_thread.daemon = True
        train_thread.start()
    
    def authenticate_employee(self):
        """Start authentication process"""
        if self.is_camera_running:
            return
            
        if not self.face_system.model_trained and not self.face_system.load_model():
            messagebox.showerror("Error", "No trained model found. Please train the model first.")
            return
            
        self.status_var.set("Starting authentication...")
        
        # Reset authentication info
        self.auth_status_var.set("Scanning...")
        self.auth_name_var.set("N/A")
        self.auth_confidence_var.set("N/A")
        self.auth_photo_label.configure(image='')
        
        # Start camera in separate thread for authentication
        self.camera_thread = threading.Thread(target=self.run_authentication)
        self.camera_thread.daemon = True
        self.camera_thread.start()
    
    def run_authentication(self):
        """Run the authentication process in a separate thread"""
        self.is_camera_running = True
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Failed to open camera")
            self.is_camera_running = False
            return
        
        start_time = cv2.getTickCount()
        timeout = 10  # seconds
        authenticated = False
        auth_result = None
        
        while self.is_camera_running:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Calculate elapsed time
            elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            if elapsed > timeout and not authenticated:
                break
            
            # Detect face
            gray, face = self.face_system.detect_face(frame)
            
            display_frame = frame.copy()
            
            # Display remaining time (using theme colors)
            remaining = max(0, timeout - elapsed)
            cv2.putText(display_frame, f"Time: {remaining:.1f}s", (10, display_frame.shape[0]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (70, 130, 180), 2)  # Dark blue in BGR
            
            if face is not None:
                x, y, w, h = face
                
                # Just draw rectangle around face without text
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (135, 206, 235), 2)  # Sky blue in BGR
                
                # Try to recognize the face
                face_img = gray[y:y+h, x:x+w]
                
                # Predict the face
                label, confidence = self.face_system.face_recognizer.predict(face_img)
                
                # Lower confidence value means better match in LBPH
                if confidence < self.face_system.face_recognizer.getThreshold():
                    employee_name = self.face_system.employee_ids.get(label, f"Unknown_{label}")
                    
                    # Update info panel instead of drawing on image
                    self.auth_status_var.set("Authenticated")
                    self.auth_name_var.set(employee_name)
                    self.auth_confidence_var.set(f"{100-confidence:.1f}%")
                    
                    # Display the detected face in the right panel
                    face_rgb = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
                    face_pil = Image.fromarray(face_rgb)
                    face_pil = face_pil.resize((150, 150), Image.LANCZOS)
                    face_photo = ImageTk.PhotoImage(image=face_pil)
                    self.auth_photo_label.configure(image=face_photo)
                    self.auth_photo_label.image = face_photo
                    
                    auth_result = {
                        "authenticated": True,
                        "employee_id": label,
                        "employee_name": employee_name,
                        "confidence": 100 - confidence
                    }
                    
                    # Mark as authenticated but keep showing camera feed for a moment
                    authenticated = True
                    start_time = cv2.getTickCount()  # Reset timer to show result for a moment
                    timeout = 2  # Show authenticated result for 2 seconds
                    
                else:
                    # Use light red for unrecognized face
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (95, 95, 255), 2)  # Light red in BGR
                    
                    # Update info panel
                    self.auth_status_var.set("Unrecognized")
                    self.auth_name_var.set("Unknown")
                    self.auth_confidence_var.set("N/A")
                    
                    # Display the unrecognized face
                    face_rgb = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
                    face_pil = Image.fromarray(face_rgb)
                    face_pil = face_pil.resize((150, 150), Image.LANCZOS)
                    face_photo = ImageTk.PhotoImage(image=face_pil)
                    self.auth_photo_label.configure(image=face_photo)
                    self.auth_photo_label.image = face_photo
            else:
                # Update info panel for no face
                self.auth_status_var.set("No face detected")
                self.auth_name_var.set("N/A")
                self.auth_confidence_var.set("N/A")
            
            # Display camera feed
            self.update_camera_feed(display_frame)
        
        cap.release()
        self.is_camera_running = False
        
        # Clear camera feed
        self.camera_label.configure(image='')
        
        # Show authentication result
        if authenticated and auth_result:
            self.status_var.set(f"Authentication successful: {auth_result['employee_name']}")
            messagebox.showinfo("Authentication Result", 
                             f"Authentication successful!\n\nEmployee: {auth_result['employee_name']}\n"
                             f"Confidence: {auth_result['confidence']:.2f}%")
        else:
            self.status_var.set("Authentication failed")
            self.auth_status_var.set("Authentication failed")
            messagebox.showinfo("Authentication Result", "Authentication failed. No recognized employee.")
    
    def update_camera_feed(self, frame):
        """Update the camera feed label with the current frame"""
        # Convert frame to format compatible with tkinter
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        
        # Resize to fit the window while maintaining aspect ratio
        width, height = self.camera_label.winfo_width(), self.camera_label.winfo_height()
        if width > 1 and height > 1:  # Ensure valid dimensions
            img_width, img_height = img.size
            ratio = min(width/img_width, height/img_height)
            new_size = (int(img_width * ratio), int(img_height * ratio))
            img = img.resize(new_size, Image.LANCZOS)
        
        photo = ImageTk.PhotoImage(image=img)
        self.camera_label.configure(image=photo)
        self.camera_label.image = photo  # Keep a reference
    
    def exit_app(self):
        """Exit the application"""
        if self.is_camera_running:
            self.is_camera_running = False
            if self.camera_thread and self.camera_thread.is_alive():
                self.camera_thread.join(1.0)  # Wait for thread to terminate
        
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    
    # Set window icon
    try:
        root.iconbitmap("face_icon.ico")  # Provide an icon file if available
    except:
        pass  # No icon available
    
    root.mainloop()