import cv2
import numpy as np
import os
from datetime import datetime
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(filename='face_recognition.log', level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('face_recognition')

class EmployeeFaceRecognitionSystem:
    def __init__(self, data_path="employee_faces"):
        """
        Initialize the Employee Face Recognition System
        
        Args:
            data_path: Path to the directory containing employee face images
        """
        self.data_path = data_path
        
        # Load the Haar cascade for face detection
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize LBPH Face Recognizer (as specified in abstract)
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Parameters to address lighting variations and partial occlusions
        self.face_recognizer.setRadius(2)  # Increased radius for better handling of lighting
        self.face_recognizer.setNeighbors(8)  # Standard 8 neighbors for LBPH
        self.face_recognizer.setGridX(8)  # More cells for better detail capture
        self.face_recognizer.setGridY(8)  # More cells for better detail capture
        self.face_recognizer.setThreshold(80)  # Threshold for recognition confidence
        
        self.employee_ids = {}  # Map from ID to employee name
        self.employee_names = set()  # Set of registered employee names
        self.id_counter = 0
        self.model_trained = False
        
        # Performance metrics
        self.accuracy = 0
        self.error_rate = 0
        self.recognition_times = []
        
        # Create directory if it doesn't exist
        if not os.path.exists(data_path):
            os.makedirs(data_path)
            os.makedirs(os.path.join(data_path, 'training'))
    
    def detect_face(self, img):
        """
        Detect faces in an image using Haar cascades
        
        Args:
            img: Input image
            
        Returns:
            gray_img: Grayscale image
            face: Detected face region
        """
        # Convert to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization to handle varying lighting conditions
        gray_img = cv2.equalizeHist(gray_img)
        
        # Detect faces
        faces = self.face_detector.detectMultiScale(
            gray_img,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) == 0:
            return None, None
        
        # Get the largest face in case multiple faces are detected
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face
        
        # Return the grayscale image and face region
        return gray_img, largest_face
    
    def check_face_similarity(self, face_img, similarity_threshold=60):
        """
        Check if the face is similar to any existing registered face
        
        Args:
            face_img: Face image to check
            similarity_threshold: Threshold for face similarity (lower is more strict)
            
        Returns:
            is_duplicate: Boolean indicating if the face already exists
            duplicate_name: Name of the matching employee if found
        """
        if not self.model_trained:
            if not self.load_model():
                return False, None
        
        # Predict the face
        label, confidence = self.face_recognizer.predict(face_img)
        
        # Check if the face is similar enough to an existing face
        if confidence < similarity_threshold:  # Lower confidence means better match
            duplicate_name = self.employee_ids.get(label, None)
            return True, duplicate_name
        
        return False, None
    
    def register_employee(self, name, camera_id=0, num_samples=20):
        """
        Register a new employee by capturing face samples
        
        Args:
            name: Employee name
            camera_id: Camera device ID
            num_samples: Number of face samples to collect
        
        Returns:
            result: Dictionary with registration result
        """
        try:
            # Check if the name already exists
            if name in self.employee_names:
                logger.warning(f"Employee with name '{name}' already exists")
                return {
                    "success": False, 
                    "error": "duplicate_name", 
                    "message": f"Employee with name '{name}' is already registered"
                }
            
            # Start video capture
            cap = cv2.VideoCapture(camera_id)
            if not cap.isOpened():
                logger.error(f"Failed to open camera with ID {camera_id}")
                return {
                    "success": False, 
                    "error": "camera_error", 
                    "message": "Failed to access camera"
                }
            
            # First capture a single face to check for duplicates
            logger.info(f"Capturing initial face sample for duplication check")
            
            # Capture initial face for duplicate check
            face_img = None
            duplicate_check_timeout = 5  # seconds
            start_time = datetime.now()
            
            while face_img is None and (datetime.now() - start_time).total_seconds() < duplicate_check_timeout:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                gray, face = self.detect_face(frame)
                if face is not None:
                    x, y, w, h = face
                    face_img = gray[y:y+h, x:x+w]
                    
                    # Display the face being captured
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, "Checking face...", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                cv2.imshow("Face Registration", frame)
                key = cv2.waitKey(100)
                if key == 27:  # ESC key
                    cap.release()
                    cv2.destroyAllWindows()
                    return {
                        "success": False, 
                        "error": "user_canceled", 
                        "message": "Registration canceled by user"
                    }
            
            # If no face was detected during the initial check
            if face_img is None:
                cap.release()
                cv2.destroyAllWindows()
                return {
                    "success": False, 
                    "error": "no_face_detected", 
                    "message": "No face detected for duplicate check"
                }
            
            # Check if this face already exists in the system
            if self.model_trained:
                is_duplicate, duplicate_name = self.check_face_similarity(face_img)
                if is_duplicate:
                    cap.release()
                    cv2.destroyAllWindows()
                    logger.warning(f"Face already registered under name '{duplicate_name}'")
                    return {
                        "success": False, 
                        "error": "duplicate_face", 
                        "message": f"This face is already registered under name '{duplicate_name}'"
                    }
            
            # Face is unique, proceed with registration
            employee_id = self.id_counter
            self.employee_ids[employee_id] = name
            self.employee_names.add(name)
            self.id_counter += 1
            
            # Create directory for this employee if it doesn't exist
            employee_dir = os.path.join(self.data_path, 'training', str(employee_id))
            if not os.path.exists(employee_dir):
                os.makedirs(employee_dir)
            
            # Save the first captured face
            face_filename = os.path.join(employee_dir, "0.jpg")
            cv2.imwrite(face_filename, face_img)
            count = 1
            
            logger.info(f"Starting face capture for employee {name}")
            
            while count < num_samples:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to capture frame from camera")
                    break
                
                gray, face = self.detect_face(frame)
                if face is not None:
                    x, y, w, h = face
                    
                    # Draw rectangle around the face
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Save the face image
                    face_img = gray[y:y+h, x:x+w]
                    face_filename = os.path.join(employee_dir, f"{count}.jpg")
                    cv2.imwrite(face_filename, face_img)
                    
                    count += 1
                    cv2.putText(frame, f"Captured: {count}/{num_samples}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                cv2.imshow("Face Registration", frame)
                
                # Wait for 100ms or key press
                key = cv2.waitKey(100)
                if key == 27:  # ESC key
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            
            # Save the employee mapping
            self.save_employee_mapping()
            
            # Retrain the model with the new face
            self.train_model()
            
            logger.info(f"Successfully registered employee {name} with ID {employee_id}")
            return {
                "success": True,
                "employee_id": employee_id,
                "employee_name": name,
                "samples_captured": count,
                "message": f"Successfully registered {name} with {count} face samples"
            }
            
        except Exception as e:
            logger.error(f"Error registering employee: {str(e)}")
            return {
                "success": False, 
                "error": "registration_error", 
                "message": f"Error during registration: {str(e)}"
            }
    
    def prepare_training_data(self):
        """
        Prepare training data from the stored face images
        
        Returns:
            faces: List of face images
            labels: List of corresponding employee IDs
            names: List of employee names
        """
        faces = []
        labels = []
        names = []
        
        # Navigate through the training directory
        training_dir = os.path.join(self.data_path, 'training')
        for employee_id in os.listdir(training_dir):
            # Skip non-directory files
            if not os.path.isdir(os.path.join(training_dir, employee_id)):
                continue
            
            # Get the name if available
            if int(employee_id) in self.employee_ids:
                name = self.employee_ids[int(employee_id)]
                # Add to the set of employee names
                self.employee_names.add(name)
            else:
                name = f"Unknown_{employee_id}"
                self.employee_ids[int(employee_id)] = name
                self.employee_names.add(name)
            
            # Get all face images for this employee
            employee_dir = os.path.join(training_dir, employee_id)
            for img_file in os.listdir(employee_dir):
                if not img_file.endswith('.jpg'):
                    continue
                
                img_path = os.path.join(employee_dir, img_file)
                # Read the image
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                # Add to training data
                faces.append(img)
                labels.append(int(employee_id))
                names.append(name)
        
        return faces, labels, names
    
    def train_model(self):
        """
        Train the LBPH face recognizer model
        
        Returns:
            success: Boolean indicating if training was successful
        """
        try:
            logger.info("Starting model training")
            
            # Prepare training data
            faces, labels, _ = self.prepare_training_data()
            
            if len(faces) == 0 or len(labels) == 0:
                logger.error("No training data available")
                return False
            
            # Train the recognizer
            self.face_recognizer.train(faces, np.array(labels))
            
            # Save the model
            self.face_recognizer.save(os.path.join(self.data_path, 'employee_model.yml'))
            
            self.model_trained = True
            logger.info(f"Model trained successfully with {len(faces)} face images")
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return False
    
    def load_model(self):
        """
        Load a previously trained model
        
        Returns:
            success: Boolean indicating if loading was successful
        """
        model_path = os.path.join(self.data_path, 'employee_model.yml')
        if os.path.exists(model_path):
            try:
                self.face_recognizer.read(model_path)
                self.model_trained = True
                
                # Load employee IDs and populate employee_names set
                employee_mapping_path = os.path.join(self.data_path, 'employee_mapping.txt')
                if os.path.exists(employee_mapping_path):
                    with open(employee_mapping_path, 'r') as f:
                        for line in f:
                            if line.strip():
                                id_val, name = line.strip().split(',', 1)
                                self.employee_ids[int(id_val)] = name
                                self.employee_names.add(name)
                                # Update id_counter to be larger than any existing ID
                                self.id_counter = max(self.id_counter, int(id_val) + 1)
                
                logger.info("Model loaded successfully")
                return True
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                return False
        else:
            logger.error("Model file not found")
            return False
    
    def authenticate_employee(self, camera_id=0, timeout=10):
        """
        Authenticate an employee using face recognition
        
        Args:
            camera_id: Camera device ID
            timeout: Timeout in seconds
        
        Returns:
            result: Dictionary with authentication result
        """
        if not self.model_trained:
            if not self.load_model():
                logger.error("Model not trained and couldn't load a saved model")
                return {"authenticated": False, "error": "Model not trained"}
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            logger.error(f"Failed to open camera with ID {camera_id}")
            return {"authenticated": False, "error": "Camera not available"}
        
        start_time = datetime.now()
        timeout_delta = timeout
        
        result = {"authenticated": False, "employee_id": None, "employee_name": None, "confidence": None}
        
        while (datetime.now() - start_time).total_seconds() < timeout_delta:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Detect face
            gray, face = self.detect_face(frame)
            if face is None:
                cv2.putText(frame, "No face detected", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                x, y, w, h = face
                
                # Predict the face
                face_img = gray[y:y+h, x:x+w]
                
                prediction_start = datetime.now()
                label, confidence = self.face_recognizer.predict(face_img)
                prediction_time = (datetime.now() - prediction_start).total_seconds() * 1000  # ms
                self.recognition_times.append(prediction_time)
                
                # Lower confidence value means better match in LBPH
                if confidence < self.face_recognizer.getThreshold():
                    employee_name = self.employee_ids.get(label, f"Unknown_{label}")
                    
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, f"{employee_name} ({100-confidence:.1f}%)", 
                                (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    result = {
                        "authenticated": True,
                        "employee_id": label,
                        "employee_name": employee_name,
                        "confidence": 100 - confidence,  # Convert to a more intuitive percentage
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "processing_time_ms": prediction_time
                    }
                    
                    # Log successful authentication
                    logger.info(f"Employee authenticated: {employee_name} (ID: {label}) with confidence {100-confidence:.1f}%")
                    
                else:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(frame, "Unknown", (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Display remaining time
            remaining = timeout_delta - (datetime.now() - start_time).total_seconds()
            cv2.putText(frame, f"Time: {remaining:.1f}s", (10, frame.shape[0]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow("Employee Authentication", frame)
            
            # If authenticated, break after showing for a second
            if result["authenticated"]:
                cv2.waitKey(1000)
                break
            
            # Check for key press
            key = cv2.waitKey(1)
            if key == 27:  # ESC key
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        return result
    
    def evaluate_model(self, test_data_path=None):
        """
        Evaluate the model's performance
        
        Args:
            test_data_path: Path to test data, if None uses a portion of training data
        
        Returns:
            metrics: Dictionary with performance metrics
        """
        if not self.model_trained:
            if not self.load_model():
                logger.error("Model not trained and couldn't load a saved model")
                return {"error": "Model not trained"}
        
        # If no specific test data, use some of the training data
        if test_data_path is None:
            test_data_path = os.path.join(self.data_path, 'training')
        
        true_labels = []
        predicted_labels = []
        confidences = []
        processing_times = []
        
        logger.info("Starting model evaluation")
        
        for employee_id in os.listdir(test_data_path):
            # Skip non-directory files
            if not os.path.isdir(os.path.join(test_data_path, employee_id)):
                continue
            
            employee_dir = os.path.join(test_data_path, employee_id)
            for img_file in os.listdir(employee_dir):
                if not img_file.endswith('.jpg'):
                    continue
                
                img_path = os.path.join(employee_dir, img_file)
                img = cv2.imread(img_path)
                
                # Detect and predict
                gray, face = self.detect_face(img)
                if face is not None:
                    x, y, w, h = face
                    face_img = gray[y:y+h, x:x+w]
                    
                    start_time = datetime.now()
                    label, confidence = self.face_recognizer.predict(face_img)
                    process_time = (datetime.now() - start_time).total_seconds() * 1000
                    
                    true_labels.append(int(employee_id))
                    predicted_labels.append(label)
                    confidences.append(confidence)
                    processing_times.append(process_time)
        
        if not true_labels:
            logger.error("No test samples were evaluated")
            return {"error": "No test samples"}
        
        # Calculate metrics
        accuracy = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == pred) / len(true_labels)
        self.accuracy = accuracy
        self.error_rate = 1 - accuracy
        
        # Generate confusion matrix
        conf_matrix = confusion_matrix(true_labels, predicted_labels)
        
        # Generate classification report
        class_report = classification_report(true_labels, predicted_labels, output_dict=True)
        
        # Performance timing
        avg_time = sum(processing_times) / len(processing_times)
        
        metrics = {
            "accuracy": accuracy * 100,
            "error_rate": self.error_rate * 100,
            "avg_processing_time_ms": avg_time,
            "confusion_matrix": conf_matrix,
            "classification_report": class_report,
            "num_test_samples": len(true_labels)
        }
        
        logger.info(f"Model evaluation completed: Accuracy={accuracy*100:.2f}%, Error rate={self.error_rate*100:.2f}%")
        
        return metrics
    
    def generate_performance_report(self, metrics=None):
        """
        Generate a performance report with visualizations
        
        Args:
            metrics: Dictionary with performance metrics, if None will run evaluation
            
        Returns:
            report: Dictionary with report data and chart paths
        """
        if metrics is None:
            metrics = self.evaluate_model()
        
        if "error" in metrics:
            return {"error": metrics["error"]}
        
        # Create directory for reports
        report_dir = os.path.join(self.data_path, 'reports')
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Confusion matrix visualization
        plt.figure(figsize=(10, 8))
        plt.imshow(metrics["confusion_matrix"], interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        classes = list(self.employee_ids.values())
        tick_marks = range(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        conf_matrix_path = os.path.join(report_dir, f'confusion_matrix_{timestamp}.png')
        plt.tight_layout()
        plt.savefig(conf_matrix_path)
        plt.close()
        
        # Performance metrics visualization
        plt.figure(figsize=(12, 6))
        
        # Accuracy and error rate
        plt.subplot(1, 2, 1)
        plt.bar(['Accuracy', 'Error Rate'], [metrics["accuracy"], metrics["error_rate"]])
        plt.title('Recognition Performance')
        plt.ylabel('Percentage (%)')
        
        # Processing time
        plt.subplot(1, 2, 2)
        plt.hist(self.recognition_times, bins=20)
        plt.title('Recognition Processing Time')
        plt.xlabel('Time (ms)')
        plt.ylabel('Frequency')
        
        performance_path = os.path.join(report_dir, f'performance_{timestamp}.png')
        plt.tight_layout()
        plt.savefig(performance_path)
        plt.close()
        
        # Generate summary report
        report = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "accuracy": f"{metrics['accuracy']:.2f}%",
            "error_rate": f"{metrics['error_rate']:.2f}%",
            "avg_processing_time": f"{metrics['avg_processing_time_ms']:.2f} ms",
            "total_employees": len(self.employee_ids),
            "test_samples": metrics['num_test_samples'],
            "confusion_matrix_path": conf_matrix_path,
            "performance_chart_path": performance_path
        }
        
        # Save report to CSV
        report_df = pd.DataFrame([report])
        report_csv = os.path.join(report_dir, f'performance_report_{timestamp}.csv')
        report_df.to_csv(report_csv, index=False)
        
        return report
    
    def save_employee_mapping(self):
        """Save the employee ID to name mapping"""
        mapping_path = os.path.join(self.data_path, 'employee_mapping.txt')
        with open(mapping_path, 'w') as f:
            for emp_id, name in self.employee_ids.items():
                f.write(f"{emp_id},{name}\n")
                
    def remove_employee(self, name=None, employee_id=None):
        """
        Remove an employee from the system
        
        Args:
            name: Employee name to remove
            employee_id: Employee ID to remove
            
        Returns:
            success: Boolean indicating if removal was successful
        """
        try:
            # Find the employee ID if name is provided
            if name is not None and employee_id is None:
                for id_val, emp_name in self.employee_ids.items():
                    if emp_name == name:
                        employee_id = id_val
                        break
                
                if employee_id is None:
                    logger.error(f"Employee with name '{name}' not found")
                    return False
            
            # Check if employee ID exists
            if employee_id not in self.employee_ids:
                logger.error(f"Employee with ID {employee_id} not found")
                return False
            
            # Get the name for logging
            name = self.employee_ids[employee_id]
            
            # Remove the employee directory
            employee_dir = os.path.join(self.data_path, 'training', str(employee_id))
            if os.path.exists(employee_dir):
                for file in os.listdir(employee_dir):
                    os.remove(os.path.join(employee_dir, file))
                os.rmdir(employee_dir)
            
            # Remove from employee_ids and employee_names
            self.employee_names.remove(name)
            del self.employee_ids[employee_id]
            
            # Save the updated mapping
            self.save_employee_mapping()
            
            # Retrain the model if there are still employees
            if self.employee_ids:
                self.train_model()
            
            logger.info(f"Successfully removed employee {name} (ID: {employee_id})")
            return True
            
        except Exception as e:
            logger.error(f"Error removing employee: {str(e)}")
            return False