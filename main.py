from face_recognition_system import EmployeeFaceRecognitionSystem

def main():
    # Initialize the system
    face_system = EmployeeFaceRecognitionSystem()
    
    while True:
        print("\n=== Employee Face Recognition System ===")
        print("1. Register new employee")
        print("2. Train the model")
        print("3. Authenticate employee")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == '1':
            name = input("Enter employee name: ")
            print("Starting camera for face capture. Press ESC to cancel.")
            success = face_system.register_employee(name)
            if success:
                print(f"Successfully registered {name}")
            else:
                print("Registration failed")
                
        elif choice == '2':
            print("Training the model...")
            success = face_system.train_model()
            if success:
                print("Model trained successfully")
            else:
                print("Training failed")
                
        elif choice == '3':
            print("Starting authentication. Press ESC to cancel.")
            result = face_system.authenticate_employee()
            
            if result["authenticated"]:
                print(f"Authentication successful!")
                print(f"Employee: {result['employee_name']}")
                print(f"Confidence: {result['confidence']:.2f}%")
            else:
                print("Authentication failed")
                
        elif choice == '4':
            print("Exiting system. Goodbye!")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()