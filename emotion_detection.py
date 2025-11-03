import cv2
from deepface import DeepFace
import numpy as np
def detect_emotion_realtime():
    cap = cv2.VideoCapture(0)  
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return   
    print("Emotion Detection Started...")
    print("Press 'q' to quit")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')    
    frame_count = 0   
    while True:
        ret, frame = cap.read()        
        if not ret:
            print("Error: Could not read frame")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if frame_count % 5 == 0:
            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                
                try:
                    result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                    if isinstance(result, list):
                        emotion = result[0]['dominant_emotion']
                        emotion_scores = result[0]['emotion']
                    else:
                        emotion = result['dominant_emotion']
                        emotion_scores = result['emotion']
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, f"Emotion: {emotion}", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    y_offset = y + h + 20
                    for emo, score in emotion_scores.items():
                        text = f"{emo}: {score:.1f}%"
                        cv2.putText(frame, text, (x, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        y_offset += 20               
                except Exception as e:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(frame, "Processing...", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)       
        else:
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)       
        frame_count += 1
        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    print("Emotion detection stopped.")
def detect_emotion_from_image(image_path):
    try:
        img = cv2.imread(image_path)       
        if img is None:
            print(f"Error: Could not load image from {image_path}")
            return
        result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
        if isinstance(result, list):
            emotion = result[0]['dominant_emotion']
            emotion_scores = result[0]['emotion']
        else:
            emotion = result['dominant_emotion']
            emotional_scores = result['emotion']
        print(f"\nDominant Emotion: {emotion}")
        print("\nEmotion Scores:")
        for emo, score in emotion_scores.items():
            if img is None:
              print(f"Error: Could not load image from {image_path}")
            return
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))       
        print(f"Found {len(faces)} face(s)")       
        for i, (x, y, w, h) in enumerate(faces):
            face_roi = img[y:y+h, x:x+w]            
            try:
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)               
                if isinstance(result,list):
                    emotion = result[0]['dominant_emotion']
                else:
                    emotion = result['dominant_emotion']
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, f"Face {i+1}: {emotion}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)                
                print(f"\nFace {i+1}: {emotion}")                
            except Exception as e:
                print(f"Error analyzing face {i+1}: {e}")
        cv2.imshow('Multiple Emotion Detection', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()       
    except Exception as e:
        print(f"Error: {e}")
def detect_emotion_with_fer():
    from fer import FER
    emotion_detector = FER()
    cap = cv2.VideoCapture(0)    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return    
    print("FER Emotion Detection Started...")
    print("Press 'q' to quit")   
    while True:
        ret, frame = cap.read()        
        if not ret:
            break
        result = emotion_detector.detect_emotions(frame)
        for face in result:
            (x, y, w, h) = face["box"]
            emotions = face["emotions"]
            dominant_emotion = max(emotions, key=emotions.get)
            confidence = emotions[dominant_emotion]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            label = f"{dominant_emotion}: {confidence:.2f}"
            cv2.putText(frame, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            y_offset = y + h + 20
            for emotion, score in emotions.items():
                text = f"{emotion}: {score:.2f}"
                cv2.putText(frame, text, (x, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20        
        cv2.imshow('FER Emotion Detection', frame)        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break    
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    print("=== EMOTION DETECTION PROGRAM ===")
    print("\nChoose an option:")
    print("1. Real-time emotion detection (DeepFace)")
    print("2. Detect emotion from image file")
    print("3. Detect multiple emotions in image")
    print("4. Real-time emotion detection (FER - lighter)")   
    choice = input("\nEnter your choice (1-4): ")   
    if choice == "1":
        detect_emotion_realtime()    
    elif choice == "2":
        image_path = input("Enter image path: ")
        detect_emotion_from_image(image_path)    
    elif choice == "3":
        image_path = input("Enter image path: ")
        detect_multiple_emotions(image_path)    
    elif choice == "4":
        detect_emotion_with_fer()   
    else:
        print("Invalid choice!")
