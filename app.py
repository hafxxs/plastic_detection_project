import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

# --------------------------
# 1. Preprocess Images
# --------------------------
def preprocess_images(input_dir, img_size=(224, 224)):
    processed_dir = "processed_dataset"
    os.makedirs(processed_dir, exist_ok=True)
    for category in os.listdir(input_dir):
        category_path = os.path.join(input_dir, category)
        out_category_path = os.path.join(processed_dir, category)
        os.makedirs(out_category_path, exist_ok=True)
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, img_size)
                cv2.imwrite(os.path.join(out_category_path, img_name), img)
    return processed_dir

# --------------------------
# 2. Train Model
# --------------------------
def train_model(processed_dir):
    datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)
    train_gen = datagen.flow_from_directory(
    processed_dir,
    target_size=(224,224),
    batch_size=32,
    subset='training',
    class_mode='binary'   # <-- add this
)
    val_gen = datagen.flow_from_directory(
    processed_dir,
    target_size=(224,224),
    batch_size=32,
    subset='validation',
    class_mode='binary'   # <-- add this
)

    
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_gen, validation_data=val_gen, epochs=5)
    model.save("plastic_detector.h5")
    return model

# --------------------------
# 3. Predict Single Image
# --------------------------
def predict_image(model, img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224,224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)[0][0]
    print(f"Plastic Probability: {pred:.2f}")
    if pred > 0.5:
        print("Plastic Detected!")
    else:
        print("No Plastic Detected.")

# --------------------------
# 4. Real-Time Camera Detection
# --------------------------
def realtime_detection(model):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot access camera")
        return

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        img = cv2.resize(frame, (224,224))
        img_input = img / 255.0
        img_input = np.expand_dims(img_input, axis=0)
        pred = model.predict(img_input)[0][0]

        # Draw prediction on frame
        label = "Plastic" if pred > 0.5 else "No Plastic"
        color = (0,0,255) if pred > 0.5 else (0,255,0)
        cv2.putText(frame, f"{label}: {pred:.2f}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        cv2.imshow("Plastic Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# --------------------------
# 5. Main Program
# --------------------------
if __name__ == "__main__":
    dataset_dir = "dataset"  # Your raw phone images
    processed_dir = preprocess_images(dataset_dir)
    
    # Train or load existing model
    if os.path.exists("plastic_detector.h5"):
        model = load_model("plastic_detector.h5")
        print("Loaded existing model.")
    else:
        model = train_model(processed_dir)
    
    choice = input("Enter 1 to test single image, 2 for real-time camera: ")
    if choice == "1":
        test_image = input("Enter image filename (e.g., test.jpg): ")
        predict_image(model, test_image)
    elif choice == "2":
        realtime_detection(model)
    else:
        print("Invalid choice.")
