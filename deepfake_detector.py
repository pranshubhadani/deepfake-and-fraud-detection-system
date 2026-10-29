import os
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models

# Default configurations and paths
MODEL_FILE = 'deepfake_cnn_model.pth'
DATASET_DIR = os.path.join('Datasets', 'Deepfake Datasets')
IMAGE_SIZE = (224, 224)

# ==========================================
# 1. CNN Model Architecture
# ==========================================
class DeepfakeCNN(nn.Module):
    """
    Convolutional Neural Network tailored to detect boundary blending,
    synthetic pixel warping, and facial manipulation artifacts.
    Supports either a custom 4-stage convolutional backbone or ResNet18 transfer learning.
    """
    def __init__(self, backbone='custom', pretrained=True):
        super(DeepfakeCNN, self).__init__()
        self.backbone_type = backbone

        if backbone == 'resnet18':
            # ResNet-18 Transfer Learning backbone
            try:
                weights = models.ResNet18_Weights.DEFAULT if pretrained else None
                base_model = models.resnet18(weights=weights)
            except Exception:
                base_model = models.resnet18(pretrained=pretrained)
                
            num_features = base_model.fc.in_features
            base_model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, 2)  # [Real, Fake]
            )
            self.model = base_model
        else:
            # Custom 4-stage Deepfake Artifact CNN
            self.features = nn.Sequential(
                # Stage 1: Initial edge and texture detection
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.1, inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),  # -> 112x112

                # Stage 2: Facial boundary & gradient transitions
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.1, inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),  # -> 56x56

                # Stage 3: Compression & synthesis artifact patterns
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.1, inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),  # -> 28x28

                # Stage 4: High-level semantic & lighting incongruities
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.1, inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),  # -> 14x14
            )
            self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.5),
                nn.Linear(256 * 4 * 4, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, 2)  # Class 0 = Real, Class 1 = Fake
            )

    def forward(self, x):
        if self.backbone_type == 'resnet18':
            return self.model(x)
        else:
            x = self.features(x)
            x = self.avgpool(x)
            return self.classifier(x)


# ==========================================
# 2. Face Detection & Bounding Cropper
# ==========================================
def get_face_detector():
    """Loads OpenCV's Haar Cascade Face Detector."""
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    return cv2.CascadeClassifier(cascade_path)

def crop_face(frame_bgr, margin=0.25, target_size=IMAGE_SIZE):
    """
    Detects human facial landmarks and crops facial regions with boundary margins
    suitable for deep learning input.
    
    The margin ensures facial perimeter boundaries (hairline, jawline, ears)
    where face-swap seams usually occur are retained in the crop.
    
    Returns:
        cropped_rgb: PIL.Image of the face (or centered crop if no face detected)
        face_detected: bool indicating whether an actual face was located
        bbox: (x, y, w, h) coordinates
    """
    detector = get_face_detector()
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )
    
    h_img, w_img, _ = frame_bgr.shape
    
    if len(faces) > 0:
        # Pick the largest detected face
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]
        
        # Add boundary margin for artifact inspection
        margin_x = int(w * margin)
        margin_y = int(h * margin)
        
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(w_img, x + w + margin_x)
        y2 = min(h_img, y + h + margin_y)
        
        face_crop = frame_bgr[y1:y2, x1:x2]
        face_detected = True
        bbox = (x, y, w, h)
    else:
        # Fallback: Center crop if no face recognized
        crop_size = min(h_img, w_img)
        start_y = (h_img - crop_size) // 2
        start_x = (w_img - crop_size) // 2
        face_crop = frame_bgr[start_y:start_y + crop_size, start_x:start_x + crop_size]
        face_detected = False
        bbox = None
        
    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(face_rgb).resize(target_size, Image.Resampling.BILINEAR)
    return pil_image, face_detected, bbox


# ==========================================
# 3. OpenCV Keyframe Sampling
# ==========================================
def extract_video_keyframes(video_path, frame_interval=15, max_frames=80):
    """
    Grabs representative video frames at regular intervals (e.g. every 15th frame)
    rather than processing full uncompressed video stream, conserving compute resources.
    
    Yields:
        frame_idx (int): Frame index in the video
        timestamp_sec (float): Timestamp in seconds
        frame (np.ndarray): OpenCV BGR frame
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # default fallback
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0
    sampled_count = 0
    
    while cap.isOpened() and sampled_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        if current_frame % frame_interval == 0:
            timestamp_sec = round(current_frame / fps, 2)
            yield current_frame, timestamp_sec, frame
            sampled_count += 1
            
        current_frame += 1
        
    cap.release()


# ==========================================
# 4. Inference & Verification
# ==========================================
# Standard normalization transform for PyTorch vision models
inference_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_or_init_model(model_path=MODEL_FILE, backbone='custom', device=None):
    """
    Loads saved model weights if available, or initializes a new model.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    model = DeepfakeCNN(backbone=backbone)
    
    if os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"Loaded trained Deepfake CNN weights from '{model_path}'.")
        except Exception as e:
            print(f"Warning: Failed to load '{model_path}': {e}. Using initialized model.")
    else:
        print(f"No trained weights found at '{model_path}'. Running with initialized CNN.")
        print("Tip: Train the model using train_model() on your dataset to get optimal accuracy.")
        
    model.to(device)
    model.eval()
    return model, device

def analyze_image(image_path_or_array, model, device=None):
    """
    Detects face, crops it, runs CNN inference, and evaluates manipulation probability.
    
    Returns:
        dict: {
            'is_fake': bool,
            'fake_confidence': float (0-100),
            'real_confidence': float (0-100),
            'face_detected': bool,
            'verdict': str
        }
    """
    if device is None:
        device = next(model.parameters()).device
        
    if isinstance(image_path_or_array, str):
        if not os.path.exists(image_path_or_array):
            raise FileNotFoundError(f"Image not found: {image_path_or_array}")
        frame = cv2.imread(image_path_or_array)
        if frame is None:
            raise ValueError(f"Could not read image file: {image_path_or_array}")
    else:
        frame = image_path_or_array
        
    face_pil, face_detected, _ = crop_face(frame)
    tensor = inference_transform(face_pil).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = F.softmax(outputs, dim=1).cpu().numpy()[0]
        
    real_prob = float(probabilities[0]) * 100
    fake_prob = float(probabilities[1]) * 100
    is_fake = fake_prob >= 50.0
    
    if fake_prob >= 75.0:
        verdict = "High Likelihood Deepfake / Manipulated"
    elif fake_prob >= 50.0:
        verdict = "Suspicious Boundary Artifacts Detected"
    else:
        verdict = "Authentic / Real Face"
        
    return {
        'is_fake': is_fake,
        'fake_confidence': round(fake_prob, 2),
        'real_confidence': round(real_prob, 2),
        'face_detected': face_detected,
        'verdict': verdict
    }

def analyze_video(video_path, model, device=None, frame_interval=15):
    """
    Analyzes a video using OpenCV Keyframe Sampling + Deepfake CNN Artifact Scoring.
    
    Returns:
        dict: Summary statistics including frame-by-frame scores and overall verdict.
    """
    if device is None:
        device = next(model.parameters()).device
        
    frame_scores = []
    frames_inspected = 0
    faces_detected_count = 0
    fake_flagged_count = 0
    
    print(f"\nScanning video: {os.path.basename(video_path)} (Sampling every {frame_interval} frames)...")
    
    for frame_idx, timestamp, frame in extract_video_keyframes(video_path, frame_interval=frame_interval):
        res = analyze_image(frame, model, device=device)
        frames_inspected += 1
        
        if res['face_detected']:
            faces_detected_count += 1
            
        if res['is_fake']:
            fake_flagged_count += 1
            
        frame_scores.append({
            'frame': frame_idx,
            'timestamp': timestamp,
            'fake_score': res['fake_confidence'],
            'face_detected': res['face_detected']
        })
        
    if frames_inspected == 0:
        return {
            'status': 'Error',
            'error': 'No readable frames found in the video.'
        }
        
    fake_scores = [f['fake_score'] for f in frame_scores]
    avg_fake_score = round(sum(fake_scores) / len(fake_scores), 2)
    peak_fake_score = round(max(fake_scores), 2)
    fake_frame_ratio = round((fake_flagged_count / frames_inspected) * 100, 2)
    
    # Verdict logic
    if avg_fake_score >= 60.0 or fake_frame_ratio >= 40.0:
        final_verdict = "Likely Deepfake Video"
        risk_level = "High"
    elif avg_fake_score >= 40.0 or fake_frame_ratio >= 20.0:
        final_verdict = "Suspicious Video (Anomalous Face Artifacts)"
        risk_level = "Medium"
    else:
        final_verdict = "Authentic Video"
        risk_level = "Low"
        
    return {
        'status': 'Success',
        'video_file': os.path.basename(video_path),
        'frames_analyzed': frames_inspected,
        'faces_detected_count': faces_detected_count,
        'fake_frames_flagged': fake_flagged_count,
        'fake_frame_percentage': fake_frame_ratio,
        'avg_fake_score': avg_fake_score,
        'peak_fake_score': peak_fake_score,
        'risk_level': risk_level,
        'verdict': final_verdict,
        'frame_details': frame_scores
    }


# ==========================================
# 5. Training Pipeline
# ==========================================
def train_deepfake_model(dataset_dir=DATASET_DIR,
                         epochs=10,
                         batch_size=32,
                         learning_rate=1e-4,
                         backbone='custom',
                         save_path=MODEL_FILE):
    """
    Trains the CNN model on the dataset and saves the weights to disk.
    
    Expected folder structure:
        dataset_dir/
            train/
                real/
                fake/
            val/ (optional, will split from train if absent)
                real/
                fake/
    """
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader, random_split
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"🚀 GPU Acceleration Enabled: {gpu_name} ({vram_gb:.1f} GB VRAM)")
    else:
        print(f"Training on device: {device} (Tip: Install PyTorch with CUDA for faster RTX 3050 training)")
    
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'val')
    if not os.path.exists(val_dir) or len(os.listdir(val_dir)) == 0:
        alt_val = os.path.join(dataset_dir, 'valid')
        if os.path.exists(alt_val) and len(os.listdir(alt_val)) > 0:
            val_dir = alt_val
            
    if not os.path.exists(train_dir):
        print(f"Error: Training directory not found at '{train_dir}'.")
        print("Please structure your dataset into: Datasets/Deepfake Datasets/train/real and /fake")
        return None
        
    # Data Augmentation & Normalization
    train_transforms = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    full_dataset = ImageFolder(root=train_dir, transform=train_transforms)
    class_names = full_dataset.classes
    print(f"Detected classes: {class_names} (Total images: {len(full_dataset)})")
    
    if os.path.exists(val_dir) and len(os.listdir(val_dir)) > 0:
        val_dataset = ImageFolder(root=val_dir, transform=val_transforms)
        train_data = full_dataset
        val_data = val_dataset
    else:
        # Split 80% train, 20% val
        val_size = int(len(full_dataset) * 0.2)
        train_size = len(full_dataset) - val_size
        train_data, val_data = random_split(full_dataset, [train_size, val_size])
        print(f"No separate val folder found. Splitting: {train_size} train, {val_size} validation.")
        
    use_pin = (device.type == 'cuda')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=use_pin)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, pin_memory=use_pin)
    
    # Initialize Model, Loss, Optimizer, and Scaler (AMP)
    model = DeepfakeCNN(backbone=backbone).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    
    best_val_acc = 0.0
    
    print("\nStarting Training...")
    print("=" * 60)
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader, 1):
            images, labels = images.to(device, non_blocking=use_pin), labels.to(device, non_blocking=use_pin)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(images)
                loss = criterion(outputs, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            if batch_idx % 100 == 0 or batch_idx == len(train_loader):
                batch_loss = running_loss / total_train if total_train > 0 else 0
                batch_acc = (correct_train / total_train * 100) if total_train > 0 else 0
                print(f"  ⚡ Epoch [{epoch}/{epochs}] Batch [{batch_idx}/{len(train_loader)}] - Loss: {batch_loss:.4f} | Acc: {batch_acc:.1f}%", end='\r')
                
        print()  # Move to new line after training batches
        epoch_loss = running_loss / total_train if total_train > 0 else 0
        train_acc = (correct_train / total_train * 100) if total_train > 0 else 0
        
        # Validation phase
        model.eval()
        correct_val = 0
        total_val = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        val_acc = (correct_val / total_val * 100) if total_val > 0 else 0
        scheduler.step()
        
        print(f"Epoch [{epoch}/{epochs}] - "
              f"Train Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Acc: {val_acc:.2f}%")
              
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  --> Saved new best model checkpoint to {save_path} (Acc: {val_acc:.2f}%)")
            
    print("=" * 60)
    print(f"Training Complete! Best Validation Accuracy: {best_val_acc:.2f}%. Model saved at: '{save_path}'")
    return model


# ==========================================
# 6. Interactive CLI Test Runner
# ==========================================
if __name__ == '__main__':
    print("=" * 60)
    print(" 👁️ DEEPFAKE CNN ARTIFACT DETECTION SYSTEM ")
    print("=" * 60)
    
    model, device = load_or_init_model()
    
    while True:
        print("\nChoose an option:")
        print(" 1. Analyze an Image (.jpg, .png)")
        print(" 2. Analyze a Video (.mp4, .avi, .mov) with Keyframe Sampling")
        print(" 3. Train / Retrain CNN Model on Dataset")
        print(" 4. Exit")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            img_path = input("Enter image path: ").strip().strip('"').strip("'")
            if not os.path.exists(img_path):
                print("File not found! Please check the path.")
                continue
            try:
                res = analyze_image(img_path, model, device)
                print("\n--- Image Analysis Results ---")
                print(f"Verdict: {res['verdict']}")
                print(f"Fake Probability: {res['fake_confidence']}%")
                print(f"Real Probability: {res['real_confidence']}%")
                print(f"Face Detected: {'Yes' if res['face_detected'] else 'No (analyzed center region)'}")
            except Exception as e:
                print(f"Error analyzing image: {e}")
                
        elif choice == '2':
            vid_path = input("Enter video path: ").strip().strip('"').strip("'")
            if not os.path.exists(vid_path):
                print("File not found! Please check the path.")
                continue
            try:
                res = analyze_video(vid_path, model, device, frame_interval=15)
                print("\n" + "=" * 45)
                print(f" 🎬 Video Verification: {res['video_file']}")
                print("=" * 45)
                print(f"Verdict:              {res['verdict']}")
                print(f"Risk Level:           {res['risk_level']}")
                print(f"Average Fake Score:   {res['avg_fake_score']}%")
                print(f"Peak Fake Score:      {res['peak_fake_score']}%")
                print(f"Frames Analyzed:      {res['frames_analyzed']}")
                print(f"Fake Frames Flagged:  {res['fake_frames_flagged']} ({res['fake_frame_percentage']}%)")
            except Exception as e:
                print(f"Error analyzing video: {e}")
                
        elif choice == '3':
            epochs_in = input("Enter number of epochs (default 10): ").strip()
            epochs = int(epochs_in) if epochs_in.isdigit() else 10
            backbone_choice = input("Select backbone ['custom' or 'resnet18'] (default 'custom'): ").strip().lower()
            backbone = 'resnet18' if backbone_choice == 'resnet18' else 'custom'
            train_deepfake_model(DATASET_DIR, epochs=epochs, backbone=backbone)
            # Reload trained weights
            model, device = load_or_init_model()
            
        elif choice == '4' or choice.lower() == 'exit':
            print("Exiting Deepfake Detection System. Goodbye!")
            break
        else:
            print("Invalid selection. Please choose 1, 2, 3, or 4.")
