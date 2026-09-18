import os
import numpy as np
import pandas as pd
import albumentations as A
import joblib

from PIL import Image
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from helper_functions import build_data_registry

augment = A.Compose([
    A.Affine(scale=(0.9, 1.1), translate_percent=0.05, rotate=(-10, 10), shear=(-5, 5), p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
    A.RandomShadow(p=0.3),
    A.Sharpen(p=0.2),
])

def extract_hog(image_path, img_size=(80, 80), augment_fn=None):
    image = Image.open(image_path).convert("L")  
    image = image.resize(img_size)
    image = np.array(image)

    if augment_fn is not None:
        image = augment_fn(image=image)["image"]

    features = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8), 
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True
    )
    return features

def process_set(data_df, is_train=False):
    features_list = []
    labels_list = []

    for _, row in data_df.iterrows():
        try:
            label = int(row["label"])
            if is_train and label == 1:
                feat = extract_hog(row["image_path"], img_size=(80, 80))
                features_list.append(feat)
                labels_list.append(label)

                # Augmentation images creation
                for _ in range(3):
                    feat = extract_hog(
                        row["image_path"],
                        img_size=(80, 80),
                        augment_fn=augment
                    )
                    features_list.append(feat)
                    labels_list.append(label)

            elif is_train and label == 0:
                feat = extract_hog(row["image_path"], img_size=(80, 80))
                features_list.append(feat)
                labels_list.append(label)

                if np.random.rand() < 0.5:
                    feat = extract_hog(
                        row["image_path"],
                        img_size=(80, 80),
                        augment_fn=augment
                    )
                    features_list.append(feat)
                    labels_list.append(label)
            else:
                feat = extract_hog(row["image_path"], img_size=(80, 80))
                features_list.append(feat)
                labels_list.append(label)
        except:
            continue

    return np.array(features_list), np.array(labels_list)

def get_svm_score_df(model, data_df):
    results = []

    for _, row in data_df.iterrows():
        filename = os.path.basename(row["image_path"])
        unique_id = os.path.splitext(filename)[0]
        feat = extract_hog(row["image_path"]).reshape(1, -1)
        score = model.decision_function(feat)[0]
        results.append({'id_session': unique_id, 'score_image': score, 'label': int(row['label'])})

    return pd.DataFrame(results)

def generate_svm_output(model, image_dir, output_file, threshold=0.0):
    results = []
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith('.png')])
    print(f"Processing {len(image_files)} images from {image_dir}...")

    for img_name in image_files:
        stem = os.path.splitext(img_name)[0]
        img_path = os.path.join(image_dir, img_name)

        try:
            feat = extract_hog(img_path, img_size=(80, 80)).reshape(1, -1)
            raw_score = model.decision_function(feat)[0]
            pred_class = 1 if raw_score > threshold else 0

            line = f"{stem} {raw_score:.10f} {pred_class}"
            results.append(line)
            
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            continue

    with open(output_file, 'w') as f:
        f.write("\n".join(results))
    
    print(f"SVM scores saved to {output_file}")

# Split of data based on sessions
if __name__ == "__main__":
    df = build_data_registry()
    test_session_name = '03'    # Test session

    targets = df[df['label'] == 1]
    strangers = df[df['label'] == 0]

    t_train = targets[targets['session'] != test_session_name]
    t_test = targets[targets['session'] == test_session_name]

    s_train, s_test = train_test_split(strangers, test_size=0.2, random_state=42)

    train_df = pd.concat([t_train, s_train])
    test_df = pd.concat([t_test, s_test])

    X_train, y_train = process_set(train_df, is_train=True)
    X_test, y_test = process_set(test_df, is_train=False)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.9)),
        ("svc", SVC(
            kernel="rbf",
            C=5,
            gamma="scale", 
            probability=True, 
            class_weight={0:1, 1:3}
        ))
    ])

    model.fit(X_train, y_train)    
    y_pred = model.predict(X_test)
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, labels=[0, 1], target_names=["Stranger", "Target"]))

    df_img_scores = get_svm_score_df(model, test_df)
    df_img_scores.to_csv("scores_image.txt", sep=" ", index=False)
    
    print("Scores saved to scores_image.txt\n")

    model_filename = "svm_face_model.joblib"
    joblib.dump(model, model_filename)

    generate_svm_output(
        model=model, 
        image_dir="./Eval_data/eval",
        output_file="svm_image_results.txt",
        threshold=0.0
    )