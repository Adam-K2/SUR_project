import pandas as pd
import os
import joblib

from sklearn.linear_model import LogisticRegression
from svm import extract_hog
from audio_gmm import score_audio

def generate_fusion_output(fuser, svm_model, gmm_target, gmm_non_target, eval_dir, output_file):
    results = []
    image_files = sorted([f for f in os.listdir(eval_dir) if f.lower().endswith('.png')])
    
    print(f"Processing {len(image_files)} pairs from {eval_dir}...")

    for img_name in image_files:
        stem = os.path.splitext(img_name)[0]
        img_path = os.path.join(eval_dir, img_name)
        aud_path = os.path.join(eval_dir, stem + ".wav")

        if not os.path.exists(aud_path):
            print(f"Skipping {stem}: Audio file missing.")
            continue

        feat = extract_hog(img_path, img_size=(80, 80)).reshape(1, -1)
        score_img = svm_model.decision_function(feat)[0]
        score_aud, _ = score_audio(gmm_target, gmm_non_target, aud_path)

        # Fusion Decision
        fusion_input = pd.DataFrame([[score_img, score_aud]], columns=['score_image', 'score_audio'])
        raw_fusion_score = fuser.decision_function(fusion_input)[0]
        pred_class = int(fuser.predict(fusion_input)[0])

        line = f"{stem} {raw_fusion_score:.10f} {pred_class}"
        results.append(line)

    with open(output_file, 'w') as f:
        f.write("\n".join(results))
    
    print(f"Fusion results saved to {output_file}")

if __name__ == "__main__":
    # Load the score files
    df_image = pd.read_csv("scores_image.txt", sep=" ")
    df_audio = pd.read_csv("scores_audio.txt", sep=" ")

    df_fusion = pd.merge(df_image, df_audio, on=['id_session', 'label'])

    # Prepare for Logistic Regression
    X = df_fusion[['score_image', 'score_audio']]
    y = df_fusion['label']

    fuser = LogisticRegression(class_weight='balanced')
    fuser.fit(X, y)

    print(f"Weights: image={fuser.coef_[0][0]:.4f}, audio={fuser.coef_[0][1]:.4f}, bias={fuser.intercept_[0]:.4f}")

    # Load all models
    svm_model = joblib.load("svm_face_model.joblib")
    gmm_target = joblib.load("gmm_target.pkl")
    gmm_non_target = joblib.load("gmm_non_target.pkl")
    
    generate_fusion_output(
        fuser=fuser,
        svm_model=svm_model,
        gmm_target=gmm_target,
        gmm_non_target=gmm_non_target,
        eval_dir='./Eval_data/eval',
        output_file='multimodal_fusion_results.txt'
    )