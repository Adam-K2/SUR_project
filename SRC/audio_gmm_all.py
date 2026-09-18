import numpy as np
import librosa

from sklearn.mixture import GaussianMixture
from helper_functions import build_data_registry, to_feat, generate_gmm_output

def get_training_frames(data_df):
    target_frames = []
    non_target_frames = []

    for _, row in data_df.iterrows():
        try:
            y, _ = librosa.load(row["audio_path"], sr=16000)
            orig_feat = to_feat(y)
            if row["label"] == 1:
                target_frames.append(orig_feat)
            else:
                non_target_frames.append(orig_feat)

        except Exception as e:
            continue

    X_target = np.vstack(target_frames) if target_frames else np.array([])
    X_non_target = np.vstack(non_target_frames) if non_target_frames else np.array([])

    return X_target, X_non_target

if __name__ == "__main__":
    df = build_data_registry()
    X_target_all, X_non_target_all = get_training_frames(df)
    
    gmm_target = GaussianMixture(n_components=16, covariance_type='diag', random_state=42)
    gmm_non_target = GaussianMixture(n_components=16, covariance_type='diag', random_state=42)
    
    gmm_target.fit(X_target_all)
    gmm_non_target.fit(X_non_target_all)

    print("Training finished.")

    generate_gmm_output(
        gmm_t=gmm_target, 
        gmm_nt=gmm_non_target, 
        audio_dir="./Eval_data/eval",
        output_file="gmm_audio_all_results.txt",
        threshold=0.0
    )
