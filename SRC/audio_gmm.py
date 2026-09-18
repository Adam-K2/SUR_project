import os
import numpy as np
import pandas as pd
import librosa
import pickle

from sklearn.mixture import GaussianMixture
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from helper_functions import build_data_registry, to_feat, score_audio, generate_gmm_output

def add_white_noise(y, noise, snr_db=10):
    noise = np.random.randn(len(y))
    signal_power = np.mean(y**2)
    noise_power = np.mean(noise**2)
    factor = np.sqrt(signal_power / (10**(snr_db / 10) * noise_power))
    return y + factor * noise

def random_gain(y, low=0.5, high=1.2):
    return y * np.random.uniform(low, high)

def get_training_frames(data_df, augment=True):
    target_frames = []
    non_target_frames = []

    for _, row in data_df.iterrows():
        try:
            y, sr = librosa.load(row["audio_path"], sr=16000)

            orig_feat = to_feat(y)
            if row["label"] == 1:
                target_frames.append(orig_feat)
                
                if augment:
                    # Randomly pick augmentation or just do not augment input
                    rand_val = np.random.random()
                    if rand_val < 0.1: 
                        target_frames.append(to_feat(librosa.effects.pitch_shift(y, sr=sr, n_steps=0.5)))
                    elif rand_val < 0.2:
                        target_frames.append(to_feat(librosa.effects.time_stretch(y, rate=1.05)))
                    elif rand_val < 0.3:
                        target_frames.append(to_feat(add_white_noise(y)))
                    elif rand_val < 0.4:
                        target_frames.append(to_feat(random_gain(y)))
            else:
                non_target_frames.append(orig_feat)

                if augment:
                    rand_val = np.random.random()
                    if rand_val < 0.05: 
                        non_target_frames.append(to_feat(librosa.effects.pitch_shift(y, sr=sr, n_steps=0.5)))
                    elif rand_val < 0.1:
                        non_target_frames.append(to_feat(librosa.effects.time_stretch(y, rate=1.05)))
                    elif rand_val < 0.15:
                        non_target_frames.append(to_feat(add_white_noise(y)))
                    elif rand_val < 0.2:
                        non_target_frames.append(to_feat(random_gain(y)))

        except Exception as e:
            continue

    X_target = np.vstack(target_frames) if target_frames else np.array([])
    X_non_target = np.vstack(non_target_frames) if non_target_frames else np.array([])

    return X_target, X_non_target

if __name__ == "__main__":
    df = build_data_registry()
    test_session_name = '03' 

    targets = df[df['label'] == 1]
    strangers = df[df['label'] == 0]

    t_train = targets[targets['session'] != test_session_name]
    t_test = targets[targets['session'] == test_session_name]

    s_train, s_test = train_test_split(strangers, test_size=0.2, random_state=42)

    train_df = pd.concat([t_train, s_train])
    test_df = pd.concat([t_test, s_test])

    X_target_train, X_non_target_train = get_training_frames(train_df)

    gmm_target = GaussianMixture(n_components=16, covariance_type='diag', random_state=42)
    gmm_non_target = GaussianMixture(n_components=16, covariance_type='diag', random_state=42)

    gmm_target.fit(X_target_train)
    gmm_non_target.fit(X_non_target_train)

    y_test_true = []
    y_test_pred = []
    audio_scores = [] # For fusion model
    
    for _, row in test_df.iterrows():
        llr, pred = score_audio(gmm_target, gmm_non_target, row["audio_path"])
        y_test_true.append(row["label"])
        y_test_pred.append(pred)

        filename = os.path.basename(row["audio_path"])
        unique_id = os.path.splitext(filename)[0]

        audio_scores.append({
        'id_session': unique_id,
        'score_audio': llr,
        'label': row['label']
    })

    print("Confusion Matrix:")
    print(confusion_matrix(y_test_true, y_test_pred, labels=[0, 1]))
    
    print("\nClassification Report:")
    print(classification_report(y_test_true, y_test_pred, labels=[0, 1], target_names=["Stranger", "Target"]))

    # Save to CSV
    df_aud_out = pd.DataFrame(audio_scores)
    df_aud_out.to_csv("scores_audio.txt", sep=" ", index=False)
    print("Audio scores saved to scores_audio.txt")

    # Save the models 
    with open('gmm_target.pkl', 'wb') as f:
        pickle.dump(gmm_target, f)
    with open('gmm_non_target.pkl', 'wb') as f:
        pickle.dump(gmm_non_target, f)

    # Posibility of load
    # with open('gmm_target_test.pkl', 'rb') as f:
    #     gmm_target = pickle.load(f)
    # with open('gmm_non_target_test.pkl', 'rb') as f:
    #     gmm_non_target = pickle.load(f)

    generate_gmm_output(
        gmm_t=gmm_target, 
        gmm_nt=gmm_non_target, 
        audio_dir="./Eval_data/eval",
        output_file="gmm_audio_results.txt",
        threshold=0.0
    )
