import os
import pandas as pd
import numpy as np
import librosa

def build_data_registry(root_dir="."):
    folders = [
        'target_train', 'target_dev', 
        'non_target_train', 'non_target_dev'
    ]
    records = []

    for folder in folders:
        folder_path = os.path.join(root_dir, folder)
        if not os.path.exists(folder_path):
            continue

        current_label = 1 if folder.startswith('target') else 0
        
        files = os.listdir(folder_path)
        images = [f for f in files if f.endswith('.png')]
        
        for img_name in images:
            parts = img_name.split('_')
            identity = parts[0]
            session = parts[1]
            
            base_name = img_name.replace('.png', '')
            wav_name = base_name + '.wav'
            wav_path = os.path.join(folder_path, wav_name)
            
            records.append({
                'id': identity,
                'session': session,
                'label': current_label,
                'is_dev': 'dev' in folder,
                'image_path': os.path.join(folder_path, img_name),
                'audio_path': wav_path if os.path.exists(wav_path) else None,
                'original_folder': folder
            })

    df = pd.DataFrame(records)
    return df

def to_feat(sig):
    sig = sig[2 * 16000:]
    sig, _ = librosa.effects.trim(sig, top_db=20)
    mfcc = librosa.feature.mfcc(y=sig, sr=16000, n_mfcc=13, n_fft=400, hop_length=160)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    return np.vstack([mfcc, delta, delta2]).T

def extract_mfcc(audio_path, sr_target=16000):
    y, sr = librosa.load(audio_path, sr=sr_target)
    
    # Remove silence in the first 2 seconds
    start_sample = 2 * sr_target
    y = y[start_sample:] 
    
    # Remove remaining silence
    y, _ = librosa.effects.trim(y, top_db=20)
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=400, hop_length=160)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    
    features = np.vstack([mfcc, delta, delta2]).T
    
    return features

def score_audio(gmm_target, gmm_non_target, audio_path, threshold=0.0):
    feat = extract_mfcc(audio_path)
    
    score_t = gmm_target.score(feat)
    score_nt = gmm_non_target.score(feat)
    
    llr = score_t - score_nt
    pred = 1 if llr > threshold else 0
    
    return llr, pred

def generate_gmm_output(gmm_t, gmm_nt, audio_dir, output_file, threshold=0.0):
    results = []
    
    audio_files = sorted([f for f in os.listdir(audio_dir) if f.lower().endswith('.wav')])
    
    print(f"Processing {len(audio_files)} audio files from {audio_dir}...")

    for aud_name in audio_files:
        stem = os.path.splitext(aud_name)[0]
        aud_path = os.path.join(audio_dir, aud_name)

        try:
            llr, pred = score_audio(gmm_t, gmm_nt, aud_path, threshold)
            line = f"{stem} {llr:.10f} {pred}"
            results.append(line)
            
        except Exception as e:
            print(f"Error processing {aud_name}: {e}")
            continue

    with open(output_file, 'w') as f:
        f.write("\n".join(results))
    
    print(f"Audio GMM scores saved to {output_file}")
