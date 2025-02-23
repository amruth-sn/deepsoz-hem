import pandas as pd
import numpy as np
import os

def create_manifest():
    
    
    
    manifest = []
    # /projectnb/seizuredet/Sz-challenge/siena-bids/processed/annotations/
    # /projectnb/seizuredet/Sz-challenge/tuh-bids/processed/annotations/
    for dataset in ['siena-bids', 'tuh-bids']:
        data_dir = f'/projectnb/seizuredet/Sz-challenge/{dataset}/processed/annotations/'
        for subject in os.listdir(data_dir):
            for session in os.listdir(data_dir + f"/{subject}"):
                for file in os.listdir(data_dir + f"/{subject}/{session}"):
                    df = pd.read_csv(data_dir + f"{subject}/{session}/{file}", sep='\t')
                    id = file.split('.')[0]
                    if len(id.split('seg')) == 1:
                        segment = '000'
                    else:
                        segment = id.split('seg')[1]
                        
                    run = id.split('M')[1].split('_')[1]
                    if df.iloc[0]['eventType'] == 'bckg':
                        manifest.append({
                            'subject': subject,
                            'session': session,
                            'run': run,
                            'segment': segment,
                            'fn': id,
                            'onset': [],
                            'nsz' : 0,
                            'duration': None,
                            'dataset': dataset
                        })
                    else:
                        onsets = []
                        durations = []
                        for _, row in df.iterrows():
                            onsets.append(row['onset'])
                            durations.append(row['duration'])
                        manifest.append({
                            'subject': subject,
                            'session': session,
                            'run': run,
                            'segment': segment,
                            'fn': id,
                            'onset': onsets,
                            'nsz' : len(onsets),
                            'duration': durations,
                            'dataset': dataset
                        })

    manifest = pd.DataFrame(manifest) 
    manifest.to_csv('/projectnb/seizuredet/Sz-challenge/code/manifest.csv', index=False)
                    
            
                
    
def main():
    create_manifest()
    
if __name__ == '__main__':
    main()