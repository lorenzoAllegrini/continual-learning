import os
import pandas as pd
from tabulate import tabulate
base_dir = "experiments"
results = []


for root, dirs, files in os.walk(base_dir):
    print(root)
    if "results.csv" in files:
        print("results.csv")
        file_path = os.path.join(root, "results.csv")
        results_df = pd.read_csv(file_path)

        if {'true_positives', 'false_positives', 'false_negatives', 'train_time'}.issubset(results_df.columns):
            tp = 0
            fp = 0
            fn = 0
            tn = 0
            eval_loss = 0
            
            avg_time = 0
            for _,row in results_df.iterrows():
                tp += row['true_positives']
                fp += row['false_positives']
                fn += row['false_negatives']
                eval_loss += row['eval_loss']
                avg_time += row['train_time']
                

            avg_time /= len(results_df) if len(results_df) > 0 else 0
            eval_loss /= len(results_df) if len(results_df) > 0 else 0
            precision = tp / (tp + fp) if tp + fp > 0 else 0.0 
            recall = tp / (tp + fn) if tp + fn > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
            # Estrai il nome della directory relativa (es. 'nasa_model1')
            model_name = os.path.relpath(root, base_dir)


            results.append({
                'eval_loss': eval_loss,
                'model': model_name,
                'avg_time': avg_time,
                'precision': precision,
                'recall': recall,
                'f1': f1,
            })

# Crea il DataFrame completo
df = pd.DataFrame(results)
print(df)
    # Droppa la colonna 'dataset' e stampa la classifica in bel formato
print(tabulate(df, headers='keys', tablefmt='fancy_grid', floatfmt=".4f"))