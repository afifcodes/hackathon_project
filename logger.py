import json
import os

def save_training_results(history_dict, epochs, num_classes, model_dir='models'):
    """
    Saves training history to JSON and a human-readable summary to a TXT file.
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # 1. Save raw history to JSON
    history_path = os.path.join(model_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=4)
    print(f"Training history saved as {history_path}")

    # 2. Save human-readable summary report
    summary_path = os.path.join(model_dir, 'training_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("CropGuard AI - Training Summary Report\n")
        f.write("====================================\n")
        f.write(f"Total Epochs: {epochs}\n")
        f.write(f"Classes Detected: {num_classes}\n")
        f.write("-" * 30 + "\n")
        
        # Guard against empty history
        if history_dict.get('accuracy'):
            f.write(f"Final Training Accuracy:   {history_dict['accuracy'][-1]:.4f}\n")
            f.write(f"Final Validation Accuracy: {history_dict['val_accuracy'][-1]:.4f}\n")
            f.write(f"Final Training Loss:       {history_dict['loss'][-1]:.4f}\n")
            f.write(f"Final Validation Loss:     {history_dict['val_loss'][-1]:.4f}\n")
            f.write("-" * 30 + "\n\n")
            
            f.write("Epoch-by-Epoch Progress:\n")
            for i in range(len(history_dict['accuracy'])):
                f.write(f"Epoch {i+1:02d}: Acc: {history_dict['accuracy'][i]:.4f} | Val_Acc: {history_dict['val_accuracy'][i]:.4f}\n")
        else:
            f.write("No history data found.\n")
            
    print(f"Summary report saved as {summary_path}")
