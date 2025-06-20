import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

def find_model_file():
    """Find the trained model file"""
    possible_paths = [
        '../saved_models/best_model.h5',  # If running from src directory
        '../saved_models/model.h5',       
        'saved_models/best_model.h5',     # If running from root directory
        'saved_models/model.h5'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # If no model found, list available files
    print("Model file not found. Available files:")
    for check_dir in ['../saved_models', 'saved_models']:
        if os.path.exists(check_dir):
            print(f"In {check_dir} directory:")
            for file in os.listdir(check_dir):
                print(f"  - {file}")
    
    return None

def load_test_data():
    """Load test data from CSV file"""
    # Try different paths for the CSV file
    csv_paths = [
        '../data/test.csv',  # If running from src directory
        'data/test.csv'      # If running from root directory
    ]
    
    test_df = None
    for path in csv_paths:
        if os.path.exists(path):
            print(f"Loading test data from: {path}")
            test_df = pd.read_csv(path)
            break
    
    if test_df is None:
        raise FileNotFoundError("Could not find test.csv file")
    
    print(f"Test data shape: {test_df.shape}")
    
    # Extract labels and features based on your CSV structure
    y_true = test_df['label'].values
    
    # Get pixel columns (pixel1 to pixel784)
    pixel_columns = [f'pixel{i}' for i in range(1, 785)]
    X_test = test_df[pixel_columns].values
    
    # Reshape data for CNN input (28x28 images)
    img_height, img_width = 28, 28
    X_test = X_test.reshape(-1, img_height, img_width, 1)
    X_test = X_test.astype('float32') / 255.0  # Normalize
    
    print(f"Loaded {len(X_test)} test samples")
    print(f"Image shape: {img_height}x{img_width}")
    print(f"Unique labels: {sorted(np.unique(y_true))}")
    
    return X_test, y_true

def get_class_names():
    """Get class names based on your dataset"""
    # Based on the label value 3 in your sample, it seems you're using numeric labels
    # You'll need to map these to actual sign letters
    # Common mapping for sign language datasets:
    label_to_letter = {
        0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 
        9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 
        17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
    }
    
    return label_to_letter

def analyze_model_performance():
    """Analyze model performance on test data"""
    
    # Load test data
    X_test, y_true = load_test_data()
    
    # Get predictions
    print("Making predictions...")
    predictions = model.predict(X_test, verbose=1)
    y_pred_classes = np.argmax(predictions, axis=1)
    
    # Get unique classes present in the data
    unique_classes = np.unique(np.concatenate([y_true, y_pred_classes]))
    label_to_letter = get_class_names()
    
    # Map numeric labels to letters for classes present in data
    present_class_names = [label_to_letter.get(i, f'Class_{i}') for i in unique_classes]
    
    print(f"Classes found in data: {unique_classes}")
    print(f"Corresponding letters: {present_class_names}")
    
    # Generate classification report
    report = classification_report(y_true, y_pred_classes,
                                 labels=unique_classes,
                                 target_names=present_class_names,
                                 output_dict=True,
                                 zero_division=0)
    
    return {
        'predictions': predictions,
        'y_pred_classes': y_pred_classes,
        'y_true': y_true,
        'class_names': present_class_names,
        'unique_classes': unique_classes,
        'label_to_letter': label_to_letter,
        'report': report
    }

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_class_performance(report, class_names):
    """Plot per-class performance metrics"""
    
    # Extract metrics for each class
    precision = []
    recall = []
    f1_score = []
    
    for class_name in class_names:
        if class_name in report and isinstance(report[class_name], dict):
            precision.append(report[class_name]['precision'])
            recall.append(report[class_name]['recall'])
            f1_score.append(report[class_name]['f1-score'])
        else:
            precision.append(0)
            recall.append(0)
            f1_score.append(0)
    
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1_score, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Sign Classes')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Performance Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('class_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

def get_problem_signs(threshold=0.7):
    """Identify signs that need improvement"""
    
    print("Analyzing model performance...")
    performance = analyze_model_performance()
    
    report = performance['report']
    class_names = performance['class_names']
    
    problem_signs = []
    
    for class_name in class_names:
        if class_name in report and isinstance(report[class_name], dict):
            metrics = report[class_name]
            f1 = metrics['f1-score']
            if f1 < threshold:
                problem_signs.append({
                    'sign': class_name,
                    'f1_score': f1,
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'support': metrics['support']
                })
    
    return problem_signs, performance

def generate_detailed_report():
    """Generate comprehensive analysis report"""
    
    print("=== DETAILED PERFORMANCE REPORT ===")
    
    try:
        problem_signs, performance = get_problem_signs(threshold=0.7)
        
        report = performance['report']
        class_names = performance['class_names']
        y_true = performance['y_true']
        y_pred_classes = performance['y_pred_classes']
        
        # Overall metrics
        print(f"\nOVERALL PERFORMANCE:")
        print(f"Accuracy: {report['accuracy']:.3f}")
        print(f"Macro Average F1: {report['macro avg']['f1-score']:.3f}")
        print(f"Weighted Average F1: {report['weighted avg']['f1-score']:.3f}")
        
        # Problem signs
        print(f"\nPROBLEM SIGNS (F1 < 0.7):")
        if problem_signs:
            for sign_info in problem_signs:
                print(f"Sign '{sign_info['sign']}': F1={sign_info['f1_score']:.3f}, "
                      f"Precision={sign_info['precision']:.3f}, "
                      f"Recall={sign_info['recall']:.3f}, "
                      f"Support={sign_info['support']}")
        else:
            print("No problem signs found! All signs performing well.")
        
        # Best performing signs
        print(f"\nBEST PERFORMING SIGNS:")
        best_signs = []
        for class_name in class_names:
            if class_name in report and isinstance(report[class_name], dict):
                f1 = report[class_name]['f1-score']
                best_signs.append((class_name, f1))
        
        best_signs.sort(key=lambda x: x[1], reverse=True)
        for sign, f1 in best_signs[:5]:
            print(f"Sign '{sign}': F1={f1:.3f}")
        
        # Generate visualizations
        print(f"\nGenerating visualizations...")
        plot_confusion_matrix(y_true, y_pred_classes, class_names)
        plot_class_performance(report, class_names)
        
        # Recommendations
        print(f"\nRECOMMENDATIONS:")
        if problem_signs:
            print("1. Focus data collection on problem signs")
            print("2. Consider data augmentation for underperforming classes")
            print("3. Review image quality for these signs")
            print("4. Consider focal loss for class imbalance")
            print("5. Use the focused_training.py script to retrain on problem signs")
        else:
            print("1. Model is performing well across all signs")
            print("2. Consider testing with more challenging data")
            print("3. Evaluate real-world performance")
        
        return problem_signs
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        print("Please check your data paths and file formats")
        return []

# Load the trained model
print("Loading model...")
model_path = find_model_file()
if model_path is None:
    raise FileNotFoundError("Could not find the trained model file in saved_models directory.")

print(f"Loading model from: {model_path}")
model = tf.keras.models.load_model(model_path)
print("Model loaded successfully!")

if __name__ == "__main__":
    problem_signs = generate_detailed_report()
    
    print(f"\nAnalysis complete! Check generated plots:")
    print("- confusion_matrix.png")
    print("- class_performance.png")
    
    if problem_signs:
        print(f"\nFound {len(problem_signs)} signs that need improvement.")
        print("Consider running focused_training.py to improve these signs.")