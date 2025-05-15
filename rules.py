import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score, accuracy_score

class FuzzyDecisionTreeGini:
    def __init__(self, max_depth=3, min_samples_split=10):  # Fixed constructor
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        self.feature_names = ['soil_moisture_fuzzy', 'temperature_fuzzy', 'humidity_fuzzy', 'ph_fuzzy', 'rainfall_fuzzy']
        self.label_names = ['OFF', 'ON']
        self.used_features = set()
    
    def _ensure_numpy(self, data):
        data = np.array(data, dtype=object)
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        return data
    
    def fit(self, X, y):
        X = self._ensure_numpy(X)
        y = self._ensure_numpy(y).flatten()
        
        if X.shape[0] != len(self.feature_names):
            X = X.T
        
        categorical_data = []
        for i in range(X.shape[1]):
            sample = {}
            for j, feature in enumerate(self.feature_names):
                sample[feature] = X[j, i]
            categorical_data.append(sample)
        
        self.used_features = set()
        self.tree = self._build_tree(categorical_data, y, depth=0)
    
    def _calculate_gini(self, labels):
        if len(labels) == 0:
            return 0
        p = np.mean(labels)
        return 1 - p*2 - (1-p)*2
    
    def _build_tree(self, data, labels, depth):
        labels = np.array(labels, dtype=int)
        if (depth >= self.max_depth or 
            len(data) < self.min_samples_split or 
            len(np.unique(labels)) == 1):
            return {'prediction': np.mean(labels) > 0.5}
        
        best_gini = float('inf')
        best_feature = None
        
        available_features = [f for f in self.feature_names if f not in self.used_features]
        
        for feature in available_features:
            value_counts = defaultdict(list)
            for i, sample in enumerate(data):
                value = str(sample[feature])
                value_counts[value].append(labels[i])
            
            weighted_gini = 0
            for value_labels in value_counts.values():
                p = len(value_labels) / len(labels)
                weighted_gini += p * self._calculate_gini(value_labels)
            
            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_feature = feature
        
        if best_feature is None:
            return {'prediction': np.mean(labels) > 0.5}
        
        self.used_features.add(best_feature)
        tree = {'feature': best_feature, 'children': {}}
        value_groups = defaultdict(list)
        for i, sample in enumerate(data):
            value = str(sample[best_feature])
            value_groups[value].append((sample, labels[i]))
        
        for value, items in value_groups.items():
            sub_data, sub_labels = zip(*items)
            tree['children'][value] = self._build_tree(sub_data, sub_labels, depth+1)
        
        self.used_features.remove(best_feature)
        return tree
    
    def predict(self, X):
        if self.tree is None:
            raise ValueError("Model not trained yet!")
            
        X = self._ensure_numpy(X)
        if X.shape[0] != len(self.feature_names):
            X = X.T
        
        predictions = []
        for i in range(X.shape[1]):
            sample = {}
            for j, feature in enumerate(self.feature_names):
                sample[feature] = X[j, i]
            
            node = self.tree
            while 'children' in node:
                feature_value = str(sample[node['feature']])
                if feature_value in node['children']:
                    node = node['children'][feature_value]
                else:
                    break
            
            if 'prediction' in node:
                predictions.append(node['prediction'])
            else:
                predictions.append(False)
        
        return np.array(predictions)
    
    def print_rules(self, node=None, path=None):
        if node is None:
            if self.tree is None:
                print("Model not trained yet!")
                return
            node = self.tree
            path = []
        
        if 'prediction' in node:
            conditions = " AND ".join(path) if path else "ALWAYS"
            print(f"IF {conditions} THEN Pump is {'ON' if node['prediction'] else 'OFF'}")
            return
        
        for value, child in node['children'].items():
            new_path = path + [f"{node['feature']} is {value}"]
            self.print_rules(child, new_path)
    
    def visualize_tree(self, filename='balanced_decision_tree_gini.png'):
        if self.tree is None:
            print("Model not trained yet!")
            return

        fig, ax = plt.subplots(figsize=(15, 10))
        ax.axis('off')
        plt.title("Fuzzy Decision Tree (Gini Index)", pad=20, fontsize=16)
        
        max_depth = self._get_tree_depth(self.tree)
        self._draw_node(ax, self.tree, x=0.5, y=0.9, width=0.4, depth=0, max_depth=max_depth)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Tree visualization saved as {filename}")
        plt.close()
    
    def _get_tree_depth(self, node):
        if 'prediction' in node:
            return 0
        max_depth = 0
        for child in node['children'].values():
            depth = self._get_tree_depth(child)
            if depth > max_depth:
                max_depth = depth
        return max_depth + 1
    
    def _draw_node(self, ax, node, x, y, width, depth, max_depth):
        y_step = 0.7 / max_depth
        
        if 'prediction' in node:
            color = 'lightgreen' if node['prediction'] else 'lightcoral'
            ax.add_patch(Rectangle((x-width/2, y-0.02), width, 0.04, 
                          color=color, ec='black', lw=1))
            ax.text(x, y, f"Pump: {'ON' if node['prediction'] else 'OFF'}",
                   ha='center', va='center', fontsize=10, bbox=dict(facecolor=color, alpha=0.7))
        else:
            ax.add_patch(Rectangle((x-width/2, y-0.02), width, 0.04,
                          color='lightblue', ec='black', lw=1))
            ax.text(x, y, f"{node['feature']}?", 
                   ha='center', va='center', fontsize=10, bbox=dict(facecolor='lightblue', alpha=0.7))
            
            num_children = len(node['children'])
            if num_children == 0:
                return
                
            child_width = width / num_children
            x_pos = x - width/2 + child_width/2
            
            for i, (value, child) in enumerate(node['children'].items()):
                ax.plot([x, x_pos], [y-0.02, y-y_step+0.02], 'k-', lw=1)
                ax.text((x+x_pos)/2, y-y_step/2, value, 
                       ha='center', va='center', fontsize=9, rotation=45)
                self._draw_node(ax, child, x_pos, y-y_step, 
                              child_width*0.9, depth+1, max_depth)
                x_pos += child_width

    def evaluate_performance(self, y_true, y_pred, dataset_name="Dataset"):
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)

        print(f"\nPerformance on {dataset_name}:")
        print("-" * 40)
        print(f"Accuracy          : {accuracy:.4f}")
        print(f"Precision         : {precision:.4f}")
        print(f"Recall            : {recall:.4f}")
        print(f"F1-Score          : {f1:.4f}")
        print(f"Balanced Accuracy : {balanced_acc:.4f}")
        

def load_fuzzy_data(filepath):
    data = pd.read_csv(filepath)
    fuzzy_columns = ['soil_moisture_fuzzy', 'temperature_fuzzy', 'humidity_fuzzy', 
                    'ph_fuzzy', 'rainfall_fuzzy']
    
    for col in fuzzy_columns:
        data[col] = data[col].str.lower().str.strip()
    
    X = data[fuzzy_columns].values.T
    y = data['Pump Data'].apply(lambda x: 1 if x == 'ON' else 0).values
    return X, y

# Fixed __name__ check
if __name__ == "__main__":
    print("Fuzzy Decision Tree with Gini Index Implementation")
    print("="*60)
    
    try:
        X, y = load_fuzzy_data("balanced_fuzzified_dataset.csv")
        print(f"Loaded dataset with {X.shape[1]} samples")
        
        split_idx = int(0.8 * X.shape[1])
        X_train, X_test = X[:, :split_idx], X[:, split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print("\nTraining Fuzzy Decision Tree using Gini Index...")
        fdt = FuzzyDecisionTreeGini(max_depth=4, min_samples_split=5)
        fdt.fit(X_train, y_train)
        
        print("\nGenerated Decision Rules:")
        print("="*40)
        fdt.print_rules()
        
        print("\nGenerating tree visualization...")
        fdt.visualize_tree()
        
        train_pred = fdt.predict(X_train)
        test_pred = fdt.predict(X_test)

        fdt.evaluate_performance(y_train, train_pred, dataset_name="Training Set")
        fdt.evaluate_performance(y_test, test_pred, dataset_name="Test Set")
        
        print("\nExample Test Predictions:")
        print("="*40)
        for i in range(min(5, X_test.shape[1])):
            print(f"Sample {i+1}:")
            for j, feature in enumerate(fdt.feature_names):
                print(f"  {feature}: {X_test[j, i]}")
            print(f"  Actual: {'ON' if y_test[i] else 'OFF'}")
            print(f"  Predicted: {'ON' if test_pred[i] else 'OFF'}\n")
            
    except FileNotFoundError:
        print("Error: 'balanced_fuzzified_dataset.csv' not found in current directory")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
