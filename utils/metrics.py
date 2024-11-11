from sklearn.metrics import precision_score, recall_score, f1_score

def compute_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return precision, recall, f1
