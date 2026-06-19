import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    # Write code here
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred, dtype=float)

    # Избегание 0
    eps = 1e-16
    y = np.clip(y_pred, eps, 1.0 - eps)

    # Бинарная классификация
    # y_true содержит только бинарные значения
    if y_pred.ndim == 1 and y_true.shape == y_pred.shape:
        if np.all((y_true == 0) | (y_true == 1)):
            loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
            return float(np.mean(loss))

    # Если метки являются индексами классов, а предсказание - вероятностями
    if y_true.ndim == 1 and y_pred.ndim == 2 and y_true.shape[0] == y_pred.shape[0]:
        y_true = y_true.astype(int)
        loss = -np.log(y_pred[np.arange(len(y_true)), y_true])
        return float(np.mean(loss))

    # One-hot/ распределение вероятностей
    if y_true.shape == y_pred.shape:
        loss = -np.sum(y_true * np.log(y_pred), axis=-1)
        return float(np.means(loss))
        
    raise ValueError("Shape y_true != y_pred")

    
    return ce