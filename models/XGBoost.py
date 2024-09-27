import xgboost as xgb
from sklearn.metrics import accuracy_score


def train_xgboost(x_train_encoded, y_train, x_test_encoded, y_test):
    # Tạo DataLoader cho XGBoost
    dtrain = xgb.DMatrix(x_train_encoded, label=y_train)
    dtest = xgb.DMatrix(x_test_encoded, label=y_test)

    # Cài đặt tham số cho mô hình XGBoost
    params = {
        'objective': 'binary:logistic',  # Bài toán phân loại nhị phân
        'max_depth': 5,                   # Độ sâu tối đa của cây
        'learning_rate': 0.5,                      # Tốc độ học
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 1,  # Thêm regularization
        'lambda': 1,   # Thêm L2 regularization
        'eval_metric': 'logloss',          # Đánh giá theo log loss
        'tree_method': 'hist',         # Sử dụng GPU cho cây
        'device': 'cuda',                     # Sử dụng GPU
    }

    # Huấn luyện mô hình
    model = xgb.train(params, dtrain, num_boost_round=100)

    # Dự đoán trên tập test
    y_pred = model.predict(dtest)
    # Chuyển đổi xác suất sang nhãn
    y_pred_binary = [1 if i > 0.5 else 0 for i in y_pred]

    # Tính toán độ chính xác
    accuracy = accuracy_score(y_test, y_pred_binary)
    return accuracy
