from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 初始化 SVM 分类器
svm = SVC(kernel='linear', random_state=42)

# 训练模型
svm.fit(X_train, y_train)

# 预测
y_pred_svm = svm.predict(X_test)

# 评估
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f'SVM Accuracy: {accuracy_svm:.2f}')
