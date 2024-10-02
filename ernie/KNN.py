from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 初始化 k-NN 分类器
knn = KNeighborsClassifier(n_neighbors=5)

# 训练模型
knn.fit(X_train, y_train)

# 预测
y_pred_knn = knn.predict(X_test)

# 评估
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f'k-NN Accuracy: {accuracy_knn:.2f}')
