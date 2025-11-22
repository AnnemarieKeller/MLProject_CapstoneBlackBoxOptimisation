scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X_init)

# Scale y (target)
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y_init.reshape(-1,1))