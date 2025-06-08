import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def add_centroids_column():
    """Thêm cột is_centroids vào dữ liệu"""
    # Đọc dữ liệu
    df = pd.read_csv('df_cluster.csv')
    
    # Các đặc trưng để clustering
    features = ['avgtemp_c', 'maxwind_kph', 'avghumidity', 'avgvis_km']
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    
    # Lấy số cluster duy nhất
    n_clusters = df['cluster'].nunique()
    
    # Fit KMeans để lấy centroids
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    
    # Tìm điểm gần nhất với mỗi centroid
    centroids = kmeans.cluster_centers_
    df['is_centroids'] = 0
    
    for i, centroid in enumerate(centroids):
        # Tìm cluster tương ứng
        cluster_data = df[df['cluster'] == i]
        if len(cluster_data) == 0:
            continue
            
        cluster_indices = cluster_data.index
        cluster_scaled = X_scaled[cluster_indices]
        
        # Tính khoảng cách đến centroid
        distances = np.sqrt(np.sum((cluster_scaled - centroid) ** 2, axis=1))
        
        # Tìm điểm gần nhất
        closest_idx = cluster_indices[np.argmin(distances)]
        df.loc[closest_idx, 'is_centroids'] = 1
    
    # Lưu lại file
    df.to_csv('df_cluster.csv', index=False)
    print(f"Đã thêm cột is_centroids. Tổng số centroids: {df['is_centroids'].sum()}")
    
    # In thông tin về các centroids
    centroids_df = df[df['is_centroids'] == 1]
    print("\nCác centroids:")
    for _, row in centroids_df.iterrows():
        print(f"Cluster {row['cluster']}: {row['city']}, {row['province']} (tháng {row['month']})")

if __name__ == "__main__":
    add_centroids_column()
