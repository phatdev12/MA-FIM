import networkx as nx
import pickle
import random

# --- CẤU HÌNH ---
input_txt_file = 'networks/karate_with_attributes.txt'  # Đổi tên này thành tên file .txt của bạn
output_pickle_file = 'networks/karate_with_attributes.pickle'

try:
    print(f"1. Đang đọc file '{input_txt_file}'...")
    
    # Đọc file txt danh sách cạnh
    # Giả sử file txt dạng: NodeA NodeB (cách nhau bởi khoảng trắng)
    G = nx.read_edgelist(input_txt_file, create_using=nx.Graph(), nodetype=str)
    
    print(f"   -> Đã load Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

    print("2. Đang tạo attribute 'age' (giả lập theo cộng đồng)...")

    # --- TẠO AGE DỰA TRÊN CẤU TRÚC MẠNG ---
    # Bước 1: Tìm các nhóm cộng đồng (Community Detection)
    # Những người trong cùng 1 nhóm sẽ có độ tuổi sàn sàn nhau
    try:
        # Dùng thuật toán Louvain để tìm nhóm
        communities = nx.community.louvain_communities(G, seed=42)
        print(f"   -> Đã tìm thấy {len(communities)} nhóm cộng đồng.")
        
        for group in communities:
            # Mỗi nhóm sẽ có một độ tuổi trung bình (Base Age)
            # Ví dụ: Nhóm sinh viên (~20), Nhóm đi làm (~30), Nhóm trung niên (~50)
            base_age = random.randint(18, 60)
            
            for node in group:
                # Tuổi của node = Tuổi gốc của nhóm + Chênh lệch ngẫu nhiên (-5 đến +5 tuổi)
                age = base_age + random.randint(-5, 5)
                
                # Giới hạn tuổi hợp lý (ví dụ: 15 đến 90)
                age = max(15, min(90, age))
                
                G.nodes[node]['age'] = age
                
    except AttributeError:
        # Nếu thư viện NetworkX cũ không có Louvain, dùng Random hoàn toàn
        print("   -> (Dùng random thuần túy do không tìm thấy module community)")
        for node in G.nodes():
            G.nodes[node]['age'] = random.randint(18, 65)

    # --- LƯU FILE ---
    print(f"3. Đang lưu kết quả ra '{output_pickle_file}'...")
    with open(output_pickle_file, 'wb') as f:
        pickle.dump(G, f)
        
    print("\n✅ THÀNH CÔNG! File pickle đã có attribute 'age'.")

    # --- KIỂM TRA LẠI ---
    print("-" * 30)
    print("Kiểm tra 3 node ngẫu nhiên:")
    # Lấy 3 node để in thử
    sample_nodes = list(G.nodes())[:3]
    for node in sample_nodes:
        print(f"- Node {node}: {G.nodes[node]}")

except FileNotFoundError:
    print(f"❌ Lỗi: Không tìm thấy file '{input_txt_file}'.")
except Exception as e:
    print(f"❌ Lỗi: {e}")