import pickle
import networkx as nx

# Tên file đầu vào và đầu ra
input_filename = 'networks/karate_with_attributes.pickle'
output_filename = 'networks/karate_with_attributes.txt'

try:
    # 1. Đọc file pickle
    print(f"Đang đọc file {input_filename}...")
    with open(input_filename, 'rb') as f:
        G = pickle.load(f)
    
    # 2. Xuất cạnh ra file .txt
    # data=False: Chỉ lưu "Nguồn Đích" (Node1 Node2)
    # data=True: Nếu bạn muốn lưu kèm trọng số hoặc thuộc tính của cạnh (nếu có)
    nx.write_edgelist(G, output_filename, data=False, encoding='utf-8')
    
    print(f"✅ Thành công! Đã lưu {G.number_of_edges()} cạnh vào file '{output_filename}'")

    # 3. (Tuỳ chọn) In thử 5 dòng đầu tiên để kiểm tra
    print("-" * 30)
    print("Nội dung xem trước (5 dòng đầu):")
    with open(output_filename, 'r') as f:
        for i in range(5):
            line = f.readline()
            if not line: break
            print(line.strip())
            
except FileNotFoundError:
    print(f"❌ Lỗi: Không tìm thấy file '{input_filename}'. Hãy chắc chắn file đang ở cùng thư mục với code.")
except Exception as e:
    print(f"❌ Có lỗi xảy ra: {e}")