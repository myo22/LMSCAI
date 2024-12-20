from flask import Flask, request, jsonify
import numpy as np
from models.siamese_network import SiameseNetwork

app = Flask(__name__)

# 모델 로드 (사전에 학습된 모델을 사용)
model = SiameseNetwork(input_dim=768)  # input_dim을 768로 수정
model.load_weights('siamese_model_weights.npz')  # 저장된 모델 로드

@app.route('/predict', methods=['POST'])
def predict_similarity():
    try:
        data = request.get_json()

        # 입력 데이터 검증
        if 'comment1' not in data or 'comment2' not in data:
            return jsonify({'error': 'Both comment1 and comment2 must be provided'}), 400

        comment1_list = np.array(data.get('comment1', [])).astype(np.float32)
        comment2_list = np.array(data.get('comment2', [])).astype(np.float32)

        # 입력 데이터가 비어 있는지 확인
        if comment1_list.size == 0 or comment2_list.size == 0:
            return jsonify({'error': 'comment1 or comment2 is empty or invalid'}), 400

        # 유사도를 계산할 리스트 초기화
        similarities = []   

        # 각 댓글 쌍에 대해 유사도 계산
        for comment1, comment2 in zip(comment1_list, comment2_list):
            if len(comment1) > 0:
                if comment1.shape[0] < 768:
                    comment1 = np.pad(comment1, (0, 768 - comment1.shape[0]), 'constant')
            else:
                return jsonify({'error': 'comment1 contains empty input'}), 400

            if len(comment2) > 0:
                if comment2.shape[0] < 768:
                    comment2 = np.pad(comment2, (0, 768 - comment2.shape[0]), 'constant')
            else:
                return jsonify({'error': 'comment2 contains empty input'}), 400

            # 모델 입력 형태로 변환
            comment1 = comment1.reshape(1, -1)
            comment2 = comment2.reshape(1, -1)

            # 모델에 입력 데이터 전달
            if comment1.shape != (1, 768) or comment2.shape != (1, 768):
                return jsonify({'error': f'Invalid input shape: {comment1.shape}, {comment2.shape}'}), 400

            similarity = model(comment1, comment2).data[0]
            similarities.append(float(similarity))

        # 유사도 리스트 반환
        return jsonify({'similarities': similarities})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)  # 외부에서 접근 가능하도록 설정