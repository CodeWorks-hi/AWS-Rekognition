from flask import Flask, request, jsonify
import boto3
import tempfile
from PIL import Image
from io import BytesIO
import uuid
import os
import csv
from werkzeug.utils import secure_filename

app = Flask(__name__)
rekognition = boto3.client('rekognition', region_name='ap-northeast-2')
bucket = "rekognition-codekookiz"
s3 = boto3.client('s3')


def extract_face_bytes(image_bytes):
    image = Image.open(BytesIO(image_bytes))
    response = rekognition.detect_faces(Image={'Bytes': image_bytes}, Attributes=['DEFAULT'])
    if not response['FaceDetails']:
        return None, 'No face detected'
    box = response['FaceDetails'][0]['BoundingBox']
    width, height = image.size
    left = int(box['Left'] * width)
    top = int(box['Top'] * height)
    right = left + int(box['Width'] * width)
    bottom = top + int(box['Height'] * height)
    face_img = image.crop((left, top, right, bottom))
    if face_img.mode != 'RGB':
        face_img = face_img.convert('RGB')
    buf = BytesIO()
    face_img.save(buf, format='JPEG')
    return buf.getvalue(), None

@app.route('/upload-image', methods=['POST'])
def upload_image():
    image_file = request.files['image']
    image_name = image_file.filename

    try:
        s3.upload_fileobj(image_file, bucket, image_name, ExtraArgs={'ACL': 'public-read'})

        url = f"https://{bucket}.s3.ap-northeast-2.amazonaws.com/{image_name}"
        return jsonify({'message': 'Upload successful', 'url': url})
    except Exception as e:
        return jsonify({'error': str(e)}), 500



# 신규 이미지와 S3 내 기존 이미지 유사도 측정
@app.route('/compare-face-db', methods=['POST'])
def compare_face_db():
    reference_image = request.form['reference_image']
    target_file = request.files['target_image']

    result = None

    with tempfile.NamedTemporaryFile(delete=True) as temp:
        target_file.save(temp.name)
        temp.seek(0)
        image_bytes, err = extract_face_bytes(temp.read())
        if err:
            return jsonify({'error': err}), 400

        try:
            response = rekognition.compare_faces(
                SourceImage={'Bytes': image_bytes},
                TargetImage={'S3Object': {'Bucket': bucket, 'Name': reference_image}}
            )

            if response['FaceMatches']:
                similarity = response['FaceMatches'][0]['Similarity']
                confidence = response['FaceMatches'][0]['Face']['Confidence']
                result = {
                    'reference_image': reference_image,
                    'similarity': similarity,
                    'confidence': confidence
                }
            else:
                result = {
                    'reference_image': reference_image,
                    'similarity': 0,
                    'confidence': 0
                }

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'match': result})


# S3 내 기존 이미지 삭제
@app.route('/delete-image', methods=['POST'])
def delete_image():
    image_name = request.form['image_name']

    try:
        s3.delete_object(Bucket=bucket, Key=image_name)
        return jsonify({'message': f'{image_name} deleted successfully from {bucket}'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# 이미지에 등장하는 각 사람의 감정 분석
@app.route('/analyze-expression', methods=['POST'])
def analyze_expression():
    image_file = request.files['image']
    image_name = image_file.filename

    try:
        # 이미지를 로컬에 저장
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            image_file.save(temp.name)
            image_path = temp.name

        with open(image_path, 'rb') as f:
            image_bytes, err = extract_face_bytes(f.read())
            if err:
                return jsonify({'error': err}), 400

        response = rekognition.detect_faces(
            Image={'Bytes': image_bytes},
            Attributes=['ALL']
        )

        results = []
        for idx, face in enumerate(response['FaceDetails']):
            emotions = sorted(face['Emotions'], key=lambda e: e['Confidence'], reverse=True)
            top_emotion = emotions[0]
            box = face['BoundingBox']
            bounding_box_pixels = {
                'left': int(box['Left'] * 100),
                'top': int(box['Top'] * 100),
                'width': int(box['Width'] * 100),
                'height': int(box['Height'] * 100)
            }

            results.append({
                'face_id': idx + 1,
                'bounding_box': bounding_box_pixels,
                'top_emotion': {
                    'type': top_emotion['Type'],
                    'confidence': top_emotion['Confidence']
                },
                'all_emotions': [
                    {'type': e['Type'], 'confidence': e['Confidence']}
                    for e in emotions
                ]
            })

        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

# 얼굴 유사도 비교 (저장 X)
@app.route('/compare-faces-direct', methods=['POST'])
def compare_faces_direct():
    image1_file = request.files['image1']
    image2_file = request.files['image2']

    result = None

    with tempfile.NamedTemporaryFile(delete=True) as temp1, tempfile.NamedTemporaryFile(delete=True) as temp2:
        image1_file.save(temp1.name)
        temp1.seek(0)
        image1_bytes, err1 = extract_face_bytes(temp1.read())

        image2_file.save(temp2.name)
        temp2.seek(0)
        image2_bytes, err2 = extract_face_bytes(temp2.read())

        if err1 or err2:
            return jsonify({'error': err1 or err2}), 400

        try:
            response = rekognition.compare_faces(
                SourceImage={'Bytes': image1_bytes},
                TargetImage={'Bytes': image2_bytes},
                SimilarityThreshold=0
            )

            if response['FaceMatches']:
                similarity = response['FaceMatches'][0]['Similarity']
                confidence = response['FaceMatches'][0]['Face']['Confidence']
                result = {
                    'similarity': similarity,
                    'confidence': confidence
                }
            else:
                result = {
                    'similarity': None,
                    'confidence': None
                }

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'match': result})


# 사물 분석
@app.route('/detect-labels', methods=['POST'])
def detect_labels():
    image_file = request.files['image']
    with tempfile.NamedTemporaryFile(delete=True) as temp:
        image_file.save(temp.name)
        temp.seek(0)
        image_bytes = temp.read()

        try:
            response = rekognition.detect_labels(Image={'Bytes': image_bytes}, MaxLabels=10, MinConfidence=70)
            labels = [{'label': label['Name'], 'confidence': label['Confidence']} for label in response['Labels']]
            return jsonify({'labels': labels})
        except Exception as e:
            return jsonify({'error': str(e)}), 500


@app.route('/detect-moderation-labels', methods=['POST'])
def detect_moderation_labels():
    image_file = request.files['image']
    with tempfile.NamedTemporaryFile(delete=True) as temp:
        image_file.save(temp.name)
        temp.seek(0)
        image_bytes = temp.read()

        try:
            response = rekognition.detect_moderation_labels(Image={'Bytes': image_bytes})
            labels = [{'label': label['Name'], 'confidence': label['Confidence']} for label in response['ModerationLabels']]
            return jsonify({'moderation_labels': labels})
        except Exception as e:
            return jsonify({'error': str(e)}), 500


@app.route('/detect-text', methods=['POST'])
def detect_text():
    image_file = request.files['image']
    with tempfile.NamedTemporaryFile(delete=True) as temp:
        image_file.save(temp.name)
        temp.seek(0)
        image_bytes = temp.read()

        try:
            response = rekognition.detect_text(Image={'Bytes': image_bytes})
            texts = [{'text': t['DetectedText'], 'confidence': t['Confidence']} for t in response['TextDetections']]
            return jsonify({'text_detections': texts})
        except Exception as e:
            return jsonify({'error': str(e)}), 500


@app.route('/detect-license-plate', methods=['POST'])
def detect_license_plate():
    image_file = request.files['image']
    with tempfile.NamedTemporaryFile(delete=True) as temp:
        image_file.save(temp.name)
        temp.seek(0)
        image_bytes = temp.read()

        try:
            response = rekognition.detect_text(Image={'Bytes': image_bytes})
            plates = [t['DetectedText'] for t in response['TextDetections']
                      if len(t['DetectedText']) >= 6 and any(char.isdigit() for char in t['DetectedText'])]
            return jsonify({'possible_plates': plates})
        except Exception as e:
            return jsonify({'error': str(e)}), 500


@app.route('/detect-id-text', methods=['POST'])
def detect_id_text():
    image_file = request.files['image']
    with tempfile.NamedTemporaryFile(delete=True) as temp:
        image_file.save(temp.name)
        temp.seek(0)
        image_bytes = temp.read()

        try:
            response = rekognition.detect_text(Image={'Bytes': image_bytes})
            text_lines = [t['DetectedText'] for t in response['TextDetections'] if t['Type'] == 'LINE']
            id_related = [text for text in text_lines if any(kw in text for kw in ['이름', '성명', '주민등록번호', '-', '생년월일', 'ID', 'Name', 'Date'])]
            return jsonify({'id_text_candidates': id_related})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        
# 얼굴 인증을 위한 새로운 엔드포인트 추가
@app.route('/verify-face', methods=['POST'])
def verify_face():
    target_file = request.files['image']

    with tempfile.NamedTemporaryFile(delete=True) as temp:
        target_file.save(temp.name)
        temp.seek(0)
        image_bytes, err = extract_face_bytes(temp.read())
        if err:
            return jsonify({'error': err}), 400

        try:
            objects = s3.list_objects_v2(Bucket=bucket)
            if 'Contents' not in objects:
                return jsonify({'verified': False, 'message': '버킷이 비어있거나 접근할 수 없습니다.'})
            for obj in objects['Contents']:
                image = obj['Key']
                try:
                    response = rekognition.compare_faces(
                        SourceImage={'Bytes': image_bytes},
                        TargetImage={'S3Object': {'Bucket': bucket, 'Name': image}},
                        SimilarityThreshold=95
                    )
                    for face_match in response['FaceMatches']:
                        similarity = face_match['Similarity']
                        if similarity >= 95:
                            return jsonify({
                                'verified': True,
                                'matched_image': image,
                                'similarity': similarity,
                                'message': '인증되었습니다.'
                            })
                except Exception:
                    continue
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'verified': False, 'message': '인증에 실패하였습니다.'})



# New endpoint for extracting face
@app.route('/extract-face', methods=['POST'])
def extract_face():
    image_file = request.files['image']
    image_name = image_file.filename
    face_only_name = f"face_only_{image_name}"

    try:
        # 원본 이미지 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp:
            image_file.save(temp.name)
            image_path = temp.name

        # Rekognition으로 얼굴 위치 감지
        with open(image_path, 'rb') as f:
            image_bytes = f.read()

        response = rekognition.detect_faces(
            Image={'Bytes': image_bytes},
            Attributes=['DEFAULT']
        )

        face_details = response.get('FaceDetails', [])
        if not face_details:
            return jsonify({'error': 'No face detected'}), 400

        box = face_details[0]['BoundingBox']

        # 이미지 크기 얻기
        with Image.open(image_path) as img:
            width, height = img.size
            left = int(box['Left'] * width)
            top = int(box['Top'] * height)
            right = left + int(box['Width'] * width)
            bottom = top + int(box['Height'] * height)

            face_img = img.crop((left, top, right, bottom))
            if face_img.mode != 'RGB':
                face_img = face_img.convert('RGB')

            # 얼굴만 저장
            temp_face = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            face_img.save(temp_face.name)

        # S3에 업로드
        with open(temp_face.name, 'rb') as f:
            s3.upload_fileobj(f, bucket, face_only_name, ExtraArgs={'ACL': 'public-read'})

        url = f"https://{bucket}.s3.ap-northeast-2.amazonaws.com/{face_only_name}"
        return jsonify({'message': 'Face extracted and uploaded', 'face_url': url})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

# 환경 설정
COLLECTION_ID = 'codekookiz-face-collection'
CSV_PATH = 'face_metadata.csv'


# CSV 파일 없으면 헤더 생성
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['uuid', 'name', 's3_path'])

@app.route('/upload-face', methods=['POST'])
def upload_face():
    if 'image' not in request.files or 'name' not in request.form:
        return jsonify({'error': 'image file and name required'}), 400

    image_file = request.files['image']
    user_name = request.form['name']

    # UUID 생성 및 S3 키 설정
    uuid_str = str(uuid.uuid4())
    filename = secure_filename(f"{uuid_str}.jpg")
    s3_key = f"faces/{filename}"

    # S3에 이미지 업로드
    try:
        s3.upload_fileobj(image_file, bucket, s3_key, ExtraArgs={'ACL': 'public-read'})
    except Exception as e:
        return jsonify({'error': f"S3 upload failed: {str(e)}"}), 500

    # Rekognition 컬렉션에 얼굴 등록
    try:
        rekognition.index_faces(
            CollectionId=COLLECTION_ID,
            Image={'S3Object': {'Bucket': bucket, 'Name': s3_key}},
            ExternalImageId=uuid_str,
            DetectionAttributes=['DEFAULT']
        )
    except Exception as e:
        return jsonify({'error': f"Rekognition indexing failed: {str(e)}"}), 500

    # 메타데이터 CSV에 기록
    with open(CSV_PATH, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([uuid_str, user_name, f"s3://{bucket}/{s3_key}"])

    return jsonify({
        'message': 'Face registered successfully',
        'uuid': uuid_str,
        'name': user_name,
        's3_path': f"s3://{bucket}/{s3_key}"
    })


@app.route('/verify-in-collection', methods=['POST'])
def verify_in_collection():
    image_file = request.files['image']
    image_bytes, err = extract_face_bytes(image_file.read())
    if err:
        return jsonify({'error': err}), 400

    try:
        response = rekognition.search_faces_by_image(
            CollectionId='codekookiz-face-collection',
            Image={'Bytes': image_bytes},
            FaceMatchThreshold=95,
            MaxFaces=1
        )

        face_matches = response.get('FaceMatches', [])
        if face_matches:
            match = face_matches[0]
            similarity = match['Similarity']
            face_id = match['Face']['FaceId']
            external_id = match['Face'].get('ExternalImageId', 'N/A')

            # CSV에서 이름 찾기
            name = None
            with open('face_metadata.csv', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if row['uuid'] == external_id:
                        name = row['name']
                        break

            return jsonify({
                'verified': True,
                'faceId': face_id,
                'name': name if name else '이름 없음',
                'externalImageId': external_id,
                'similarity': similarity,
                'message': '인증되었습니다.'
            })

        return jsonify({'verified': False, 'message': '인증에 실패하였습니다.'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500



# =============== 연우님 코드 ===============
# 얼굴 감지 및 속성 분석
@app.route('/detect', methods=['POST'])
def detect_faces():
    image_file = request.files['image']
    image_name = image_file.filename

    try:
        # S3에 이미지 업로드
        s3.upload_fileobj(image_file, bucket, image_name, ExtraArgs={'ACL': 'public-read'})

        # Rekognition 얼굴 감지 호출
        response = rekognition.detect_faces(
            Image={'S3Object': {'Bucket': bucket, 'Name': image_name}},
            Attributes=['ALL']
        )

        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# 얼굴 유사도 측정 (80% 이상만 OK)
@app.route('/compare',methods=['POST'])
def compare_faces():
    source_file=request.files.get('source_file')
    target_file=request.files.get('target_file')

    try:
        if source_file is None or target_file is None:
            return jsonify({'error':'두개의 이미지를 업로드 하세요.'})
        
        # 이미지를 Bytes로 읽기
        source_bytes, err1 = extract_face_bytes(source_file.read())
        target_bytes, err2 = extract_face_bytes(target_file.read())
        if err1 or err2:
            return jsonify({'error': err1 or err2}), 400

        # 호출
        response = rekognition.compare_faces(
            SourceImage={'Bytes': source_bytes},
            TargetImage={'Bytes': target_bytes},
            SimilarityThreshold=80
        )

        # 결과 처리
        face_matches = response.get('FaceMatches', [])
        matches_info = [
            {
                'Similarity': match['Similarity'],
                'BoundingBox': match['Face']['BoundingBox']
            }
            for match in face_matches
        ]

        return jsonify({'matches': matches_info})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

# 컬렉션 ID 만들기
def create_collection(collection_id, region):
    # Rekognition 클라이언트 생성
    client = boto3.client('rekognition', region_name=region)
    
    try:
        # 컬렉션 생성 요청
        print(f"Creating collection: {collection_id}")
        response = client.create_collection(CollectionId=collection_id)
        
        # 결과 출력
        print(f"Collection ARN: {response['CollectionArn']}")
        print(f"Status code: {response['StatusCode']}")
        print("Collection created successfully.")
    except client.exceptions.ResourceAlreadyExistsException:
        print(f"Collection '{collection_id}' already exists.")
    except Exception as e:
        print(f"Error creating collection: {str(e)}")

# 컬렉션 ID 및 리전 설정
collection_id = "codekookiz-face-collection"  # 원하는 컬렉션 이름
region = "ap-northeast-2"  # AWS 리전 (예: 서울 리전)

# 컬렉션 생성 호출
create_collection(collection_id, region)


# 얼굴 인덱싱 - 얼굴을 감지하고, 해당 얼굴의 특징을 추출하여 **컬렉션(Collection)**에 저장하는 작업
# 얼굴 데이터를 시스템에 등록하여 인증 시스템 구축 / 여러 사람의 얼굴 데이터를 저장하여 관리
@app.route('/index', methods=['POST'])
def index_faces():
    image_file = request.files['image']
    image_name = image_file.filename
    collection_id = 'codekookiz-face-collection'

    try:
        # 파일을 한 번 읽어 메모리에 저장
        file_bytes = image_file.read()
        image_stream = BytesIO(file_bytes)  # S3 업로드용
        image_for_face = BytesIO(file_bytes)  # Rekognition용

        # S3에 업로드
        s3.upload_fileobj(image_stream, bucket, image_name, ExtraArgs={'ACL': 'public-read'})

        # Rekognition 인덱싱용 얼굴 바이트 추출
        image_bytes, err = extract_face_bytes(image_for_face.read())
        if err:
            return jsonify({'error': err}), 400

        response = rekognition.index_faces(
            CollectionId=collection_id,
            Image={'Bytes': image_bytes},
            ExternalImageId=image_name.split('.')[0],
            DetectionAttributes=['ALL']
        )

        face_records = response.get('FaceRecords', [])
        indexed_faces = [
            {
                'FaceId': record['Face']['FaceId'],
                'BoundingBox': record['Face']['BoundingBox'],
                'Confidence': record['Face']['Confidence']
            }
            for record in face_records
        ]

        return jsonify({'IndexedFaces': indexed_faces})
    except Exception as e:
        return jsonify({'error': str(e)}), 500




# =============== 성현님 코드 ===============
@app.route("/faces", methods=["POST"])
def face_storage():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files["image"]
    image_name = image_file.filename
    collection_id = 'codekookiz-face-collection'

    # ✅ 한 번만 읽고 복사
    file_bytes = image_file.read()
    image_stream = BytesIO(file_bytes)      # S3 업로드용
    image_for_face = BytesIO(file_bytes)    # Rekognition 용

    try:
        s3.upload_fileobj(image_stream, bucket, image_name, ExtraArgs={'ACL': 'public-read'})
    except Exception as e:
        return jsonify({"error": f"Failed to upload image to S3: {str(e)}"}), 500

    try:
        image_bytes, err = extract_face_bytes(image_for_face.read())
        if err:
            return jsonify({'error': err}), 400

        response = rekognition.index_faces(
            CollectionId=collection_id,
            Image={'Bytes': image_bytes},
            ExternalImageId=image_name.split('.')[0],
            DetectionAttributes=['ALL']
        )

        face_records = response.get("FaceRecords", [])
        if not face_records:
            return jsonify({"error": "No face detected or failed to index"}), 400

        face_id = face_records[0]["Face"]["FaceId"]

        return jsonify({
            "message": "Face stored successfully",
            "faceId": face_id
        }), 201

    except Exception as e:
        return jsonify({"error": f"AWS Service Error: {str(e)}"}), 500
    

@app.route('/verify-face-in-collection', methods=['POST'])
def verify_face_in_collection():
    image_file = request.files['image']
    image_bytes, err = extract_face_bytes(image_file.read())
    if err:
        return jsonify({'error': err}), 400

    try:
        response = rekognition.search_faces_by_image(
            CollectionId='codekookiz-face-collection',
            Image={'Bytes': image_bytes},
            FaceMatchThreshold=95,
            MaxFaces=1
        )

        face_matches = response.get('FaceMatches', [])
        if face_matches:
            match = face_matches[0]
            similarity = match['Similarity']
            face_id = match['Face']['FaceId']
            external_id = match['Face'].get('ExternalImageId', 'N/A')

            if similarity >= 95:
                return jsonify({
                    'verified': True,
                    'faceId': face_id,
                    'externalImageId': external_id,
                    'similarity': similarity,
                    'message': '인증되었습니다.'
                })

        return jsonify({'verified': False, 'message': '인증에 실패하였습니다.'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route("/compare-face", methods=["POST"])
def face_compare():
    try:
        if "source" not in request.files or "target" not in request.files:
            return jsonify({"error": "Both source and target images required"}), 400

        source = request.files["source"]
        target = request.files["target"]

        source_bytes, err1 = extract_face_bytes(source.read())
        target_bytes, err2 = extract_face_bytes(target.read())
        if err1 or err2:
            return jsonify({"error": err1 or err2}), 400

        response = rekognition.compare_faces(
            SourceImage={"Bytes": source_bytes},
            TargetImage={"Bytes": target_bytes},
            SimilarityThreshold=80
        )

        matches = [{
            "similarity": match["Similarity"],
            "boundingBox": match["Face"]["BoundingBox"]
        } for match in response.get("FaceMatches", [])]

        return jsonify({
            "matches": matches,
            "unmatchedCount": len(response.get("UnmatchedFaces", []))
        }), 200

    except Exception as e:
        return jsonify({"error": f"Face comparison failed: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True)