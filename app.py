from flask import Flask, request, jsonify
import boto3
import tempfile
from PIL import Image

app = Flask(__name__)
rekognition = boto3.client('rekognition', region_name='ap-northeast-2')
bucket = "rekognition-codekookiz"
s3 = boto3.client('s3')


# S3에 신규 이미지 추가
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
        image_bytes = temp.read()

        try:
            response = rekognition.compare_faces(
                SourceImage={'Bytes': image_bytes},
                TargetImage={'S3Object': {'Bucket': bucket, 'Name': reference_image}}
            )

            if response['FaceMatches']:
                similarity = response['FaceMatches'][0]['Similarity']
                result = {
                    'reference_image': reference_image,
                    'similarity': similarity
                }
            else:
                result = {
                    'reference_image': reference_image,
                    'similarity': 0
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
        # 이미지를 로컬에 저장해서 크기 계산
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            image_file.save(temp.name)
            image_path = temp.name
            with Image.open(image_path) as img:
                img_width, img_height = img.size

        with open(image_path, 'rb') as f:
            image_bytes = f.read()

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
                'left': int(box['Left'] * img_width),
                'top': int(box['Top'] * img_height),
                'width': int(box['Width'] * img_width),
                'height': int(box['Height'] * img_height)
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
        image1_bytes = temp1.read()

        image2_file.save(temp2.name)
        temp2.seek(0)
        image2_bytes = temp2.read()

        try:
            response = rekognition.compare_faces(
                SourceImage={'Bytes': image1_bytes},
                TargetImage={'Bytes': image2_bytes},
                SimilarityThreshold=0
            )

            if response['FaceMatches']:
                similarity = response['FaceMatches'][0]['Similarity']
                result = {
                    'similarity': similarity
                }
            else:
                result = {
                    'similarity': None
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

    # 이미지 파일을 임시 디렉토리에 저장하고 바이트로 읽기
    with tempfile.NamedTemporaryFile(delete=True) as temp:
        target_file.save(temp.name)
        temp.seek(0)
        image_bytes = temp.read()

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
    return jsonify({
        'verified': False,
        'message': '인증에 실패하였습니다.'
    })



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

# =============== 연우님 코드 ===============
from io import BytesIO

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
        
        # 이미지를 Bytes 형식으로 변환
        source_bytes=BytesIO(source_file.read())
        target_bytes=BytesIO(target_file.read())

        # 호출
        response = rekognition.compare_faces(
            SourceImage={'Bytes': source_bytes.read()},
            TargetImage={'Bytes': target_bytes.read()},
            SimilarityThreshold=95
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
@app.route('/index',methods=['POST'])
def index_faces():
    # 클라이언트에서 이미지 파일 가져오기
    image_file=request.files['image']
    image_name=image_file.filename
    collection_id='codekookiz-face-collection' # 컬렉션 ID(없으면 생성)

    try:
        # S3에 이미지 업로드
        s3.upload_fileobj(image_file, bucket, image_name, ExtraArgs={'ACL': 'public-read'})

        # Rekognition 얼굴 인덱싱 호출
        response = rekognition.index_faces(
            CollectionId=collection_id,
            Image={'S3Object': {'Bucket': bucket, 'Name': image_name}},
            ExternalImageId=image_name.split('.')[0],  # 외부 이미지 ID (고유값)
            DetectionAttributes=['ALL']
        )

        # 결과 반환
        face_records = response.get('FaceRecords', [])
        indexed_faces = [
            {
                'FaceId': record['Face']['FaceId'],
                'BoundingBox': record['Face']['BoundingBox'],
                'Confidence': record['Face']['Confidence']
            }
            for record in face_records
        ]
        
        return jsonify({'IndexedFaces': indexed_faces}) # 고유 ID로 나옴
    except Exception as e:
        return jsonify({'error': str(e)}), 500




# =============== 성현님 코드 ===============
from botocore.exceptions import NoCredentialsError, ClientError

@app.route("/faces", methods=["POST"])
def face_storage():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files["image"]
    image_name = image_file.filename
    collection_id = 'codekookiz-face-collection'

    try:
        # S3에 이미지 업로드
        s3.upload_fileobj(image_file, bucket, image_name, ExtraArgs={'ACL': 'public-read'})
    except Exception as e:
        return jsonify({"error": f"Failed to upload image to S3: {str(e)}"}), 500

    try:
        # Rekognition에 얼굴 등록
        response = rekognition.index_faces(
            CollectionId=collection_id,
            Image={'S3Object': {'Bucket': bucket, 'Name': image_name}},
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

    except NoCredentialsError:
        return jsonify({"error": "AWS credentials not configured"}), 500

    except ClientError as e:
        return jsonify({"error": f"AWS Service Error: {str(e)}"}), 500
    

@app.route('/verify-face-in-collection', methods=['POST'])
def verify_face_in_collection():
    image_file = request.files['image']
    image_bytes = image_file.read()

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

        return jsonify({
            'verified': False,
            'message': '인증에 실패하였습니다.'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route("/compare-face", methods=["POST"])
def face_compare():
    try:
        if "source" not in request.files or "target" not in request.files:
            return jsonify({"error": "Both source and target images required"}), 400

        source = request.files["source"]
        target = request.files["target"]

        response = rekognition.compare_faces(
            SourceImage={"Bytes": source.read()},
            TargetImage={"Bytes": target.read()},
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

    except ClientError as e:
        return jsonify({"error": f"Face comparison failed: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True)