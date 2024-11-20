from flask_cors import CORS
import torch
from pyannote.audio import Model, Pipeline
from pyannote.core import Annotation
from flask import Flask, request, jsonify, render_template
import os
import subprocess
import sys
from datasets import load_dataset, DatasetDict
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import librosa
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
# .env 파일 로드
load_dotenv()

app = Flask(__name__)
CORS(app)  # 모든 라우트에 CORS 적용



# 화자 분리 모델 경로 설정
finetuned_model_path = os.getenv("FINETUNED_MODEL_PATH")
segmentation_model = Model.from_pretrained("pyannote/segmentation-3.0").to(torch.device("cuda"))
checkpoint = torch.load(finetuned_model_path, map_location="cuda")
segmentation_model.load_state_dict(checkpoint, strict=False)

# 화자 분리 파이프라인 설정
finetuned_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization").to(torch.device("cuda"))
finetuned_pipeline.segmentation = segmentation_model
finetuned_pipeline.segmentation.threshold = 0.6292053016753839
finetuned_pipeline.segmentation.min_duration_off = 0.52173913046875
finetuned_pipeline.clustering.method = "centroid"
finetuned_pipeline.clustering.min_cluster_size = 15
finetuned_pipeline.clustering.threshold = 0.7601635586431199

# Whisper 모델 설정
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id = 'pdh0184/whisper_Large_ko_LoRA_V2'
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)
processor = AutoProcessor.from_pretrained(model_id)
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

# OpenAI 설정
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# 화자 분리 및 텍스트 추출 함수
def diarize_and_transcribe_combined(audio_file_path: str, segment_duration: float = 30.0):
    """
    긴 화자의 발언 구간을 Whisper가 처리할 수 있는 길이(segment_duration)로 나누어 순차적으로 변환합니다.
    """
    # Diarization을 통한 화자 구간 추출
    single_file = {"uri": "unique_identifier", "audio": audio_file_path}
    diarization_result = finetuned_pipeline(single_file)

    labeled_segments = []
    current_speaker = None
    combined_audio = []
    start_time = None

    for speech_turn, _, speaker in diarization_result.itertracks(yield_label=True):
        # 동일 화자 구간을 결합
        if speaker == current_speaker:
            segment_audio, _ = librosa.load(audio_file_path, sr=None, offset=speech_turn.start, duration=speech_turn.end - speech_turn.start)
            combined_audio.extend(segment_audio)
        else:
            # 현재 화자 발언을 Whisper로 변환
            if combined_audio:
                # 긴 오디오 구간을 Whisper의 최대 처리 길이로 나눠서 순차적으로 처리
                audio_sample_rate = 16000  # Whisper가 사용하는 샘플링 레이트
                total_duration = len(combined_audio) / audio_sample_rate
                offset = 0

                # 긴 구간을 segment_duration 단위로 나눠서 처리
                while offset < total_duration:
                    end_offset = min(offset + segment_duration, total_duration)
                    segment = combined_audio[int(offset * audio_sample_rate):int(end_offset * audio_sample_rate)]
                    sample = dict({'path': audio_file_path, 'array': np.array(segment), 'sampling_rate': audio_sample_rate})
                    transcription_result = asr_pipeline(sample)
                    transcription = transcription_result.get("text", "").strip() if transcription_result else ""
                    
                    # 텍스트 저장
                    labeled_segments.append({
                        "speaker": current_speaker,
                        "start": start_time + offset,
                        "end": start_time + end_offset,
                        "transcription": transcription
                    })
                    offset += segment_duration

            # 새로운 화자로 전환
            current_speaker = speaker
            start_time = speech_turn.start
            combined_audio = []
            segment_audio, _ = librosa.load(audio_file_path, sr=None, offset=speech_turn.start, duration=speech_turn.end - speech_turn.start)
            combined_audio.extend(segment_audio)

    # 마지막 화자에 대해 변환 수행
    if combined_audio:
        total_duration = len(combined_audio) / 16000
        offset = 0

        while offset < total_duration:
            end_offset = min(offset + segment_duration, total_duration)
            segment = combined_audio[int(offset * 16000):int(end_offset * 16000)]
            sample = dict({'path': audio_file_path, 'array': np.array(segment), 'sampling_rate': 16000})
            transcription_result = asr_pipeline(sample)
            transcription = transcription_result.get("text", "").strip() if transcription_result else ""
            
            labeled_segments.append({
                "speaker": current_speaker,
                "start": start_time + offset,
                "end": start_time + end_offset,
                "transcription": transcription
            })
            offset += segment_duration

    print("Diarization result:", diarization_result)
    print("Transcription:", [seg['transcription'] for seg in labeled_segments])
    return labeled_segments



# 요약 생성 함수
def summarize_text(labeled_segments):
    dialogue = "\n".join([
        f"{segment['speaker']} [{segment['start']:.3f} --> {segment['end']:.3f}]: {segment['transcription']}" 
        for segment in labeled_segments
    ])
    
    # 원샷 예제 포함
    example_dialogue = (
        "SPEAKER_01 : 아 그러면 혹시 그 기획조정실장을 상대로 지리하실 의원이 계십니까? "
        "네. 더 이상 지리하실 회원님이 없으심으로 서울특별시 교육감 소속 지방공문 정원졸에 일부 세력 세력 세력 세력.\n\n"

        "SPEAKER_01 [00:00:00.008 --> 00:00:07.960]: 아 그러면 혹시 그 기획조정실장을 상대로 질의하실 의원님 계십니까? "
        "네. 더 이상 질의하실 의원님이 없으심으로 서울특별시 교육감 소속 지방공문 정원조례 일부 세력."
        
        "요약 : 기획조정실장을 상대로 질의할 의원의 유무에 대해서 묻고 있다."
    )
    
    messages = [
        {'role': 'system', 'content': (
            "Transcribe the following dialogue in Korean as it appears, including start and end times for each speaker's statement. "
            "Each speaker's identifier must be reassigned sequentially, starting from SPEAKER_01 (e.g., SPEAKER_01, SPEAKER_02, etc.). "
            "Ensure there are no gaps or skipped numbers in the sequence. "
            "Label each speaker's statement in the format '[start_time --> end_time]'. "
            "Correct obvious phonetic errors only if necessary to ensure readability, "
            "but strictly preserve the original tone, terminology, and wording.\n\n"
            "After the transcription, provide a single summary sentence starting with '요약 :' that captures the main focus or context of the conversation.\n\n"
            f"Example:\n{example_dialogue}"
        )},
        {'role': 'user', 'content': dialogue}
    ]
    summary = client.chat.completions.create(model="gpt-4o", messages=messages)
    return summary.choices[0].message.content
# 루트 경로에서 HTML 파일 렌더링
@app.route('/')
def index():
    return render_template('temp.html')

# 오디오 파일을 분석하여 화자 분리, 텍스트 변환, 라벨링 및 요약 결과 제공
@app.route('/process_audio', methods=['POST'])
def process_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['audio']
    # 저장 경로를 변경
    upload_folder = './uploaded_files'
    os.makedirs(upload_folder, exist_ok=True)  # 디렉토리가 없으면 생성

    file_path = os.path.join(upload_folder, file.filename)
    file.save(file_path)  # 파일 저장

    labeled_segments = diarize_and_transcribe_combined(file_path)
    summary = summarize_text(labeled_segments)

    return jsonify({
        "summary": summary
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
