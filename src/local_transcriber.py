import os
import time
import datetime
import subprocess
import argparse
from pathlib import Path
from faster_whisper import WhisperModel

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Определение динамических путей относительно папки src
SRC_DIR = Path(__file__).parent
BASE_DIR = SRC_DIR.parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
TEMP_DIR = BASE_DIR / "temp"

# Гарантируем существование нужных папок
for directory in [INPUT_DIR, OUTPUT_DIR, TEMP_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

def extract_audio(video_path: Path, audio_path: Path) -> bool:
    """Извлекает аудио из видеофайла и сжимает для экономии ОЗУ."""
    print(f"\n[~] Извлечение аудио из видео: {video_path.name}...")
    command = [
        'ffmpeg', '-y', '-i', str(video_path),
        '-vn', '-ac', '1', '-ar', '16000', '-q:a', '0', str(audio_path)
    ]
    try:
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, check=True)
        print(f"[+] Аудио подготовлено: {audio_path.name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[-] Ошибка при извлечении аудио: {e}")
        return False
    except FileNotFoundError:
        print("[-] FFmpeg не найден в системе. Проверьте установку.")
        return False

def process_lecture_pipeline(video_path: Path, model_size: str, lang: str, translate: bool):
    original_name = video_path.stem 
    temp_audio_path = TEMP_DIR / f"temp_audio_{original_name}.wav"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = OUTPUT_DIR / f"lecture_{original_name}_{timestamp}.md"

    if not extract_audio(video_path, temp_audio_path):
        return

    print(f"\n[~] Загрузка локальной модели Whisper ({model_size})...")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    
    task = "translate" if translate else "transcribe"
    display_lang = lang if lang else "Автоопределение"
    print(f"[~] Режим: {task.upper()} | Язык исходника: {display_lang} | Обработка...")
    
    start_time = time.time()
    try:
        segments, info = model.transcribe(
            str(temp_audio_path), 
            beam_size=5, 
            language=lang,
            task=task
        )
        
        with open(output_filename, "w", encoding="utf-8") as f:
            # Адаптируем промпт для нейросети в зависимости от режима перевода
            if translate:
                f.write("Act as a professional editor. Below is a raw audio transcript translated into English. ")
                f.write("Create a structured summary: A brief overview in 3 sentences, followed by Key Takeaways as a bulleted list. ")
                f.write("If there are specific tasks or agreements, highlight them in an 'Action Items' section.\n\n")
                f.write(f"# English Translation: {original_name}\n\n")
            else:
                f.write("Действуй как профессиональный редактор. Ниже приведена сырая расшифровка аудио. ")
                f.write("Сделай структурированное саммари: Краткая суть в 3 предложениях, затем Ключевые тезисы списком. ")
                f.write("Если в тексте есть конкретные задачи или договоренности - выдели их в отдельный блок 'Action Items'.\n\n")
                f.write(f"# Транскрипция: {original_name}\n\n")
            
            for segment in segments:
                start_fmt = time.strftime('%H:%M:%S', time.gmtime(segment.start))
                end_fmt = time.strftime('%H:%M:%S', time.gmtime(segment.end))
                
                f.write(f"**[{start_fmt} - {end_fmt}]** {segment.text}\n")
                print(f"\r[>] Обработано аудио до: {end_fmt} ...", end="", flush=True)
                
        print(f"\n[+] Завершено за {(time.time() - start_time) / 60:.2f} минут.")
        print(f"[+] Файл сохранен: {output_filename.resolve()}")

    finally:
        if temp_audio_path.exists():
            try:
                temp_audio_path.unlink()
                print(f"[~] Временный файл удален.")
            except Exception as e:
                print(f"[-] Не удалось удалить временный файл: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Пакетная транскрибация лекций из папки input.")
    parser.add_argument("--model", default="small", help="Размер модели Whisper (base, small, medium)")
    parser.add_argument("--lang", default=None, help="Код языка оригинала (ru, en, es). Пусто для автоопределения.")
    parser.add_argument("--translate", action="store_true", help="Перевести на английский язык")
    
    args = parser.parse_args()

    supported_extensions = {".mp4", ".wav", ".mp3", ".mkv", ".m4a"}

    video_files = [
        f for f in INPUT_DIR.iterdir() 
        if f.is_file() and f.suffix.lower() in supported_extensions
    ]
    
    if not video_files:
        print(f"[-] В папке {INPUT_DIR.resolve()} не найдено видеофайлов ({', '.join(supported_extensions)}).")
    else:
        print(f"\n[+] Найдено файлов для пакетной обработки: {len(video_files)}")
        
        for i, video_file in enumerate(video_files, 1):
            print(f"\n{'='*60}")
            print(f"[~] Файл {i} из {len(video_files)}: {video_file.name}")
            print(f"{'='*60}")
            
            process_lecture_pipeline(video_file, model_size=args.model, lang=args.lang, translate=args.translate)
            
        print("\n[+] Пакетная очередь пуста. Все лекции успешно обработаны!")   