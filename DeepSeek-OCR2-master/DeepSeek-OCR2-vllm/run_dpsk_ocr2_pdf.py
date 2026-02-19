import os
import fitz
import img2pdf
import io
import re
from tqdm import tqdm
import torch
from concurrent.futures import ThreadPoolExecutor

if torch.version.cuda == '11.8':
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"
os.environ['VLLM_USE_V1'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from config import MODEL_PATH, PROMPT, SKIP_REPEAT, MAX_CONCURRENCY, NUM_WORKERS, CROP_MODE

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from deepseek_ocr2 import DeepseekOCR2ForCausalLM

from vllm.model_executor.models.registry import ModelRegistry
from vllm import LLM, SamplingParams
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from process.image_process import DeepseekOCR2Processor

# 전역 변수 (한 번만 로드)
llm = None
sampling_params = None
processor = None


def init_model():
    """모델 초기화 (한 번만 호출)"""
    global llm, sampling_params, processor
    
    if llm is not None:
        return  # 이미 로드됨
    
    ModelRegistry.register_model("DeepseekOCR2ForCausalLM", DeepseekOCR2ForCausalLM)
    
    llm = LLM(
        model=MODEL_PATH,
        hf_overrides={"architectures": ["DeepseekOCR2ForCausalLM"]},
        block_size=256,
        enforce_eager=False,
        trust_remote_code=True,
        max_model_len=8192,
        swap_space=0,
        max_num_seqs=MAX_CONCURRENCY,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        disable_mm_preprocessor_cache=True
    )
    
    logits_processors = [
        NoRepeatNGramLogitsProcessor(
            ngram_size=20,
            window_size=50,
            whitelist_token_ids={128821, 128822}
        )
    ]
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=8192,
        logits_processors=logits_processors,
        skip_special_tokens=False,
        include_stop_str_in_output=True,
    )
    
    processor = DeepseekOCR2Processor()
    print("✅ 모델 로드 완료")


def select_page_indices(total_pages, head_pages=3, tail_pages=50):
    """처리할 페이지 인덱스(0-based) 계산"""
    if total_pages <= tail_pages:
        return list(range(total_pages))
    head = list(range(min(head_pages, total_pages)))
    tail_start = max(0, total_pages - tail_pages)
    tail = list(range(tail_start, total_pages))
    return sorted(set(head + tail))


def pdf_to_images_high_quality(pdf_input, dpi=144, image_format="PNG", page_indices=None):
    """PDF를 이미지로 변환 (파일 경로 또는 bytes 지원)"""
    images = []
    
    if isinstance(pdf_input, bytes):
        pdf_document = fitz.open(stream=pdf_input, filetype="pdf")
    else:
        pdf_document = fitz.open(pdf_input)

    total_pages = pdf_document.page_count
    selected_page_indices = (
        list(range(total_pages))
        if page_indices is None
        else sorted(set(i for i in page_indices if 0 <= i < total_pages))
    )
    
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    
    for page_num in selected_page_indices:
        page = pdf_document[page_num]
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        Image.MAX_IMAGE_PIXELS = None
        
        img_data = pixmap.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        
        if image_format.upper() != "PNG" and img.mode in ('RGBA', 'LA'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
            img = background
        
        images.append(img)
    
    pdf_document.close()
    return images, total_pages, selected_page_indices


def pil_to_pdf_img2pdf(pil_images, output_path):
    if not pil_images:
        return
    
    image_bytes_list = []
    for img in pil_images:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='JPEG', quality=95)
        image_bytes_list.append(img_buffer.getvalue())
    
    try:
        pdf_bytes = img2pdf.convert(image_bytes_list)
        with open(output_path, "wb") as f:
            f.write(pdf_bytes)
    except Exception as e:
        print(f"error: {e}")


def re_match(text):
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)
    
    matches_image = []
    matches_other = []
    for a_match in matches:
        if '<|ref|>image<|/ref|>' in a_match[0]:
            matches_image.append(a_match[0])
        else:
            matches_other.append(a_match[0])
    return matches, matches_image, matches_other


def extract_coordinates_and_label(ref_text, image_width, image_height):
    try:
        label_type = ref_text[1]
        cor_list = eval(ref_text[2])
    except Exception as e:
        print(e)
        return None
    return (label_type, cor_list)


def draw_bounding_boxes(image, refs, jdx, output_path):
    image_width, image_height = image.size
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    
    overlay = Image.new('RGBA', img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    
    img_idx = 0
    
    for i, ref in enumerate(refs):
        try:
            result = extract_coordinates_and_label(ref, image_width, image_height)
            if result:
                label_type, points_list = result
                color = (np.random.randint(0, 200), np.random.randint(0, 200), np.random.randint(0, 255))
                color_a = color + (20,)
                
                for points in points_list:
                    x1, y1, x2, y2 = points
                    x1 = int(x1 / 999 * image_width)
                    y1 = int(y1 / 999 * image_height)
                    x2 = int(x2 / 999 * image_width)
                    y2 = int(y2 / 999 * image_height)
                    
                    if label_type == 'image':
                        try:
                            cropped = image.crop((x1, y1, x2, y2))
                            cropped.save(f"{output_path}/images/{jdx}_{img_idx}.jpg")
                        except Exception as e:
                            print(e)
                        img_idx += 1
                    
                    try:
                        width = 4 if label_type == 'title' else 2
                        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
                        draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)
                        
                        text_x = x1
                        text_y = max(0, y1 - 15)
                        text_bbox = draw.textbbox((0, 0), label_type, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                        draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height],
                                      fill=(255, 255, 255, 30))
                        draw.text((text_x, text_y), label_type, font=font, fill=color)
                    except:
                        pass
        except:
            continue
    
    img_draw.paste(overlay, (0, 0), overlay)
    return img_draw


def process_single_image(image):
    """단일 이미지 전처리"""
    cache_item = {
        "prompt": PROMPT,
        "multi_modal_data": {
            "image": processor.tokenize_with_images(
                images=[image],
                bos=True,
                eos=True,
                cropping=CROP_MODE
            )
        },
    }
    return cache_item


def run_ocr(pdf_input, output_path, filename="output"):
    """
    OCR 실행 메인 함수
    
    Args:
        pdf_input: PDF 파일 경로 또는 bytes
        output_path: 출력 디렉토리
        filename: 출력 파일명 (확장자 제외)
    
    Returns:
        dict: 결과 파일 경로들
    """
    global llm, sampling_params, processor
    
    # 모델 초기화 확인
    if llm is None:
        init_model()
    
    # 출력 디렉토리 생성
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(f'{output_path}/images', exist_ok=True)
    
    # PDF -> 이미지 변환 (50p 이하 전체, 초과 시 앞 3p + 뒤 50p)
    print("PDF loading...")
    if isinstance(pdf_input, bytes):
        tmp_doc = fitz.open(stream=pdf_input, filetype="pdf")
    else:
        tmp_doc = fitz.open(pdf_input)
    original_total_pages = tmp_doc.page_count
    tmp_doc.close()

    selected_page_indices = select_page_indices(original_total_pages, head_pages=3, tail_pages=50)
    images, _, selected_page_indices = pdf_to_images_high_quality(
        pdf_input,
        page_indices=selected_page_indices,
    )
    
    # 배치 입력 생성
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        batch_inputs = list(tqdm(
            executor.map(process_single_image, images),
            total=len(images),
            desc="Pre-processed images"
        ))
    
    # OCR 실행
    outputs_list = llm.generate(batch_inputs, sampling_params=sampling_params)
    
    # 출력 파일 경로
    mmd_det_path = f"{output_path}/{filename}_det.mmd"
    mmd_path = f"{output_path}/{filename}.mmd"
    pdf_out_path = f"{output_path}/{filename}_layouts.pdf"
    
    contents_det = ''
    contents = ''
    draw_images = []
    jdx = 0
    
    for output, img, original_page_idx in zip(outputs_list, images, selected_page_indices):
        content = output.outputs[0].text
        
        if '<｜end▁of▁sentence｜>' in content:
            content = content.replace('<｜end▁of▁sentence｜>', '')
        else:
            if SKIP_REPEAT:
                continue
        
        page_num = f'\n<--- Page {original_page_idx + 1} Split --->'
        contents_det += content + f'\n{page_num}\n'
        
        image_draw = img.copy()
        matches_ref, matches_images, matches_other = re_match(content)
        result_image = draw_bounding_boxes(image_draw, matches_ref, jdx, output_path)
        draw_images.append(result_image)
        
        for idx, a_match_image in enumerate(matches_images):
            content = content.replace(a_match_image, f'![](images/{jdx}_{idx}.jpg)\n')
        
        for idx, a_match_other in enumerate(matches_other):
            content = content.replace(a_match_other, '') \
                .replace('\\coloneqq', ':=') \
                .replace('\\eqqcolon', '=:') \
                .replace('\n\n\n\n', '\n\n') \
                .replace('\n\n\n', '\n\n')
        
        contents += content + f'\n{page_num}\n'
        jdx += 1
    
    # 파일 저장
    with open(mmd_det_path, 'w', encoding='utf-8') as f:
        f.write(contents_det)
    
    with open(mmd_path, 'w', encoding='utf-8') as f:
        f.write(contents)
    
    pil_to_pdf_img2pdf(draw_images, pdf_out_path)
    
    return {
        "mmd_path": mmd_path,
        "mmd_det_path": mmd_det_path,
        "layout_pdf_path": pdf_out_path,
        "images_dir": f"{output_path}/images",
        "total_pages": original_total_pages,
        "processed_page_count": len(images),
        "processed_pages": [i + 1 for i in selected_page_indices],
    }


# CLI로 직접 실행할 때
if __name__ == "__main__":
    from config import INPUT_PATH, OUTPUT_PATH
    
    init_model()
    filename = INPUT_PATH.split('/')[-1].replace('.pdf', '')
    result = run_ocr(INPUT_PATH, OUTPUT_PATH, filename)
    print(f"완료: {result}")