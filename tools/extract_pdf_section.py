import re
import sys
import argparse
from typing import List, Optional, Tuple

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None  # type: ignore


SECTION_HEADER_PATTERN = re.compile(
    r"(问题\s*[一二三四五六七八九十\d]+|第\s*[一二三四五六七八九十\d]+\s*题|任务\s*[一二三四五六七八九十\d]+)")


def extract_text_from_pdf(file_path: str) -> str:
    if PdfReader is None:
        raise RuntimeError("pypdf is not installed. Please install with: pip install pypdf")
    reader = PdfReader(file_path)
    texts: List[str] = []
    for idx, page in enumerate(reader.pages):
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        # Add a clear page delimiter to help regex across pages
        texts.append(f"\n\n[[PAGE_{idx+1}]]\n\n" + page_text)
    return "".join(texts)


def parse_headers_positions(text: str) -> List[Tuple[int, int, str]]:
    matches = []
    for m in SECTION_HEADER_PATTERN.finditer(text):
        matches.append((m.start(), m.end(), m.group(0)))
    return matches


def is_number_three(token: str) -> bool:
    token = token.strip()
    # quick checks
    if re.search(r"三|\b3\b", token):
        return True
    # handle Chinese numerals like 第三题 / 问题三 / 任务三
    return False


def find_section_range(text: str) -> Optional[Tuple[int, int, str]]:
    headers = parse_headers_positions(text)
    if not headers:
        return None

    # Find the header that denotes question 3
    start_idx = None
    start_header_text = ""
    for i, (s, e, htext) in enumerate(headers):
        if is_number_three(htext):
            start_idx = i
            start_header_text = htext
            break

    if start_idx is None:
        # Fallback: look for explicit keywords
        fallback = re.search(r"(问题\s*[3三]|第\s*[3三]\s*题|任务\s*[3三])", text)
        if fallback:
            s = fallback.start()
            # Find the next header after this fallback
            next_start = len(text)
            for (hs, he, _) in headers:
                if hs > s:
                    next_start = hs
                    break
            return (s, next_start, fallback.group(0))
        return None

    # Determine the end boundary as the next header after start
    start_pos = headers[start_idx][0]
    if start_idx + 1 < len(headers):
        end_pos = headers[start_idx + 1][0]
    else:
        end_pos = len(text)

    return (start_pos, end_pos, start_header_text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract the section for 问题3 from a PDF.")
    parser.add_argument("--file", "-f", required=True, help="Path to the PDF file")
    parser.add_argument("--context", type=int, default=0, help="Extra characters of context before and after the section")
    args = parser.parse_args()

    try:
        full_text = extract_text_from_pdf(args.file)
    except Exception as exc:
        print(f"[ERROR] Failed to read PDF: {exc}", file=sys.stderr)
        sys.exit(2)

    result = find_section_range(full_text)
    if result is None:
        print("[WARN] 未找到与‘问题3’匹配的章节标题。尝试在全文中搜索关键词：问题三/问题3/第3题/任务三。\n")
        # As last resort, dump a few lines around likely markers
        candidates = list(re.finditer(r"问题\s*[3三]|第\s*[3三]\s*题|任务\s*[3三]", full_text))
        if not candidates:
            print("[FAIL] 未能在 PDF 中定位‘问题3’文本。")
            sys.exit(1)
        m = candidates[0]
        start = max(0, m.start() - 300)
        end = min(len(full_text), m.end() + 1500)
        snippet = full_text[start:end]
        print(snippet)
        sys.exit(0)

    start_pos, end_pos, header_text = result
    start_pos = max(0, start_pos - args.context)
    end_pos = min(len(full_text), end_pos + args.context)

    section_text = full_text[start_pos:end_pos].strip()

    print("===== 抽取到的章节标题 =====")
    print(header_text)
    print("===== 章节内容（截取） =====")
    print(section_text)


if __name__ == "__main__":
    main() 