from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ultralytics import YOLO


GROUP_MEANINGS_UZ = {
    "I": "Ogohlantiruvchi belgilar",
    "II": "Imtiyoz / ustuvorlik belgilari",
    "III": "Taqiqlovchi belgilar",
    "IV": "Buyuruvchi belgilar",
    "V": "Axborot-ko'rsatish belgilari",
    "VI": "Servis / xizmat ko'rsatish belgilari",
    "VII": "Qo'shimcha axborot tablichkalari",
}


def class_meaning(class_name: str) -> str:
    group = class_name.split("-", 1)[0]
    group_meaning = GROUP_MEANINGS_UZ.get(group, "Noma'lum guruh")
    return f"{group_meaning}, kod: {class_name}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="best.pt YOLO modeli orqali yo'l belgilarini aniqlash."
    )
    parser.add_argument(
        "source",
        help="Rasm, video, kamera indeksi yoki papka yo'li. Masalan: image.jpg yoki 0",
    )
    parser.add_argument("--model", default="best.pt", help="Model fayli yo'li.")
    parser.add_argument("--conf", type=float, default=0.25, help="Ishonch chegarasi.")
    parser.add_argument("--imgsz", type=int, default=640, help="Modelga beriladigan rasm o'lchami.")
    parser.add_argument("--out", default="runs/detect_road_signs", help="Natija papkasi.")
    parser.add_argument("--save-crop", action="store_true", help="Topilgan belgilarni alohida crop qilib saqlash.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.out).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    results = model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        save=True,
        save_txt=True,
        save_conf=True,
        save_crop=args.save_crop,
        project=str(output_dir.parent),
        name=output_dir.name,
        exist_ok=True,
    )

    detections = []
    for result in results:
        image_path = str(result.path)
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            class_id = int(box.cls[0].item())
            class_name = model.names[class_id]
            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
            detections.append(
                {
                    "image": image_path,
                    "class_id": class_id,
                    "class_name": class_name,
                    "meaning_uz": class_meaning(class_name),
                    "confidence": round(confidence, 4),
                    "box_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                }
            )

    json_path = output_dir / "detections.json"
    csv_path = output_dir / "detections.csv"
    json_path.write_text(json.dumps(detections, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["image", "class_id", "class_name", "meaning_uz", "confidence", "box_xyxy"],
        )
        writer.writeheader()
        writer.writerows(detections)

    print(f"Topilgan belgilar soni: {len(detections)}")
    print(f"Chizilgan natijalar papkasi: {output_dir}")
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")
    for item in detections:
        print(
            f"- {item['class_name']} ({item['meaning_uz']}), "
            f"confidence={item['confidence']}, box={item['box_xyxy']}"
        )


if __name__ == "__main__":
    main()
