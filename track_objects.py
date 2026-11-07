import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# Mapeamento de tipos de tracker para suas respectivas funções de criação
def _tracker_factory(name: str):
    """Return cv2.Tracker<name>_create, falling back to cv2.legacy (OpenCV >= 4.5.1)."""
    attr = f"Tracker{name}_create"
    if hasattr(cv2, attr):
        return getattr(cv2, attr)
    legacy = getattr(cv2, "legacy", None)
    if legacy is not None and hasattr(legacy, attr):
        return getattr(legacy, attr)
    return None


TRACKER_CREATORS = {
    name: factory
    for name in ("CSRT", "KCF", "MOSSE", "MIL", "MedianFlow", "TLD", "Boosting")
    if (factory := _tracker_factory(name)) is not None
}


def create_tracker(tracker_type: str):
    """Cria um tracker do OpenCV baseado no tipo especificado."""
    try:
        return TRACKER_CREATORS[tracker_type]()
    except KeyError:
        raise ValueError(f"Tipo de tracker inválido: {tracker_type}")


def define_output_path(input_path: str, output_path: str) -> str:
    """Gera o caminho do arquivo de saída com base no input fornecido."""
    input_path = Path(input_path)
    if not output_path or output_path.strip() == "":
        return str(input_path.parent / f"{input_path.stem}_proc{input_path.suffix}")
    output_path = Path(output_path)
    if output_path.is_dir():
        return str(output_path / f"{input_path.stem}_proc{input_path.suffix}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return str(output_path)


def bb_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    """Calcula o Intersection over Union (IoU) entre duas bounding boxes."""
    x_a1, y_a1, w_a, h_a = box_a
    x_b1, y_b1, w_b, h_b = box_b

    x_a2, y_a2 = x_a1 + w_a, y_a1 + h_a
    x_b2, y_b2 = x_b1 + w_b, y_b1 + h_b

    x_i1, y_i1 = max(x_a1, x_b1), max(y_a1, y_b1)
    x_i2, y_i2 = min(x_a2, x_b2), min(y_a2, y_b2)
    inter_area = max(0, x_i2 - x_i1) * max(0, y_i2 - y_i1)

    box_a_area = w_a * h_a
    box_b_area = w_b * h_b
    return inter_area / float(box_a_area + box_b_area - inter_area + 1e-5)


def update_tracker_predictions(trackers: dict, frame: np.ndarray) -> dict:
    """Atualiza os trackers com o frame atual e retorna a predição da bbox."""
    for obj_id, info in trackers.items():
        success, pred_bbox = info["tracker"].update(frame)
        info["pred_bbox"] = pred_bbox if success else info["bbox"]
    return trackers


def create_new_tracker(
    updated_trackers: dict,
    frame: np.ndarray,
    tracker_type: str,
    bbox: tuple[int, int, int, int],
    next_id: int,
    detection_associations: list,
    used_trackers: set,
) -> int:
    """Fabrica um novo tracker e atualiza as estruturas correspondentes."""
    tracker = create_tracker(tracker_type)
    tracker.init(frame, bbox)
    updated_trackers[next_id] = {"tracker": tracker, "bbox": bbox, "lost": 0, "pred_bbox": bbox}
    detection_associations.append((next_id, bbox))
    used_trackers.add(next_id)
    return next_id + 1


def associate_detections_with_trackers(
    trackers: dict,
    new_bboxes: list[tuple[tuple[int, int, int, int], float]],
    frame: np.ndarray,
    tracker_type: str,
    new_track_thresh: float,
    track_high_thresh: float,
    track_low_thresh: float,
    match_thresh: float,
    next_id: int,
) -> tuple[list[tuple[int, tuple[int, int, int, int]]], dict, int]:
    """Associa as detecções com os trackers existentes ou cria novos trackers conforme necessário."""
    detection_associations: list[tuple[int, tuple[int, int, int, int]]] = []
    used_trackers: set[int] = set()

    for det_bbox, det_conf in new_bboxes:
        best_tracker_id = None
        best_iou = 0.0
        # Busca o tracker com melhor IoU para a detecção atual
        for obj_id, info in trackers.items():
            if obj_id in used_trackers:
                continue
            iou = bb_iou(det_bbox, info["pred_bbox"])
            if iou > best_iou:
                best_iou, best_tracker_id = iou, obj_id

        if best_tracker_id is not None:
            # Se houver associação forte, atualiza o tracker existente
            if best_iou >= track_high_thresh:
                detection_associations.append((best_tracker_id, det_bbox))
                trackers[best_tracker_id].update({"bbox": det_bbox, "lost": 0})
                used_trackers.add(best_tracker_id)
            # Se a associação for fraca, cria um novo tracker se a confiança for alta
            elif best_iou < track_low_thresh:
                if det_conf >= new_track_thresh:
                    next_id = create_new_tracker(
                        trackers,
                        frame,
                        tracker_type,
                        det_bbox,
                        next_id,
                        detection_associations,
                        used_trackers,
                    )
            # Se a associação for intermediária, atualiza o tracker existente
            elif best_iou > match_thresh:
                detection_associations.append((best_tracker_id, det_bbox))
                trackers[best_tracker_id].update({"bbox": det_bbox, "lost": 0})
                used_trackers.add(best_tracker_id)
            # Caso contrário, cria novo tracker se a confiança for suficiente
            elif det_conf >= new_track_thresh:
                next_id = create_new_tracker(
                    trackers,
                    frame,
                    tracker_type,
                    det_bbox,
                    next_id,
                    detection_associations,
                    used_trackers,
                )
        # Se nenhum tracker corresponde, cria um novo tracker se a confiança for suficiente
        elif det_conf >= new_track_thresh:
            next_id = create_new_tracker(
                trackers,
                frame,
                tracker_type,
                det_bbox,
                next_id,
                detection_associations,
                used_trackers,
            )
    return detection_associations, trackers, next_id


def increment_lost_counters(trackers: dict, used_trackers: set) -> dict:
    """Incrementa o contador de frames perdidos para os trackers não associados a nenhuma detecção."""
    for obj_id, info in trackers.items():
        if obj_id not in used_trackers:
            info["lost"] += 1
    return trackers


def remove_lost_trackers(trackers: dict, track_buffer: int) -> dict:
    """Remove trackers que não detectaram objetos por mais de 'track_buffer' frames."""
    return {obj_id: info for obj_id, info in trackers.items() if info["lost"] < track_buffer}


def visualize_detections(frame: np.ndarray, detection_associations: list[tuple[int, tuple[int, int, int, int]]]) -> None:
    """Desenha as bounding boxes e IDs dos trackers associados à detecção."""
    for obj_id, bbox in detection_associations:
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {obj_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def process_video(
    input_path: str,
    output_path: str,
    model_path: str = "yolov8n.pt",
    start_frame: int = 0,
    max_frames: int = 300,
    tracker_type: str = "CSRT",
    track_high_thresh: float = 0.7,
    track_low_thresh: float = 0.3,
    new_track_thresh: float = 0.5,
    track_buffer: int = 5,
    match_thresh: float = 0.3,
) -> None:
    """Processa o vídeo realizando detecção com YOLO e rastreamento com OpenCV."""
    model = YOLO(model_path)
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Erro ao abrir o vídeo: {input_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if start_frame >= total_frames:
        print(f"Erro: start_frame ({start_frame}) é maior que o total de frames ({total_frames})")
        cap.release()
        return

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    trackers: dict = {}
    next_id: int = 1
    frame_number: int = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame_number >= max_frames:
            break

        # Detecção: considerando apenas a classe "person" (classe 0)
        results = model(frame, classes=[0])
        # Utiliza list comprehension para obter as bounding boxes e as confidências
        new_bboxes = [
            ((int(x1), int(y1), int(x2 - x1), int(y2 - y1)), conf)
            for result in results
            for x1, y1, x2, y2, conf, _ in result.boxes.data.tolist()
        ]

        trackers = update_tracker_predictions(trackers, frame)
        detection_associations, trackers, next_id = associate_detections_with_trackers(
            trackers,
            new_bboxes,
            frame,
            tracker_type,
            new_track_thresh,
            track_high_thresh,
            track_low_thresh,
            match_thresh,
            next_id,
        )

        used_tracker_ids = {tracker_id for tracker_id, _ in detection_associations}
        trackers = increment_lost_counters(trackers, used_tracker_ids)
        trackers = remove_lost_trackers(trackers, track_buffer)

        # Plota apenas as bboxes associadas à detecção no frame atual
        visualize_detections(frame, detection_associations)
        out.write(frame)
        frame_number += 1

    cap.release()
    out.release()
    print(f"Processamento concluído. Vídeo salvo em {output_path}")


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser(
        description="Processa um vídeo e rastreia objetos usando YOLOv8 e OpenCV.",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Caminho para o vídeo de entrada",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Caminho para o vídeo de saída (gerado automaticamente se omitido)",
    )
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Modelo YOLO a ser utilizado")
    parser.add_argument("--start_frame", type=int, default=0, help="Frame inicial para processamento")
    parser.add_argument("--frames", type=int, default=3000000, help="Número máximo de frames a processar")
    parser.add_argument(
        "--tracker",
        type=str,
        default="KCF",
        choices=list(TRACKER_CREATORS.keys()),
        help="Tipo de tracker do OpenCV",
    )
    parser.add_argument("--track_high_thresh", type=float, default=0.7, help="Limite alto de IoU para associação")
    parser.add_argument("--track_low_thresh", type=float, default=0.3, help="Limite baixo de IoU para associação")
    parser.add_argument("--new_track_thresh", type=float, default=0.5, help="Confiança mínima para criar um novo tracker")
    parser.add_argument("--track_buffer", type=int, default=30, help="Máximo de frames sem detecção para manter um tracker")
    parser.add_argument("--match_thresh", type=float, default=0.3, help="Limite de IoU para associação intermediária")

    args = parser.parse_args()
    if not args.input.strip():
        print("Erro: Nenhum caminho de vídeo foi informado.")
        exit(1)
    args.output = define_output_path(args.input, args.output)

    process_video(
        args.input,
        args.output,
        args.model,
        args.start_frame,
        args.frames,
        args.tracker,
        args.track_high_thresh,
        args.track_low_thresh,
        args.new_track_thresh,
        args.track_buffer,
        args.match_thresh,
    )

    elapsed_time = time.time() - start_time
    hours, minutes, seconds = int(elapsed_time // 3600), int((elapsed_time % 3600) // 60), int(elapsed_time % 60)
    print(f"Tempo de execução: {hours}h {minutes}m {seconds}s")
