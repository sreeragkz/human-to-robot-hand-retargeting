#!/usr/bin/env python3
"""Export every frame of a video for hand retargeting.

Uses the same MediaPipe Solutions Hands API as the original script. Keep your
working MediaPipe environment; this script does not require an API migration.

lm_i_x/y: normalized by image width/height respectively.
lm_i_z: MediaPipe relative depth, NOT metres (wrist is the depth origin).
world_lm_i_x/y/z: model-estimated metres around the hand's geometric centre;
these are NOT a camera-to-robot calibration or absolute wrist position.
confidence: legacy alias of handedness_confidence, NOT landmark accuracy.

No smoothing or interpolation is applied. Missing detections remain blank.
time_s = frame / fps preserves the original pipeline's nominal timeline. For
variable-frame-rate videos this is not exact presentation timing; source_pos_ms
is exported as backend diagnostic data only. Use constant-frame-rate recordings
when exact timing matters. All outputs share the source video's decoded frames.

Example: python extract_hand.py --video videos/fingers.mp4 --csv output_csv/fingers.csv
Keep this file beside track_cube.py: it also supplies shared file/video helpers.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def video_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def metadata_path(csv_path):
    return Path(str(csv_path) + '.meta.json')


def check_paths(inputs, outputs):
    ins = {Path(p).resolve() for p in inputs}
    outs = [Path(p).resolve() for p in outputs if p is not None]
    if len(set(outs)) != len(outs) or ins.intersection(outs):
        raise ValueError('Input and output paths must be distinct.')
    for path in outs:
        path.parent.mkdir(parents=True, exist_ok=True)


def open_video(cv2, path, width):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f'Cannot open video: {path}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    ok, first = cap.read()
    if not ok or not math.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError('Video needs a readable frame and positive finite FPS.')
    ih, iw = first.shape[:2]
    # Even dimensions avoid codec cropping. Default matches the original 1280.
    height = max(2, 2 * round(width * ih / iw / 2))
    info = dict(fps=fps, source_width=iw, source_height=ih,
                image_width=width, image_height=height,
                timing='frame_index_divided_by_fps')
    return cap, first, info


def open_writer(cv2, path, info):
    if path is None:
        return None
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'),
                             info['fps'], (info['image_width'], info['image_height']))
    if not writer.isOpened():
        writer.release()
        raise ValueError(f'Cannot write annotated video: {path}')
    return writer


def hand_row(results, frame_no, info, mirrored=False, requested_hand='any'):
    """Pure conversion: preserves both coordinate representations and labels."""
    row = dict(frame=frame_no, time_s=frame_no/info['fps'], detected=0,
               hand_detected=0, world_detected=0, handedness='',
               mediapipe_handedness='', confidence='', handedness_confidence='',
               detection_status='not_detected', input_mirrored=int(mirrored), **info)
    hands = results.multi_hand_landmarks or []
    labels = results.multi_handedness or []
    candidates = []
    for i, hand in enumerate(hands):
        raw, score = '', ''
        if i < len(labels) and labels[i].classification:
            c = labels[i].classification[0]
            raw, score = c.label, c.score
        # Legacy Solutions Hands assumes mirrored input. This changes the label
        # only: pixels and landmark coordinates are never flipped here.
        corrected = raw if mirrored else {'Left':'Right', 'Right':'Left'}.get(raw, '')
        if requested_hand == 'any' or corrected.lower() == requested_hand:
            candidates.append((i, hand, raw, corrected, score))
    if len(candidates) != 1:
        row['detection_status'] = 'ambiguous_hands' if len(candidates)>1 else 'requested_hand_missing'
        return row, None
    i, hand, raw, corrected, score = candidates[0]
    if len(hand.landmark) != 21 or not all(math.isfinite(getattr(lm,a))
                                         for lm in hand.landmark for a in 'xyz'):
        row['detection_status'] = 'invalid_landmarks'
        return row, None
    row.update(detected=1, hand_detected=1, handedness=corrected,
               mediapipe_handedness=raw, confidence=score,
               handedness_confidence=score, detection_status='detected')
    for j, lm in enumerate(hand.landmark):
        for axis in 'xyz':
            row[f'lm_{j}_{axis}'] = getattr(lm, axis)
    worlds = getattr(results, 'multi_hand_world_landmarks', None) or []
    if i < len(worlds) and len(worlds[i].landmark)==21 and all(
            math.isfinite(getattr(lm,a)) for lm in worlds[i].landmark for a in 'xyz'):
        row['world_detected'] = 1
        for j,lm in enumerate(worlds[i].landmark):
            for axis in 'xyz':
                row[f'world_lm_{j}_{axis}'] = getattr(lm,axis)
    return row, hand


def extract(args):
    import cv2
    import mediapipe as mp
    if not hasattr(mp, 'solutions') or not hasattr(mp.solutions, 'hands'):
        raise ValueError('This script uses mp.solutions.hands, as your original did. '
                         'Run it in your existing working MediaPipe environment.')
    video, output = Path(args.video), Path(args.csv)
    annotated = None if args.no_annotated else Path(args.annotated or
                       output.with_name(output.stem + '_annotated.mp4'))
    meta = metadata_path(output)
    check_paths([video], [output, annotated, meta])
    digest = video_hash(video)
    cap, raw, info = open_video(cv2, video, args.width)
    writer = None
    # A sidecar is marked incomplete before processing, so a failed run cannot
    # silently be merged as though it were a complete recording.
    metadata = dict(schema_version=2, complete=False, source_sha256=digest,
                    source_name=video.name, input_mirrored=args.input_mirrored,
                    requested_hand=args.hand, mediapipe_version=mp.__version__, **info)
    try:
        meta.write_text(json.dumps(metadata, indent=2)+'\n')
        writer = open_writer(cv2, annotated, info)
        fields = ['frame','time_s','detected','hand_detected','world_detected',
                  'handedness','mediapipe_handedness','confidence','handedness_confidence',
                  'detection_status','input_mirrored',*info.keys(),'source_pos_ms']
        fields += [f'{prefix}{i}_{a}' for prefix in ('lm_','world_lm_')
                   for i in range(21) for a in 'xyz']
        count = detected = 0
        with open(output, 'w', newline='') as f, mp.solutions.hands.Hands(
                static_image_mode=False, max_num_hands=2, model_complexity=1,
                min_detection_confidence=.5, min_tracking_confidence=.5) as hands:
            out = csv.DictWriter(f, fieldnames=fields)
            out.writeheader()
            while raw is not None:
                frame = cv2.resize(raw, (info['image_width'],info['image_height']))
                result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                row, selected = hand_row(result,count,info,args.input_mirrored,args.hand)
                row['source_pos_ms'] = cap.get(cv2.CAP_PROP_POS_MSEC)
                out.writerow(row)
                detected += row['detected']
                if writer is not None:
                    if selected is not None:
                        mp.solutions.drawing_utils.draw_landmarks(
                            frame,selected,mp.solutions.hands.HAND_CONNECTIONS)
                    label = row['handedness'] or row['detection_status']
                    cv2.putText(frame,f"{count/info['fps']:.2f}s {label}",(15,30),
                                cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,0),2)
                    writer.write(frame)
                count += 1
                if count % 100 == 0:
                    print(f'Processed {count} frames',flush=True)
                ok, raw = cap.read()
                if not ok:
                    raw = None
        reported_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if reported_count > 0 and count != reported_count:
            raise ValueError(f'Decoded {count} frames but container reports {reported_count}; '
                             'check the video for truncation before using this CSV.')
        metadata.update(complete=True,frames=count,detected_frames=detected)
        meta.write_text(json.dumps(metadata,indent=2)+'\n')
        print(f'Saved {output}: {count} frames, {detected} detected.')
        if annotated:
            print(f'Annotated video: {annotated}')
        print(f'Metadata: {meta}\nCoordinates remain raw; no gaps are filled.')
    finally:
        cap.release()
        if writer is not None:
            writer.release()


def main():
    p = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--video',default='videos/cube_hold.mp4')
    p.add_argument('--csv',default='output_csv/hand_landmarks.csv')
    p.add_argument('--annotated',help='Default: <CSV stem>_annotated.mp4 beside CSV')
    p.add_argument('--no-annotated',action='store_true')
    p.add_argument('--width',type=int,default=1280)
    p.add_argument('--hand',choices=['any','right','left'],default='any',
                   help='Anatomical hand after mirror correction; ambiguous matches are skipped.')
    p.add_argument('--input-mirrored',action='store_true',
                   help='Set only if the saved video pixels are horizontally mirrored.')
    a = p.parse_args()
    if a.width < 2 or a.width % 2:
        p.error('--width must be an even integer >= 2.')
    try:
        extract(a)
    except (ValueError,OSError,ImportError) as exc:
        p.exit(2,f'ERROR: {exc}\n')


if __name__ == '__main__':
    main()
