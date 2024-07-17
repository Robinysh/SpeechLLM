import io
import json
from copy import deepcopy
from pathlib import Path

import torchaudio


def export_json(row, output_path):
    save_path = Path(output_path) / "json" / f"{row['original_index']}.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    row["json_fpath"] = save_path
    if save_path.exists():
        return row
    data = deepcopy(row)
    data.pop("audio_second_last_turn")
    save_path.write_text(json.dumps(data, ensure_ascii=False, indent=4))
    return row


def export_audio(row, output_path):
    save_path = Path(output_path) / "audio" / f"{row['original_index']}.opus"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    row["audio_fpath"] = save_path
    if save_path.exists():
        return row

    f = io.BytesIO(row["audio_second_last_turn"]["bytes"])
    data, sr = torchaudio.load(f)
    assert sr == 16000, f'Sampling rate is {sr} for item {row["original_index"]}'
    torchaudio.save(
        save_path,
        data,
        sr,
    )
    return row
