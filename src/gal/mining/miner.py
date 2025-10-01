#!/usr/bin/env python3
"""
Batch-mine Minimal-Non-Redundant rules with the choco-mining CLI and
*automatically* convert any binary CSV datasets to .dat first.
Temporary .dat files produced from CSVs are removed right after use.

Usage
-----
python miner.py <dataset-folder> <cli-fat-jar> [support] [confidence]
"""

import csv
import sys
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, List


class MNRBatchMiner:
    # ------------------------------------------------------------------
    def __init__(self, jar_path: Path, support: int = 10, confidence: int = 99):
        self.jar = Path(jar_path).expanduser().resolve()
        if not self.jar.is_file():
            raise FileNotFoundError(self.jar)
        self.support = support
        self.conf = confidence

    # ------------------------------------------------------------------
    @staticmethod
    def csv_to_dat(csv_file: Path, dat_file: Path) -> None:
        """Convert a *binary* CSV (0/1) to Pasquier-style .dat"""
        with csv_file.open(newline="") as f:
            first = f.readline()
            delim = ";" if ";" in first else "," if "," in first else " "
            f.seek(0)
            rows: List[List[str]] = list(csv.reader(f, delimiter=delim))

        if not rows:
            raise ValueError(f"{csv_file} is empty")

        header = any(tok not in ("0", "1") for tok in rows[0])
        data_rows = rows[1:] if header else rows
        n_tx, n_items = len(data_rows), len(data_rows[0])

        for r, row in enumerate(data_rows, start=2 if header else 1):
            if len(row) != n_items:
                raise ValueError(
                    f"{csv_file}: line {r} has {len(row)} "
                    f"columns (expected {n_items})"
                )

        dat_file.parent.mkdir(parents=True, exist_ok=True)
        with dat_file.open("w") as out:
            out.write(f"{n_tx} {n_items}\n")
            for row in data_rows:
                items = [str(i + 1) for i, tok in enumerate(row) if tok.strip() == "1"]
                out.write(" ".join(items) + "\n")

    # ------------------------------------------------------------------
    def run_all(
        self,
        data_root: Path,
        out_root: Optional[Path] = None,
        glob: str = "*",
        recursive: bool = False,
    ):
        data_root = Path(data_root).expanduser().resolve()
        if not data_root.is_dir():
            raise NotADirectoryError(data_root)

        repo_root = Path(__file__).resolve().parent
        out_root = Path(out_root or repo_root / "mined_rules").resolve()
        out_root.mkdir(parents=True, exist_ok=True)

        pattern = "**/" + glob if recursive else glob

        for ds in sorted(data_root.glob(pattern)):
            if not ds.is_file() or ds.suffix.lower() not in {".csv", ".dat"}:
                continue

            stem = ds.stem.replace(" ", "_")
            dst_csv = out_root / f"{stem}_mnr.csv"

            tmp_dat = None  # track whether we created one
            if ds.suffix.lower() == ".csv":
                tmp_dat = out_root / f"{stem}.dat"
                try:
                    self.csv_to_dat(ds, tmp_dat)
                except Exception as e:
                    print(
                        f"[{datetime.now():%H:%M:%S}] {ds.name}: CSV→DAT failed → {e}"
                    )
                    continue
                dat_path = tmp_dat
            else:
                dat_path = ds

            print(f"[{datetime.now():%H:%M:%S}] Mining {ds.name} → {dst_csv.name}")
            try:
                subprocess.run(
                    [
                        "java",
                        "-jar",
                        str(self.jar),
                        f"--data={dat_path}",
                        f"--support={self.support}",
                        f"--confidence={self.conf}",
                    ],
                    cwd=out_root,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                tmp = out_root / "mnr_rules.csv"
                if not tmp.exists():
                    print("   ERROR: CLI did not emit mnr_rules.csv")
                    continue
                tmp.rename(dst_csv)
                print("   ✓ saved")
            except subprocess.CalledProcessError as exc:
                print("   FAILED — see output below")
                print(exc.stdout.decode(errors="ignore"))
            finally:
                # ------- clean up temporary .dat -------------------
                if tmp_dat and tmp_dat.exists():
                    try:
                        tmp_dat.unlink()
                    except OSError:
                        pass

        print("\nBatch finished.")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "usage: python miner.py <dataset-folder> <cli-fat-jar> "
            "[support] [confidence]"
        )
        sys.exit(1)

    folder = Path(sys.argv[1])
    jar = Path(sys.argv[2])
    sup = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    conf = int(sys.argv[4]) if len(sys.argv) > 4 else 5

    MNRBatchMiner(jar, sup, conf).run_all(folder)
