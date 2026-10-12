r"""
backup_data.py

Back up raw data (.avi, .dat, .rhd, .xml) from Z:\Isabel\data to an external
hard drive and to S3 Deep Archive, mirroring the server's folder structure
exactly, verifying every copy with a CRC64NVME checksum. Optionally (only with
--delete) remove server files that are verified on BOTH backups.

    Server:  Z:\Isabel\data\<EXPERIMENT>\<BIRD>\<BIRD_YYMMDD>\<file>
    HDD:     E:\isabel-lhy-hpc\<EXPERIMENT>\<BIRD>\<BIRD_YYMMDD>\<file>
    S3:      s3://isabel-lhy-hpc/<EXPERIMENT>/<BIRD>/<BIRD_YYMMDD>/<file>

Destination paths are always computed relative to DATA_ROOT, so you can point
the script at the whole data folder, one experiment, one bird or one session
and files still land in the right place.

Usage (pick exactly one of --to hdd, --to aws, --delete):
    python backup_data.py <folder> [<folder> ...] --to hdd  [--dry-run]
    python backup_data.py <folder> [<folder> ...] --to aws  [--dry-run]
    python backup_data.py <folder> [<folder> ...] --delete  [--dry-run] [--yes]

    e.g. python backup_data.py Z:\Isabel\data\behavior --to hdd --dry-run
         python backup_data.py Z:\Isabel\data\behavior --to hdd
         python backup_data.py Z:\Isabel\data\behavior --to aws
         python backup_data.py Z:\Isabel\data\behavior --delete --dry-run
         python backup_data.py Z:\Isabel\data\behavior --delete

    --to hdd     Copy to HDD_ROOT and verify each copy by re-reading it from
                 the drive. Files are written under a temporary name and only
                 renamed to their real name after they verify, so a file with
                 its real name on the drive was verified when it was written.
    --to aws     Upload to S3 Deep Archive. The AWS CLI sends a CRC64NVME
                 checksum that S3 checks on arrival; afterwards the script
                 compares S3's stored size and checksum with its own reading
                 of the server file.
    --delete     Does not copy anything. For every file, re-reads the server
                 file and the hard-drive copy, asks S3 for its stored checksum,
                 and marks the file deletable only if size and checksum match
                 in ALL THREE places. Then asks you to type DELETE before
                 removing those server files. Files that don't verify, and all
                 files with other extensions, are left on the server.
    --dry-run    Report what would happen; copies, uploads and deletes nothing.
                 With --delete it still does the full verification, so
                 `--delete --dry-run` is a complete integrity check of both
                 backups.
    --yes        With --delete, skip the typed confirmation.
    --from-file  Text file with one folder per line ('#' lines ignored).

The ONLY operation that ever removes a server file is --delete, and only after
the checks above. Re-running any mode is safe. Every run writes a log and a
CSV manifest to LOG_DIR.

Requirements:
    AWS CLI v2, version 2.23.0 or newer, configured with `aws configure`
    Python 3.8+ with awscrt:  python -m pip install awscrt
"""

import argparse
import base64
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

# ---- Configuration ------------------------------------------------------

AWS_EXE       = r"C:\Program Files\Amazon\AWSCLIV2\aws.exe"
S3_BUCKET     = "isabel-lhy-hpc"
DATA_ROOT     = Path(r"Z:\Isabel\data")
HDD_ROOT      = Path(r"E:\isabel-lhy-hpc")    # create this folder on the drive once
LOG_DIR       = Path(r"C:\Users\arono\Documents\data_backup_logs")
EXTENSIONS    = {".avi", ".dat", ".rhd", ".xml"}
STORAGE_CLASS = "DEEP_ARCHIVE"
EXCLUDED_DIRS = {"$RECYCLE.BIN", "System Volume Information"}

MIN_CLI_VERSION = (2, 23, 0)          # first AWS CLI v2 release with CRC64NVME
READ_CHUNK      = 16 * 1024 * 1024    # bytes per read when copying/checksumming
PARTIAL_SUFFIX  = ".partial-copy"     # temporary name while a copy is unverified
MTIME_TOLERANCE = 2.0                 # seconds (exFAT stores coarse timestamps)

# -------------------------------------------------------------------------

BUCKET_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$")


def now():
    return datetime.now().isoformat(timespec="seconds")


def human_size(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def human_rate(nbytes, seconds):
    return f"{nbytes / max(seconds, 1e-9) / 1e6:.0f} MB/s"


def human_time(seconds):
    h, rem = divmod(int(seconds), 3600)
    return f"{h}h {rem // 60:02d}m" if h else f"{rem // 60}m {rem % 60:02d}s"


# ---- Finding files ------------------------------------------------------

def read_dir_list(list_file):
    dirs = []
    for line in Path(list_file).read_text(encoding="utf-8-sig").splitlines():
        line = line.strip().strip('"')
        if line and not line.startswith("#"):
            dirs.append(Path(line))
    return dirs


def resolve_folders(raw, data_root):
    folders = []
    for d in raw:
        d = d.resolve()
        if not d.is_dir():
            sys.exit(f"ERROR: {d} is not a directory (is the Z: drive connected?)")
        if d != data_root and data_root not in d.parents:
            sys.exit(f"ERROR: {d} is not inside DATA_ROOT ({data_root}).")
        folders.append(d)
    return folders


def scan(folder, data_root):
    """Return ({relative key: path} for target files, Counter of other
    extensions, list of folders that could not be read)."""
    targets, others, errors = {}, Counter(), []
    for dirpath, dirnames, filenames in os.walk(folder, onerror=errors.append):
        dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDED_DIRS)
        for name in sorted(filenames):
            p = Path(dirpath) / name
            ext = p.suffix.lower()
            if ext in EXTENSIONS:
                targets[p.relative_to(data_root).as_posix()] = p
            else:
                others[ext or "(no extension)"] += 1
    return targets, others, errors


def hdd_path_for(key):
    return HDD_ROOT.joinpath(*key.split("/"))


# ---- Checksums ----------------------------------------------------------

_crc64nvme = None


def load_crc64nvme():
    global _crc64nvme
    try:
        from awscrt import checksums
    except ImportError:
        sys.exit("ERROR: the 'awscrt' Python package is required. Install it with:\n"
                 "    python -m pip install awscrt")
    if not hasattr(checksums, "crc64nvme"):
        sys.exit("ERROR: your awscrt is too old. Update it with:\n"
                 "    python -m pip install --upgrade awscrt")
    _crc64nvme = checksums.crc64nvme


def encode_crc(crc):
    """Base64 of the big-endian CRC, the format S3 reports."""
    return base64.b64encode(crc.to_bytes(8, "big")).decode("ascii")


def file_checksum(path):
    """(CRC64NVME base64, bytes read) for a file, opened read-only."""
    crc, n = 0, 0
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(READ_CHUNK)
            if not chunk:
                break
            crc = _crc64nvme(chunk, crc)
            n += len(chunk)
    return encode_crc(crc), n


# ---- Hard drive ---------------------------------------------------------

def check_hdd(data_root):
    hdd = HDD_ROOT.resolve()
    if not hdd.is_dir():
        sys.exit(f"ERROR: {HDD_ROOT} not found. Plug in the external drive and make sure "
                 f"the folder exists (create it once by hand).")
    if hdd == data_root or data_root in hdd.parents or hdd in data_root.parents:
        sys.exit(f"ERROR: HDD_ROOT ({hdd}) overlaps DATA_ROOT ({data_root}). Check the config.")


def hdd_already_has(src, dst):
    """True if dst exists with the same size and modification time as src."""
    try:
        s, d = src.stat(), dst.stat()
    except OSError:
        return False
    return s.st_size == d.st_size and abs(s.st_mtime - d.st_mtime) <= MTIME_TOLERANCE


def copy_to_hdd(src, dst):
    """Copy src to dst via a temporary file, verify by re-reading the copy from
    the drive, then rename. Returns (ok, source checksum, reason)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + PARTIAL_SUFFIX)
    st = src.stat()
    crc, n = 0, 0
    try:
        with open(src, "rb") as fin, open(tmp, "wb") as fout:
            while True:
                chunk = fin.read(READ_CHUNK)
                if not chunk:
                    break
                crc = _crc64nvme(chunk, crc)
                fout.write(chunk)
                n += len(chunk)
            fout.flush()
            os.fsync(fout.fileno())
        src_sum = encode_crc(crc)

        st_after = src.stat()
        if n != st.st_size or st_after.st_size != st.st_size or st_after.st_mtime_ns != st.st_mtime_ns:
            raise ValueError("server file changed while it was being copied")

        dst_sum, dst_n = file_checksum(tmp)
        if dst_n != n or dst_sum != src_sum:
            raise ValueError(f"copy does not match (server={src_sum}/{n} B, "
                             f"drive={dst_sum}/{dst_n} B)")

        os.utime(tmp, ns=(st.st_atime_ns, st.st_mtime_ns))   # keep original timestamp
        os.replace(tmp, dst)
        return True, src_sum, ""
    except (OSError, ValueError) as e:
        try:
            tmp.unlink()          # remove only our own unverified temp copy on the drive
        except OSError:
            pass
        return False, "", str(e)


# ---- AWS ----------------------------------------------------------------

def find_aws_exe():
    if Path(AWS_EXE).is_file():
        return AWS_EXE
    found = shutil.which("aws")
    if found:
        return found
    sys.exit(f"ERROR: AWS CLI not found at {AWS_EXE} or on PATH.")


def check_aws():
    """Pre-flight: CLI present and new enough, bucket reachable. Returns (exe, version)."""
    if not BUCKET_NAME_RE.match(S3_BUCKET) or ".." in S3_BUCKET:
        sys.exit(f"ERROR: '{S3_BUCKET}' is not a valid S3 bucket name.")
    aws = find_aws_exe()
    out = subprocess.run([aws, "--version"], capture_output=True, text=True)
    text = (out.stdout + out.stderr).strip()
    m = re.search(r"aws-cli/(\d+)\.(\d+)\.(\d+)", text)
    if not m:
        sys.exit(f"ERROR: could not read the AWS CLI version from: {text!r}")
    if tuple(int(x) for x in m.groups()) < MIN_CLI_VERSION:
        sys.exit(f"ERROR: {text.split()[0]} is too old; reinstall the latest AWS CLI v2.")
    out = subprocess.run([aws, "s3api", "head-bucket", "--bucket", S3_BUCKET],
                         capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"ERROR: cannot access bucket '{S3_BUCKET}'. Check the name and "
                 f"`aws configure`.\nAWS said: {out.stderr.strip()}")
    return aws, text


def head_object(aws, key, log_fh):
    """S3 metadata (size, checksum, ...) for key, or None if missing/unreadable.
    Works on Deep Archive objects without restoring them."""
    cmd = [aws, "s3api", "head-object", "--bucket", S3_BUCKET, "--key", key,
           "--checksum-mode", "ENABLED", "--output", "json"]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        if "404" not in out.stderr and "Not Found" not in out.stderr:
            log_fh.write(f"head-object error key={key}: {out.stderr.strip()}\n")
        return None
    try:
        return json.loads(out.stdout)
    except json.JSONDecodeError as e:
        log_fh.write(f"head-object JSON parse error key={key}: {e}\n{out.stdout}\n")
        return None


def s3_matches(info, size, checksum):
    if info is None:
        return False, "not found in S3"
    if info.get("ContentLength") != size:
        return False, f"S3 size differs (server={size}, s3={info.get('ContentLength')})"
    s3_sum = info.get("ChecksumCRC64NVME")
    if s3_sum is None:
        return False, "S3 has no CRC64NVME checksum for this object"
    if s3_sum != checksum:
        return False, f"S3 checksum differs (server={checksum}, s3={s3_sum})"
    return True, ""


def upload_file(aws, local_file, key, log_fh):
    cmd = [aws, "s3", "cp", str(local_file), f"s3://{S3_BUCKET}/{key}",
           "--storage-class", STORAGE_CLASS, "--checksum-algorithm", "CRC64NVME"]
    log_fh.write(f"[{now()}] UPLOAD {local_file} -> s3://{S3_BUCKET}/{key}\n")
    log_fh.flush()
    out = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
    if out.stderr.strip():
        print(out.stderr.strip())
        log_fh.write(f"aws stderr: {out.stderr.strip()}\n")
    log_fh.write(f"[{now()}] returncode={out.returncode}\n")
    log_fh.flush()
    return out.returncode == 0


# ---- Modes --------------------------------------------------------------

class Run:
    """Shared bookkeeping: log file, CSV manifest, counters."""

    def __init__(self, log_fh, man_fh):
        self.log = log_fh
        self.man_fh = man_fh
        self.manifest = csv.writer(man_fh)
        self.manifest.writerow(["server_path", "destination", "size_bytes",
                                "crc64nvme_base64", "status", "note"])
        self.done, self.skipped, self.failures = [], [], []
        self.bytes_moved, self.t0 = 0, time.time()

    def record(self, src, dest, size, checksum, status, note=""):
        self.manifest.writerow([str(src), dest, size, checksum, status, note])
        self.man_fh.flush()

    def fail(self, key, src, dest, size, checksum, why):
        print(f"  -> FAILED: {why}")
        self.log.write(f"FAIL key={key}: {why}\n")
        self.failures.append((key, why))
        self.record(src, dest, size, checksum, "FAILED", why)


def run_hdd(keys, targets, run):
    for i, key in enumerate(keys, 1):
        src, dst = targets[key], hdd_path_for(key)
        print(f"[{i}/{len(keys)}] {key}")
        try:
            size = src.stat().st_size
        except OSError as e:
            run.fail(key, src, str(dst), "", "", f"cannot read server file: {e}")
            continue
        if hdd_already_has(src, dst):
            print("  -> already on drive (same size and timestamp); skipping")
            run.log.write(f"SKIP (already on drive) {dst}\n")
            run.skipped.append(key)
            run.record(src, str(dst), size, "", "already_on_drive")
            continue
        t = time.time()
        print(f"  copying {human_size(size)} and verifying ...")
        ok, checksum, why = copy_to_hdd(src, dst)
        if not ok:
            run.fail(key, src, str(dst), size, checksum, why)
            continue
        dt = time.time() - t
        print(f"  -> copied and verified ({human_time(dt)}, {human_rate(size, dt)} incl. verify)")
        run.log.write(f"OK (verified) {dst} crc64nvme={checksum}\n")
        run.done.append(key)
        run.bytes_moved += size
        run.record(src, str(dst), size, checksum, "copied_verified")


def run_aws(keys, targets, run, aws):
    for i, key in enumerate(keys, 1):
        src, uri = targets[key], f"s3://{S3_BUCKET}/{key}"
        print(f"[{i}/{len(keys)}] {key}")
        try:
            size = src.stat().st_size
            print(f"  checksumming {human_size(size)} ...")
            checksum, _ = file_checksum(src)
        except OSError as e:
            run.fail(key, src, uri, "", "", f"cannot read server file: {e}")
            continue

        info = head_object(aws, key, run.log)
        ok, why = s3_matches(info, size, checksum)
        if ok:
            print("  -> already in S3, size and checksum match; skipping")
            run.log.write(f"SKIP (verified identical) key={key}\n")
            run.skipped.append(key)
            run.record(src, uri, size, checksum, "already_in_s3")
            continue
        if info is not None:
            print(f"  -> exists in S3 but {why}; re-uploading")
            run.log.write(f"RE-UPLOAD key={key}: {why}\n")

        t = time.time()
        if not upload_file(aws, src, key, run.log):
            run.fail(key, src, uri, size, checksum, "upload failed")
            continue
        dt = time.time() - t
        ok, why = s3_matches(head_object(aws, key, run.log), size, checksum)
        if ok and src.stat().st_size != size:
            ok, why = False, "server file changed size during upload"
        if not ok:
            run.fail(key, src, uri, size, checksum, f"verify failed: {why}")
            continue
        print(f"  -> uploaded and verified ({human_time(dt)}, {human_rate(size, dt)} upload)")
        run.log.write(f"OK (verified) key={key} crc64nvme={checksum}\n")
        run.done.append(key)
        run.bytes_moved += size
        run.record(src, uri, size, checksum, "uploaded_verified")


def run_delete(keys, targets, run, aws, args):
    """Verify server == drive == S3 for every file; delete only fully verified files."""
    deletable = []   # (key, src, size, mtime_ns)
    for i, key in enumerate(keys, 1):
        src, dst = targets[key], hdd_path_for(key)
        print(f"[{i}/{len(keys)}] {key}")
        try:
            st = src.stat()
        except OSError as e:
            run.fail(key, src, "", "", "", f"cannot read server file: {e}")
            continue
        size = st.st_size

        # Cheap checks first, so missing backups are found without reading data.
        if not dst.is_file():
            run.fail(key, src, str(dst), size, "", "NOT on hard drive")
            continue
        if dst.stat().st_size != size:
            run.fail(key, src, str(dst), size, "", "hard-drive copy has a different size")
            continue
        info = head_object(aws, key, run.log)
        if info is None:
            run.fail(key, src, "", size, "", "NOT in S3")
            continue

        try:
            print(f"  checksumming server file and drive copy ({human_size(size)} each) ...")
            src_sum, _ = file_checksum(src)
            dst_sum, _ = file_checksum(dst)
        except OSError as e:
            run.fail(key, src, str(dst), size, "", f"read error: {e}")
            continue
        if dst_sum != src_sum:
            run.fail(key, src, str(dst), size, src_sum,
                     f"hard-drive checksum differs (server={src_sum}, drive={dst_sum})")
            continue
        ok, why = s3_matches(info, size, src_sum)
        if not ok:
            run.fail(key, src, "", size, src_sum, why)
            continue

        print("  -> verified on hard drive AND in S3")
        run.log.write(f"VERIFIED BOTH key={key} crc64nvme={src_sum}\n")
        deletable.append((key, src, size, st.st_mtime_ns))
        run.record(src, f"{dst} | s3://{S3_BUCKET}/{key}", size, src_sum,
                   "verified_both_backups")

    total = sum(d[2] for d in deletable)
    print(f"\n{len(deletable)} of {len(keys)} server file(s) are verified on both backups "
          f"({human_size(total)}).")
    if run.failures:
        print(f"{len(run.failures)} file(s) did NOT verify and will be kept on the server.")

    if not deletable:
        return
    if args.dry_run:
        print("DRY RUN: nothing deleted. These files would be deleted:")
        for key, src, _, _ in deletable:
            print(f"  {src}")
        return
    if not args.yes:
        reply = input(f"\nType DELETE to permanently remove these {len(deletable)} verified "
                      f"file(s) from the server, or anything else to cancel: ").strip()
        if reply != "DELETE":
            print("Cancelled. Nothing deleted.")
            run.log.write("Deletion cancelled by user.\n")
            return

    run.log.write(f"\nDELETION started {now()}\n")
    for key, src, size, mtime_ns in deletable:
        try:
            st = src.stat()
            if st.st_size != size or st.st_mtime_ns != mtime_ns:
                raise ValueError("server file changed since it was verified; not deleted")
            os.remove(src)
            run.log.write(f"  DELETED {src}\n")
            run.done.append(key)
            run.record(src, "", size, "", "deleted_from_server")
        except (OSError, ValueError) as e:
            print(f"  failed to delete {src}: {e}")
            run.log.write(f"  FAILED TO DELETE {src}: {e}\n")
            run.failures.append((key, f"delete failed: {e}"))
    print(f"Deleted {len(run.done)}/{len(deletable)} server file(s).")


# ---- Main ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folders", type=Path, nargs="*",
                        help=r"Folders under DATA_ROOT (e.g. Z:\Isabel\data\behavior)")
    parser.add_argument("--from-file", type=Path, help="Text file listing folders.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--to", choices=["hdd", "aws"], help="Back up to the hard drive or to AWS.")
    mode.add_argument("--delete", action="store_true",
                      help="Delete server files verified on BOTH backups.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report only; copy, upload and delete nothing.")
    parser.add_argument("--yes", action="store_true",
                        help="With --delete, skip the typed confirmation.")
    args = parser.parse_args()
    if args.yes and not args.delete:
        sys.exit("ERROR: --yes only applies together with --delete")
    mode_name = "delete" if args.delete else args.to

    raw = list(args.folders) + (read_dir_list(args.from_file) if args.from_file else [])
    if not raw:
        sys.exit("ERROR: no folders given (pass paths and/or --from-file)")
    data_root = DATA_ROOT.resolve()
    folders = resolve_folders(raw, data_root)

    # -- Scan the server ----------------------------------------------------
    targets, others, scan_errors = {}, Counter(), []
    for folder in folders:
        t, o, e = scan(folder, data_root)
        targets.update(t)
        others.update(o)
        scan_errors.extend(e)
    keys = sorted(targets)
    sizes = {k: targets[k].stat().st_size for k in keys}
    by_ext = Counter(Path(k).suffix.lower() for k in keys)

    dest = {"hdd": str(HDD_ROOT), "aws": f"s3://{S3_BUCKET}/",
            "delete": f"verify {HDD_ROOT} and s3://{S3_BUCKET}/, then delete from server"}
    print(f"Mode:      {mode_name.upper()}{' (DRY RUN)' if args.dry_run else ''}  ->  {dest[mode_name]}")
    print(f"Folders:   {', '.join(str(f) for f in folders)}")
    print(f"Files:     {len(keys)}, {human_size(sum(sizes.values()))}  "
          f"({', '.join(f'{n} {e}' for e, n in sorted(by_ext.items())) or 'none'})")
    by_top = Counter()
    for k in keys:
        by_top[k.split("/")[0]] += sizes[k]
    for top in sorted(by_top):
        n = sum(1 for k in keys if k.split("/")[0] == top)
        print(f"             {top}: {n} file(s), {human_size(by_top[top])}")
    if others:
        print("Ignored (other file types, never copied or deleted): "
              + ", ".join(f"{n} {e}" for e, n in sorted(others.items())))
    if scan_errors:
        print(f"\nWARNING: {len(scan_errors)} folder(s) could not be read and are NOT included:")
        for e in scan_errors:
            print(f"  {e}")
    if keys and not args.delete:
        print("Example destination paths:")
        for k in keys[:3]:
            print(f"  {hdd_path_for(k) if args.to == 'hdd' else f's3://{S3_BUCKET}/{k}'}")
    print()
    if not keys:
        print("Nothing to do.")
        sys.exit(1 if scan_errors else 0)

    if args.dry_run and not args.delete:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        plan = LOG_DIR / f"{args.to}_dryrun_filelist_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(plan, "w", encoding="utf-8") as fh:
            for k in keys:
                to = hdd_path_for(k) if args.to == "hdd" else f"s3://{S3_BUCKET}/{k}"
                fh.write(f"{targets[k]}\t->\t{to}\n")
        print(f"Full list of files: {plan}")

    # -- Pre-flight checks ----------------------------------------------------
    aws = cli_version = None
    if args.to == "hdd" or args.delete:
        check_hdd(data_root)
    if args.to == "aws" or args.delete:
        aws, cli_version = check_aws()

    if args.to == "hdd":
        todo = [k for k in keys if not hdd_already_has(targets[k], hdd_path_for(k))]
        need, free = sum(sizes[k] for k in todo), shutil.disk_usage(HDD_ROOT).free
        print(f"Drive: {len(keys) - len(todo)} already copied, {len(todo)} to copy "
              f"({human_size(need)}); free space {human_size(free)}.")
        if need > free:
            sys.exit("ERROR: not enough free space on the drive.")
        if args.dry_run:
            print("DRY RUN: nothing copied.")
            return

    if args.to == "aws" and args.dry_run:
        present = sum(1 for k in keys
                      if (head_object(aws, k, sys.stderr) or {}).get("ContentLength") == sizes[k])
        print(f"S3: {present} file(s) already present with matching size (checksums are "
              f"checked on a real run); {len(keys) - present} would be uploaded.")
        print("DRY RUN: nothing uploaded.")
        return

    load_crc64nvme()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = "_".join(folders[0].relative_to(data_root).parts) or data_root.name
    if len(folders) > 1:
        label += f"_and_{len(folders) - 1}_more"
    tag = f"{mode_name}{'_dryrun' if args.dry_run else ''}"
    log_path = LOG_DIR / f"{label}_{tag}_{stamp}.log"
    man_path = LOG_DIR / f"{label}_{tag}_manifest_{stamp}.csv"
    print(f"Log file:  {log_path}\nManifest:  {man_path}\n")

    with open(log_path, "w", encoding="utf-8") as log_fh, \
         open(man_path, "w", newline="", encoding="utf-8") as man_fh:
        run = Run(log_fh, man_fh)
        log_fh.write(f"Run started {now()}  mode={tag}\n")
        if cli_version:
            log_fh.write(f"{cli_version}\n")
        log_fh.write(f"Folders: {', '.join(str(f) for f in folders)}\n")
        log_fh.write(f"Files: {len(keys)}, {human_size(sum(sizes.values()))}\n")
        for e in scan_errors:
            log_fh.write(f"SCAN ERROR (folder not read): {e}\n")
        log_fh.write("\n")

        interrupted = False
        try:
            if args.to == "hdd":
                run_hdd(keys, targets, run)
            elif args.to == "aws":
                run_aws(keys, targets, run, aws)
            else:
                run_delete(keys, targets, run, aws, args)
        except KeyboardInterrupt:
            interrupted = True
            print("\nInterrupted. Re-run the same command to resume.")
            log_fh.write(f"\nINTERRUPTED at {now()}\n")

        elapsed = time.time() - run.t0
        if args.delete:
            summary = (f"Deleted from server: {len(run.done)}\n"
                       f"Kept (not verified on both backups, or delete failed): "
                       f"{len(run.failures)}\n")
        else:
            summary = (f"Copied/uploaded and verified: {len(run.done)} "
                       f"({human_size(run.bytes_moved)}, {human_rate(run.bytes_moved, elapsed)} "
                       f"overall)\n"
                       f"Already backed up: {len(run.skipped)}\n"
                       f"Failed: {len(run.failures)}\n")
        summary += f"Elapsed: {human_time(elapsed)}\nUnreadable folders: {len(scan_errors)}\n"
        log_fh.write(f"\nRun ended {now()}\n{summary}")
        print("\n" + summary)
        if run.failures:
            print("Problems:")
            for key, why in run.failures:
                print(f"  [{why}] {key}")
                log_fh.write(f"PROBLEM [{why}] {key}\n")

    complete = not run.failures and not scan_errors and not interrupted
    if not args.delete:
        print("ALL FILES BACKED UP AND VERIFIED." if complete
              else "NOT COMPLETE: see above. Re-running the same command is safe.")
    sys.exit(0 if complete else 1)


if __name__ == "__main__":
    main()
