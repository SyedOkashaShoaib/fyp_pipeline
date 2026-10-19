"""Extract the official JASPAR motif-to-cluster mapping.

The script reads the official JASPAR matrix-clustering ZIP directly,
ignores the combined cluster_XX.tf summary files(Each cluster folder has an extra file with the names
of all the motifs in that cluster -> We need to ignore that file for this), validates every
individual motif against the matching JASPAR MEME collection, and
writes a deterministic TSV mapping.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


INDIVIDUAL_FILE_PATTERN = re.compile(
    r"(?:^|/)"
    r"plants_CORE_motifs/motifs_sep_by_cluster/"
    r"(?P<cluster>cluster_\d{2})/"
    r"JASPAR_plants_CORE_"
    r"(?P<matrix_id>MA\d+\.\d+)_"
    r"n(?P<archive_index>\d+)_oriented\.tf$"
)

SUMMARY_FILE_PATTERN = re.compile(
    r"(?:^|/)"
    r"plants_CORE_motifs/motifs_sep_by_cluster/"
    r"(?P<cluster>cluster_\d{2})/"
    r"(?P=cluster)\.tf$"
)

AC_LINE_PATTERN = re.compile(
    r"^AC\s+JASPAR_plants_CORE_"
    r"(?P<matrix_id>MA\d+\.\d+)_"
    r"n(?P<archive_index>\d+)\s*$",
    re.MULTILINE,
)

ID_LINE_PATTERN = re.compile(
    r"^ID\s+(?P<motif_name>.+?)\s*$",
    re.MULTILINE,
)

MEME_MOTIF_PATTERN = re.compile(
    r"^MOTIF\s+"
    r"(?P<matrix_id>MA\d+\.\d+)"
    r"(?:\s+(?P<motif_name>.+?))?\s*$"
)


class ValidationError(Exception):
    """Raised when the official inputs fail a consistency check."""


@dataclass(frozen=True)
class ClusterRecord:
    matrix_id: str
    motif_name: str
    jaspar_cluster: str
    archive_index: int
    source_member: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def preview(values: set[str], limit: int = 10) -> str:
    ordered = sorted(values)
    shown = ", ".join(ordered[:limit])

    if len(ordered) > limit:
        shown += f", ... ({len(ordered) - limit} more)"

    return shown


def parse_meme_file(path: Path) -> dict[str, str]:
    motifs: dict[str, str] = {}

    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.rstrip("\r\n")
            match = MEME_MOTIF_PATTERN.match(line)

            if match is None:
                continue

            matrix_id = match.group("matrix_id")
            motif_name = (match.group("motif_name") or "").strip()

            if not motif_name:
                raise ValidationError(
                    f"MEME motif {matrix_id} has no motif name "
                    f"at line {line_number}."
                )

            if matrix_id in motifs:
                raise ValidationError(
                    f"Duplicated Matrix ID in MEME file: {matrix_id}"
                )

            motifs[matrix_id] = motif_name

    if not motifs:
        raise ValidationError(
            "No JASPAR Matrix IDs were found in the MEME file."
        )

    return motifs


def decode_tf_file(data: bytes, member_name: str) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValidationError(
            f"Could not decode archive member as UTF-8: {member_name}"
        ) from error


def parse_individual_tf(
    text: str,
    member_name: str,
    filename_matrix_id: str,
    filename_archive_index: int,
    cluster: str,
) -> ClusterRecord:
    ac_matches = list(AC_LINE_PATTERN.finditer(text))
    id_matches = list(ID_LINE_PATTERN.finditer(text))

    if len(ac_matches) != 1:
        raise ValidationError(
            f"{member_name}: expected exactly one AC line, "
            f"found {len(ac_matches)}."
        )

    if len(id_matches) != 1:
        raise ValidationError(
            f"{member_name}: expected exactly one ID line, "
            f"found {len(id_matches)}."
        )

    ac_matrix_id = ac_matches[0].group("matrix_id")
    ac_archive_index = int(ac_matches[0].group("archive_index"))
    motif_name = id_matches[0].group("motif_name").strip()

    if ac_matrix_id != filename_matrix_id:
        raise ValidationError(
            f"{member_name}: filename Matrix ID {filename_matrix_id} "
            f"does not match AC Matrix ID {ac_matrix_id}."
        )

    if ac_archive_index != filename_archive_index:
        raise ValidationError(
            f"{member_name}: filename archive index "
            f"{filename_archive_index} does not match AC index "
            f"{ac_archive_index}."
        )

    if not motif_name:
        raise ValidationError(
            f"{member_name}: the ID line has an empty motif name."
        )

    return ClusterRecord(
        matrix_id=filename_matrix_id,
        motif_name=motif_name,
        jaspar_cluster=cluster,
        archive_index=filename_archive_index,
        source_member=member_name,
    )


def parse_cluster_archive(
    path: Path,
) -> tuple[list[ClusterRecord], set[str]]:
    records: list[ClusterRecord] = []
    summary_clusters: set[str] = set()
    unexpected_tf_files: list[str] = []

    with zipfile.ZipFile(path, "r") as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue

            member_name = member.filename.replace("\\", "/")

            if "motifs_sep_by_cluster/" not in member_name:
                continue

            if not member_name.lower().endswith(".tf"):
                continue

            summary_match = SUMMARY_FILE_PATTERN.search(member_name)

            if summary_match is not None:
                cluster = summary_match.group("cluster")

                if cluster in summary_clusters:
                    raise ValidationError(
                        f"More than one summary file found for {cluster}."
                    )

                summary_clusters.add(cluster)
                continue

            motif_match = INDIVIDUAL_FILE_PATTERN.search(member_name)

            if motif_match is None:
                unexpected_tf_files.append(member_name)
                continue

            cluster = motif_match.group("cluster")
            matrix_id = motif_match.group("matrix_id")
            archive_index = int(motif_match.group("archive_index"))

            text = decode_tf_file(
                archive.read(member),
                member_name,
            )

            records.append(
                parse_individual_tf(
                    text=text,
                    member_name=member_name,
                    filename_matrix_id=matrix_id,
                    filename_archive_index=archive_index,
                    cluster=cluster,
                )
            )

    if unexpected_tf_files:
        example = "\n  - ".join(unexpected_tf_files[:10])

        raise ValidationError(
            "Unexpected .tf files were found under "
            f"motifs_sep_by_cluster:\n  - {example}"
        )

    return records, summary_clusters


def validate_records(
    records: list[ClusterRecord],
    summary_clusters: set[str],
    meme_motifs: dict[str, str],
    expected_motifs: int,
    expected_clusters: int,
) -> Counter[str]:
    if len(records) != expected_motifs:
        raise ValidationError(
            f"Expected {expected_motifs} individual motif records, "
            f"found {len(records)}."
        )

    matrix_id_counts = Counter(
        record.matrix_id for record in records
    )

    duplicated_ids = {
        matrix_id
        for matrix_id, count in matrix_id_counts.items()
        if count > 1
    }

    if duplicated_ids:
        raise ValidationError(
            "Duplicated Matrix IDs in clustering archive: "
            f"{preview(duplicated_ids)}"
        )

    record_clusters = {
        record.jaspar_cluster for record in records
    }

    if len(record_clusters) != expected_clusters:
        raise ValidationError(
            f"Expected {expected_clusters} represented clusters, "
            f"found {len(record_clusters)}."
        )

    if len(summary_clusters) != expected_clusters:
        raise ValidationError(
            f"Expected {expected_clusters} cluster summary files, "
            f"found {len(summary_clusters)}."
        )

    if record_clusters != summary_clusters:
        missing_summaries = record_clusters - summary_clusters
        empty_summaries = summary_clusters - record_clusters

        raise ValidationError(
            "Cluster folders and summary files disagree. "
            f"Missing summaries: {preview(missing_summaries)}. "
            f"Summaries without individual motifs: "
            f"{preview(empty_summaries)}."
        )

    archive_ids = set(matrix_id_counts)
    meme_ids = set(meme_motifs)

    missing_from_archive = meme_ids - archive_ids
    extra_in_archive = archive_ids - meme_ids

    if missing_from_archive or extra_in_archive:
        raise ValidationError(
            "The Matrix IDs in the clustering archive do not exactly "
            "match the MEME file. "
            f"Missing from archive: {preview(missing_from_archive)}. "
            f"Extra in archive: {preview(extra_in_archive)}."
        )

    name_mismatches = [
        (
            record.matrix_id,
            record.motif_name,
            meme_motifs[record.matrix_id],
        )
        for record in records
        if record.motif_name != meme_motifs[record.matrix_id]
    ]

    if name_mismatches:
        examples = "; ".join(
            f"{matrix_id}: archive={archive_name!r}, "
            f"MEME={meme_name!r}"
            for matrix_id, archive_name, meme_name
            in name_mismatches[:10]
        )

        raise ValidationError(
            f"Found {len(name_mismatches)} motif-name mismatch(es). "
            f"Examples: {examples}"
        )

    return Counter(
        record.jaspar_cluster for record in records
    )


def cluster_number(cluster_name: str) -> int:
    return int(cluster_name.split("_", maxsplit=1)[1])


def write_mapping(
    path: Path,
    records: list[ClusterRecord],
    cluster_sizes: Counter[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_name(path.name + ".tmp")

    ordered_records = sorted(
        records,
        key=lambda record: (
            cluster_number(record.jaspar_cluster),
            record.matrix_id,
        ),
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "matrix_id",
                "motif_name",
                "jaspar_cluster",
                "cluster_size",
            ],
            delimiter="\t",
            lineterminator="\n",
        )

        writer.writeheader()

        for record in ordered_records:
            writer.writerow(
                {
                    "matrix_id": record.matrix_id,
                    "motif_name": record.motif_name,
                    "jaspar_cluster": record.jaspar_cluster,
                    "cluster_size": cluster_sizes[
                        record.jaspar_cluster
                    ],
                }
            )

    temporary_path.replace(path)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract and validate the official JASPAR 2026 CORE "
            "Plants motif-to-cluster mapping."
        )
    )

    parser.add_argument(
        "--archive",
        required=True,
        type=Path,
        help="Official JASPAR plant matrix-clustering ZIP.",
    )

    parser.add_argument(
        "--meme",
        required=True,
        type=Path,
        help="Matching JASPAR 2026 CORE Plants MEME file.",
    )

    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output TSV path.",
    )

    parser.add_argument(
        "--expected-motifs",
        type=int,
        default=927,
    )

    parser.add_argument(
        "--expected-clusters",
        type=int,
        default=55,
    )

    return parser


def run() -> int:
    args = build_argument_parser().parse_args()

    if not args.archive.is_file():
        raise ValidationError(
            f"Clustering archive was not found: {args.archive}"
        )

    if not args.meme.is_file():
        raise ValidationError(
            f"MEME file was not found: {args.meme}"
        )

    meme_motifs = parse_meme_file(args.meme)

    records, summary_clusters = parse_cluster_archive(
        args.archive
    )

    cluster_sizes = validate_records(
        records=records,
        summary_clusters=summary_clusters,
        meme_motifs=meme_motifs,
        expected_motifs=args.expected_motifs,
        expected_clusters=args.expected_clusters,
    )

    write_mapping(
        path=args.out,
        records=records,
        cluster_sizes=cluster_sizes,
    )

    largest_clusters = sorted(
        cluster_sizes.items(),
        key=lambda item: (-item[1], item[0]),
    )[:10]

    print("PASS: official JASPAR cluster mapping is valid.")
    print(f"Archive: {args.archive}")
    print(f"Archive SHA-256: {sha256_file(args.archive)}")
    print(f"MEME file: {args.meme}")
    print(f"MEME SHA-256: {sha256_file(args.meme)}")
    print(f"Individual motifs: {len(records)}")
    print(f"Unique Matrix IDs: {len({r.matrix_id for r in records})}")
    print(f"Official clusters: {len(cluster_sizes)}")
    print(f"Cluster summary files: {len(summary_clusters)}")
    print("Ten largest clusters:")

    for cluster, size in largest_clusters:
        print(f"  {cluster}: {size}")

    print(f"Wrote: {args.out}")

    return 0


def main() -> int:
    try:
        return run()
    except (
        ValidationError,
        OSError,
        zipfile.BadZipFile,
    ) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
